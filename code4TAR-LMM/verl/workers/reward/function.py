# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import re
import sys
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Tuple, TypedDict

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict, total=False):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]
    image_selection: Optional[float]
    thinking_length: Optional[float]
    bert: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]
BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs, response_lengths = self._build_reward_inputs(data)
        self._attach_bert_scores(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, reward_input in enumerate(reward_inputs):
            score = self.reward_fn(reward_input)
            reward_tensor[i, response_lengths[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs, response_lengths = self._build_reward_inputs(data)
        self._attach_bert_scores(reward_inputs)
        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_lengths[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward plus optional dynamic BERT reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer
        self.bert_client = None
        self.bert_trainer = None
        if self.config.use_bert_reward:
            self._init_bert_reward()

    def _init_bert_reward(self) -> None:
        if self.config.use_bert_service:
            try:
                from .bert_reward_client import BertRewardClient

                self.bert_client = BertRewardClient(
                    api_url=self.config.bert_service_url,
                    timeout=self.config.bert_service_timeout,
                    max_retries=self.config.bert_service_max_retries,
                    fallback_to_local=self.config.bert_service_fallback_to_local,
                )
                return
            except Exception as e:
                print(f"Failed to initialize BERT reward client: {e}")

        if self.config.bert_service_fallback_to_local:
            try:
                from .bert_reward_trainer import BertRewardTrainer

                self.bert_trainer = BertRewardTrainer(
                    model_name=self.config.bert_model_name,
                    checkpoint_dir=self.config.bert_checkpoint_dir,
                )
                self.bert_trainer.load_model()
            except Exception as e:
                print(f"Failed to initialize local BERT reward model: {e}")
                self.bert_trainer = None

    def _extract_thinking_process(self, text: str) -> str:
        try:
            thinking_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            return thinking_match.group(1).strip() if thinking_match else ""
        except Exception:
            return ""

    def _to_scalar(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.item() if value.shape == () else value
        return value

    def _build_reward_inputs(self, data: DataProto) -> tuple[list[RewardInput], list[int]]:
        reward_inputs = []
        response_lengths = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_input: RewardInput = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            for key, value in data.non_tensor_batch.items():
                if i < len(value):
                    reward_input[key] = self._to_scalar(value[i])

            reward_inputs.append(reward_input)
            response_lengths.append(cur_response_length)

        return reward_inputs, response_lengths

    def _attach_bert_scores(self, reward_inputs: list[RewardInput]) -> None:
        if not self.config.use_bert_reward or (self.bert_client is None and self.bert_trainer is None):
            return

        questions, thinking_processes, indices = [], [], []
        for i, reward_input in enumerate(reward_inputs):
            thinking_process = self._extract_thinking_process(reward_input["response"])
            if thinking_process:
                questions.append(str(reward_input.get("problem", "")))
                thinking_processes.append(thinking_process)
                indices.append(i)

        if not thinking_processes:
            return

        try:
            if self.bert_client is not None:
                bert_scores = self.bert_client.predict_quality(questions, thinking_processes)
            else:
                bert_scores = self.bert_trainer.predict_proba(questions, thinking_processes)
        except Exception as e:
            print(f"BERT reward scoring failed: {e}")
            return

        for index, score in zip(indices, bert_scores):
            reward_inputs[index]["bert_score"] = float(score)
            reward_inputs[index]["bert_reward_weight"] = self.config.bert_reward_weight

    def train_bert_model(self, data: DataProto) -> bool:
        if not self.config.use_bert_reward or (self.bert_client is None and self.bert_trainer is None):
            return False

        reward_inputs, _ = self._build_reward_inputs(data)
        questions, thinking_processes, labels = [], [], []
        for reward_input in reward_inputs:
            thinking_process = self._extract_thinking_process(reward_input["response"])
            if not thinking_process:
                continue

            label_score = self.reward_fn(
                {
                    **reward_input,
                    "bert_score": None,
                    "bert_reward_weight": 0.0,
                }
            )
            label = 1 if label_score.get("accuracy", 0.0) > 0.5 else 0
            questions.append(str(reward_input.get("problem", "")))
            thinking_processes.append(thinking_process)
            labels.append(label)

        if not questions:
            print("No valid samples for BERT reward model training.")
            return False

        try:
            if self.bert_client is not None:
                return self.bert_client.train_bert_model(
                    questions=questions,
                    thinking_processes=thinking_processes,
                    labels=labels,
                    batch_size=self.config.bert_batch_size,
                    epochs=self.config.bert_training_epochs,
                    learning_rate=self.config.bert_learning_rate,
                )

            self.bert_trainer.train(
                questions=questions,
                thinking_processes=thinking_processes,
                labels=labels,
                batch_size=self.config.bert_batch_size,
                epochs=self.config.bert_training_epochs,
                learning_rate=self.config.bert_learning_rate,
            )
            return True
        except Exception as e:
            print(f"BERT reward model training failed: {e}")
            return False

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
