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
"""
Reward config
"""

from dataclasses import dataclass, field
from typing import Optional

from ...utils.py_functional import get_abs_path


@dataclass
class RewardConfig:
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    use_bert_reward: bool = False
    bert_reward_weight: float = 0.0
    bert_model_name: str = "sentence_bert"
    bert_checkpoint_dir: str = "bert_reward_model"
    bert_training_epochs: int = 1
    bert_batch_size: int = 128
    bert_learning_rate: float = 5e-6
    bert_train_interval: int = 10
    use_bert_service: bool = True
    bert_service_url: str = "http://localhost:8008"
    bert_service_timeout: int = 240
    bert_service_max_retries: int = 3
    bert_service_fallback_to_local: bool = True
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            self.reward_function = get_abs_path(self.reward_function, prompt="Reward function")
