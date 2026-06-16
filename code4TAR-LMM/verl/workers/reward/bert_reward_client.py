import json
import time
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from requests.exceptions import RequestException, Timeout, ConnectionError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertRewardClient:
    """BERT reward model client, for communication with FastAPI server"""
    
    def __init__(
        self, 
        api_url: str = "http://localhost:8008",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 2,
        fallback_to_local: bool = True
    ):
        """
        Initialize BERT reward client
        
        Args:
            api_url: URL address of FastAPI server
            timeout: request timeout (seconds)
            max_retries: maximum number of retries
            retry_delay: retry delay time (seconds)
            fallback_to_local: whether to fallback to local model when service is unavailable
        """
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_to_local = fallback_to_local
        self.session = requests.Session()
        self.local_model = None
        
        # check if service is available
        self.service_available = self._check_service()
        
        if not self.service_available and self.fallback_to_local:
            self._init_local_model()
    
    def _check_service(self) -> bool:
        """Check if BERT service is available"""
        try:
            response = self.session.get(f"{self.api_url}/status", timeout=self.timeout)
            if response.status_code == 200:
                status_data = response.json()
                logger.info(f"BERT service status: {status_data}")
                return True
            return False
        except Exception as e:
            logger.warning(f"BERT service is unavailable: {str(e)}")
            return False
    
    def _init_local_model(self):
        """Initialize local BERT model (fallback mode)"""
        try:
            from .bert_reward_trainer import BertRewardTrainer
            logger.info("BERT service is unavailable, initialize local model as fallback")
            self.local_model = BertRewardTrainer()
            # try to load existing model
            self.local_model.load_model()
            logger.info("Local BERT model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize local BERT model: {str(e)}")
            self.local_model = None
    
    def _request_with_retry(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Any]:
        """
        Send request to server and handle retry logic
        
        Args:
            method: request method, "get" or "post"
            endpoint: API endpoint path
            data: request data
            
        Returns:
            Tuple[bool, Any]: (success, response data)
        """
        url = f"{self.api_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                if method.lower() == "get":
                    response = self.session.get(url, timeout=self.timeout)
                else:  # post
                    response = self.session.post(url, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    return True, response.json()
                else:
                    error_msg = f"Request failed: HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "message" in error_data:
                            error_msg += f" - {error_data['message']}"
                    except:
                        pass
                    logger.warning(f"{error_msg} (尝试 {attempt+1}/{self.max_retries})")
            
            except (ConnectionError, Timeout) as e:
                logger.warning(f"Connection error: {str(e)} (attempt {attempt+1}/{self.max_retries})")
            except RequestException as e:
                logger.warning(f"Request error: {str(e)} (attempt {attempt+1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)} (attempt {attempt+1}/{self.max_retries})")
            
            # if not the last attempt, wait and retry
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        # all retries failed
        return False, None
    
    def train_bert_model(
        self, 
        questions: List[str], 
        thinking_processes: List[str], 
        labels: List[int],
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> bool:
        """
        Train BERT model
        
        Args:
            questions: list of questions
            thinking_processes: list of thinking processes
            labels: list of labels (0/1)
            batch_size: batch size
            epochs: number of training epochs
            learning_rate: learning rate
            
        Returns:
            bool: whether training is successful
        """
        if not questions or not thinking_processes or not labels:
            logger.warning("Training data is empty")
            return False
        
        if len(questions) != len(thinking_processes) or len(questions) != len(labels):
            logger.warning("Training data length mismatch")
            return False
        
        # check if service is available
        if not self.service_available:
            self.service_available = self._check_service()
        
        # if service is available, train via API
        if self.service_available:
            train_data = {
                "questions": questions,
                "thinking_processes": thinking_processes,
                "labels": labels
            }
            
            # add optional parameters
            if batch_size is not None:
                train_data["batch_size"] = batch_size
            if epochs is not None:
                train_data["epochs"] = epochs
            if learning_rate is not None:
                train_data["learning_rate"] = learning_rate
            
            success, response = self._request_with_retry("post", "train", train_data)
            if success:
                logger.info("Successfully started BERT model training")
                return True
            else:
                logger.error("Failed to start BERT model training via API")
        
        # if service is unavailable and fallback to local model
        if self.fallback_to_local and self.local_model:
            try:
                logger.info("Using local model for BERT training")
                self.local_model.train(
                    questions=questions,
                    thinking_processes=thinking_processes,
                    labels=labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                return True
            except Exception as e:
                logger.error(f"Failed to train local BERT model: {str(e)}")
        
        return False
    
    def predict_quality(
        self, 
        questions: List[str], 
        thinking_processes: List[str],
        batch_size: Optional[int] = None
    ) -> List[float]:
        """
        Predict the quality of thinking processes
        
        Args:
            questions: list of questions
            thinking_processes: list of thinking processes
            batch_size: batch size
            
        Returns:
            List[float]: list of predicted scores (0~1 float), default value if prediction fails
        """
        default_scores = [0.5] * len(questions)  # default neutral score
        
        if not questions or not thinking_processes:
            logger.warning("Prediction data is empty")
            return default_scores
        
        if len(questions) != len(thinking_processes):
            logger.warning("Prediction data length mismatch")
            return default_scores
        
        # check if service is available
        if not self.service_available:
            self.service_available = self._check_service()
        
        # if service is available, predict via API
        if self.service_available:
            predict_data = {
                "questions": questions,
                "thinking_processes": thinking_processes
            }
            
            if batch_size is not None:
                predict_data["batch_size"] = batch_size
            
            success, response = self._request_with_retry("post", "predict", predict_data)
            if success and "scores" in response:
                return response["scores"]
            else:
                logger.error("Failed to get prediction results via API")
        
        # if service is unavailable and fallback to local model
        if self.fallback_to_local and self.local_model:
            try:
                logger.info("Using local model for BERT prediction")
                scores = self.local_model.predict_proba(
                    questions=questions,
                    thinking_processes=thinking_processes,
                    batch_size=batch_size
                )
                return scores
            except Exception as e:
                logger.error(f"Failed to predict local BERT model: {str(e)}")
        
        return default_scores
    
    def extract_thinking_process(self, text: str) -> str:
        """
        Extract thinking process from text
        
        Args:
            text: text containing thinking process
            
        Returns:
            str: extracted thinking process
        """
        import re
        try:
            thinking_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            return thinking_match.group(1).strip() if thinking_match else ""
        except Exception:
            return ""
            
    def train_from_data_protos(self, data_protos: List):
        """
        Extract training data from DataProto object list and train BERT model
        
        Args:
            data_protos: list of DataProto objects
            
        Returns:
            bool: whether training is successful
        """
        if not data_protos:
            logger.warning("No training data provided")
            return False
        
        # collect training data
        questions = []
        thinking_processes = []
        labels = []
        
        for data_proto in data_protos:
            try:
                # check if required fields exist
                if ("problem" not in data_proto.non_tensor_batch or 
                    "ground_truth" not in data_proto.non_tensor_batch or 
                    "response" not in data_proto.non_tensor_batch):
                    continue
                
                # get original question
                problem_data = data_proto.non_tensor_batch["problem"]
                if hasattr(problem_data, "size") and problem_data.size > 0:
                    question = problem_data[0]
                else:
                    question = problem_data
                
                # get response content
                response_data = data_proto.non_tensor_batch["response"]
                if hasattr(response_data, "size") and response_data.size > 0:
                    response_str = response_data[0]
                else:
                    response_str = response_data
                
                # extract thinking process
                thinking_process = self.extract_thinking_process(response_str)
                if not thinking_process:
                    continue
                
                # get standard answer
                ground_truth_data = data_proto.non_tensor_batch["ground_truth"]
                if hasattr(ground_truth_data, "size") and ground_truth_data.size > 0:
                    ground_truth = ground_truth_data[0]
                else:
                    ground_truth = ground_truth_data
                
                if not ground_truth:
                    continue
                
                # judge quality label (simple logic, can be adjusted according to actual needs)
                # here simplified processing, assuming 1 is high quality, 0 is low quality
                label = 1  # default to high quality
                
                # add to training data
                questions.append(question)
                thinking_processes.append(thinking_process)
                labels.append(label)
            except Exception as e:
                logger.error(f"Error processing data sample: {str(e)}")
                continue
        
        # check if there is enough training data
        if len(questions) == 0:
            logger.warning("No valid training data extracted")
            return False
        
        # train model
        logger.info(f"Starting to train BERT reward model, using {len(questions)} samples")
        return self.train_bert_model(questions, thinking_processes, labels) 