import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
import uvicorn
import logging
from threading import Lock

# import BERT reward model trainer
from verl.workers.reward.bert_reward_trainer import BertRewardTrainer

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create FastAPI application
app = FastAPI(title="BERT Reward Model Service")

# define data models
class TrainingData(BaseModel):
    questions: List[str]
    thinking_processes: List[str]
    labels: List[int]
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None

class PredictionData(BaseModel):
    questions: List[str]
    thinking_processes: List[str]
    batch_size: Optional[int] = None

class ServiceStatus(BaseModel):
    status: str
    model_loaded: bool
    is_training: bool
    gpu_available: bool
    device: str

# global variables
bert_trainer = None
training_lock = Lock()
is_training = False

# initialize BERT trainer
def initialize_trainer(model_name="sentence_bert", checkpoint_dir="bert_reward_model"):
    global bert_trainer
    if bert_trainer is None:
        try:
            logger.info(f"Initialize BERT trainer, model name: {model_name}, checkpoint directory: {checkpoint_dir}")
            bert_trainer = BertRewardTrainer(
                model_name=model_name,
                checkpoint_dir=checkpoint_dir
            )
            # try to load existing model
            bert_trainer.load_model()
            logger.info("BERT trainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize BERT trainer: {str(e)}")
            raise

# initialize BERT trainer when server starts
@app.on_event("startup")
async def startup_event():
    # initialize BERT trainer in async context
    try:
        initialize_trainer()
    except Exception as e:
        logger.error(f"Failed to initialize BERT trainer when server starts: {str(e)}")

# get service status
@app.get("/status", response_model=ServiceStatus)
async def get_status():
    global bert_trainer, is_training
    model_loaded = bert_trainer is not None
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available and bert_trainer and str(next(bert_trainer.model.parameters()).device) == "cuda" else "cpu"
    
    return ServiceStatus(
        status="running",
        model_loaded=model_loaded,
        is_training=is_training,
        gpu_available=gpu_available,
        device=device
    )

# train model
@app.post("/train")
async def train_model(data: TrainingData, background_tasks: BackgroundTasks):
    global bert_trainer, is_training, training_lock
    
    # check if initialized
    if bert_trainer is None:
        try:
            initialize_trainer()
        except Exception as e:
            return {"status": "error", "message": f"Failed to initialize BERT trainer: {str(e)}"}
    
    # check if training is already in progress
    if is_training:
        return {"status": "busy", "message": "Training task is already in progress"}
    
    # check data validity
    if len(data.questions) == 0 or len(data.thinking_processes) == 0 or len(data.labels) == 0:
        return {"status": "error", "message": "Training data is empty"}
    
    if len(data.questions) != len(data.thinking_processes) or len(data.questions) != len(data.labels):
        return {"status": "error", "message": "Training data length mismatch"}
    
    # define background training function
    def run_training():
        global is_training
        with training_lock:
            is_training = True
            try:
                logger.info(f"Start BERT model training, sample number: {len(data.questions)}")
                bert_trainer.train(
                    questions=data.questions,
                    thinking_processes=data.thinking_processes,
                    labels=data.labels,
                    batch_size=data.batch_size,
                    epochs=data.epochs,
                    learning_rate=data.learning_rate
                )
                logger.info("BERT model training completed")
            except Exception as e:
                logger.error(f"BERT model training failed: {str(e)}")
            finally:
                is_training = False
    
    # run training in background
    background_tasks.add_task(run_training)
    return {"status": "started", "message": "Training task started"}

# predict
@app.post("/predict")
async def predict(data: PredictionData):
    global bert_trainer
    
    # check if initialized
    if bert_trainer is None:
        try:
            initialize_trainer()
        except Exception as e:
            return {"status": "error", "message": f"Failed to initialize BERT trainer: {str(e)}"}
    
    # check data validity
    if len(data.questions) == 0 or len(data.thinking_processes) == 0:
        return {"status": "error", "message": "Prediction data is empty"}
    
    if len(data.questions) != len(data.thinking_processes):
        return {"status": "error", "message": "Prediction data length mismatch"}
    
    try:
        logger.info(f"Execute BERT model prediction, sample number: {len(data.questions)}")
        scores = bert_trainer.predict_proba(
            questions=data.questions,
            thinking_processes=data.thinking_processes,
            batch_size=data.batch_size
        )
        logger.info("BERT model prediction completed")
        return {"status": "success", "scores": scores}
    except Exception as e:
        logger.error(f"BERT model prediction failed: {str(e)}")
        return {"status": "error", "message": f"Prediction failed: {str(e)}"}

# entry point for running server directly
def run_server(host="0.0.0.0", port=8008, model_name="sentence_bert", checkpoint_dir="bert_reward_model"):
    """Start BERT reward model service"""
    try:
        # initialize BERT trainer
        global bert_trainer
        bert_trainer = BertRewardTrainer(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir
        )
        # try to load existing model
        bert_trainer.load_model()
        
        # avoid local variable name conflict with module name - do not define variables with the same name as the module
        # start server
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        # use variable name e instead of os.error
        logger.error(f"Failed to start server: {str(e)}")
        # propagate exception for better debugging
        raise

# independent entry point, avoid using code in __main__ directly
def main():
    """Command line entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="BERT reward model FastAPI service")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8008, help="Server port")
    parser.add_argument("--model", default="sentence_bert", help="BERT model name or path")
    parser.add_argument("--checkpoint", default="bert_reward_model", help="Model checkpoint save directory")
    args = parser.parse_args()
    
    # run server
    run_server(host=args.host, port=args.port, model_name=args.model, checkpoint_dir=args.checkpoint)

if __name__ == "__main__":
    main() 
