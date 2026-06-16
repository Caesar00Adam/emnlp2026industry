import os
import sys
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bert_service")

def main():
    """Main function to start BERT service"""
    # Ensure verl package can be imported
    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root directory (up three levels: verl/workers/reward -> verl/workers -> verl -> root)
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    # Add to Python path at the beginning
    sys.path.insert(0, project_root)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start BERT reward model service")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8008, help="Server port")
    parser.add_argument("--model", default="sentence_bert", help="BERT model name or path")
    parser.add_argument("--checkpoint", default="bert_reward_model", help="Model checkpoint save directory")
    args = parser.parse_args()
    
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            logger.info(f"GPU available: {gpu_available}, Device count: {device_count}, Devices: {device_names}")
            
            # Print detailed CUDA information
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"Available CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        else:
            logger.warning("No GPU available, service will run on CPU, which may be slow")
    except ImportError:
        logger.warning("Could not import torch to check GPU availability")
    
    # Check model directory
    if not os.path.exists(args.model):
        logger.warning(f"Model directory does not exist: {args.model}, may need to download or create")
    
    # Check checkpoint directory
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint directory does not exist: {checkpoint_dir}, will create new directory")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Start service
    logger.info(f"Starting BERT reward model service, host: {args.host}, port: {args.port}")
    logger.info(f"Using model: {args.model}, checkpoint directory: {checkpoint_dir}")
    
    try:
        # Ensure these imports happen after sys.path is updated
        import uvicorn
        
        # Absolute import for BERT trainer and FastAPI app
        from verl.workers.reward.bert_reward_trainer import BertRewardTrainer
        from verl.workers.reward.bert_reward_server import app
        
        # Actively initialize BERT trainer
        trainer = BertRewardTrainer(
            model_name=args.model,
            checkpoint_dir=checkpoint_dir
        )
        
        # Try to load model
        logger.info("Attempting to load BERT model...")
        trainer.load_model()
        
        # Set global variable
        import verl.workers.reward.bert_reward_server as server_module
        server_module.bert_trainer = trainer
        logger.info("BERT model loaded and set as global variable")
        
        # Start service directly with uvicorn
        logger.info(f"Starting FastAPI service: http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        
    except ImportError as e:
        logger.error(f"Import error, please ensure all dependencies are installed: {e}")
        logger.error(f"Current Python path: {sys.path}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
