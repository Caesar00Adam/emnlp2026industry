# 🚀 Thinking with Adversarial Reward in LMM (TAR-LMM) for Multimodal Reasoning Training

TAR-LMM is a multimodal reinforcement learning training framework specifically designed for the financial domain, supporting joint training of text and images, particularly suitable for financial chart understanding, multimodal question answering, and other tasks.

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Installation Guide](#-installation-guide)
- [Quick Start](#-quick-start)
- [Training Process](#-training-process)
- [BERT Reward Service](#-bert-reward-service)
- [Configuration Guide](#-configuration-guide)

## 🔍 Project Overview

TAR-LMM implements a complete two-stage training process aimed at enhancing multimodal model performance in the financial domain:

1. **Stage 1**: Basic reinforcement learning training to optimize the model's fundamental capabilities
2. **Stage 2**: Advanced training with BERT rewards and image selection features to enhance reasoning quality

## 📦 Installation Guide

### System Requirements

- Python 3.9+
- CUDA 11.8+ (recommended)
- At least 32GB memory
- NVIDIA GPU support (A100/H100 recommended)

### Dependency Installation

1. **Install Dependencies**

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install package in development mode (recommended)
pip install -e .

# Or install with additional development tools
pip install -e ".[dev]"
```

### Main Dependencies

```
accelerate         # Distributed training acceleration
flash-attn>=2.4.3  # Efficient attention mechanism
transformers>=4.51.0  # Hugging Face model support
vllm>=0.7.3        # Efficient inference engine
ray[default]       # Distributed computing framework
wandb              # Experiment tracking and visualization
```

## 🗂️ Directory Structure

```bash
.
├── verl/                      # Main package directory
│   ├── workers/               # Worker implementations for different components
│   ├── trainer/               # Training orchestration code
│   ├── models/                # Model implementations and adapters
│   ├── utils/                 # Utility functions and helpers
│   └── protocol.py            # Data protocol definitions
├── examples/                  # Example configurations and prompts
│   ├── format_prompt/         # Prompt templates
│   ├── score_function/        # Reward function implementations
│   ├── stage1_config.yaml     # Stage 1 training configuration
│   └── stage2_config.yaml     # Stage 2 training configuration with BERT rewards
├── scripts/                   # Utility scripts
├── stage_1_2B.sh              # Script for running stage 1 training
├── stage_2_2B.sh              # Script for running stage 2 training
├── requirements.txt           # Python dependencies
└── setup.py                   # Package installation script
```

## 🚀 Quick Start

After completing the installation, you can start training by following these steps:

1. **Prepare the Model**

   Ensure you have access to pre-trained models (such as Qwen2.5-7B-Instruct or Qwen3-VL-2B-thinking)

2. **Configure Training Parameters**

   Modify `examples/stage1_config.yaml` and `examples/stage2_config.yaml` according to your needs

3. **Start Training**

   Follow the instructions in the [Training Process](#-training-process) section below to run the training scripts

## 🔄 Training Process

The training process is divided into two stages:

### 🔹 Stage 1: Basic Reinforcement Learning Training

```bash
# Edit stage_1_3B.sh to set your model path
MODEL_PATH=/path/to/your/model

# Run stage 1 training
bash stage_1_2B.sh
```

### 🔹 Stage 2: Advanced Reinforcement Learning Training with BERT Rewards

```bash
# Edit stage_2_3B.sh to set your model path
MODEL_PATH=/path/to/your/model

# Run stage 2 training
bash stage_2_2B.sh
```

## 🧠 BERT Reward Service

The BERT reward model is a key component of stage 2 training and can be started and used through the following steps:

### 1. Start the BERT Reward Service

```bash
# Navigate to the project directory
cd verl/workers/reward

# Start the BERT service (local mode)
python start_bert_service.py --host 0.0.0.0 --port 8000 --model fin_bert --checkpoint bert_reward_model
```

Parameter description:
- `--host`: Server host address, default is 0.0.0.0
- `--port`: Server port, default is 8000
- `--model`: BERT model name or path, default is sentence_bert
- `--checkpoint`: Model checkpoint save directory, default is bert_reward_model

### 2. Configure BERT Reward Parameters

Configure BERT reward-related parameters in `examples/stage2_config.yaml`:

```yaml
worker:
  reward:
    use_bert_reward: true                  # Enable BERT reward
    bert_reward_weight: 0.15               # BERT reward weight
    bert_model_name: fin_bert         # BERT model name
    bert_checkpoint_dir: bert_reward_model # BERT checkpoint directory
    bert_training_epochs: 1                # BERT training epochs
    bert_batch_size: 128                   # BERT training batch size
    bert_learning_rate: 5e-6               # BERT training learning rate
    bert_train_interval: 10                # BERT training interval
    
    # Using BERT service (optional)
    use_bert_service: true                 # Enable BERT service
    bert_service_url: "http://localhost:8000"  # BERT service URL
    bert_service_timeout: 10               # Request timeout (seconds)
    bert_service_max_retries: 3            # Maximum retry attempts
    bert_service_fallback_to_local: true   # Fall back to local model if service fails
```

### 3. BERT Reward Model Training

The BERT reward model is automatically trained during the training process, with training data sourced from:
- Thinking processes generated by the model (marked with `<think>...</think>`)
- Labels generated based on answer accuracy (correct/incorrect)

The training process is triggered under the following conditions:
- Every `bert_train_interval` iterations
- When sufficient training samples have accumulated
- When training is manually triggered

### 4. Monitor the BERT Reward Service

After starting the service, you can monitor it using the following methods:

```bash
# View logs
tail -f bert_service.log

# Check service status
curl http://localhost:8000/health
```

## 📝 Configuration Guide

Both stages use YAML configuration files:

- `examples/stage1_config.yaml`: Stage 1 training configuration
- `examples/stage2_config.yaml`: Stage 2 training configuration with BERT rewards

Key configuration sections include:

### 1. Data Configuration

```yaml
data:
  train_files: math12k@train  # Training dataset
  val_files: math12k@test     # Validation dataset
  prompt_key: problem         # Prompt field name
  answer_key: answer          # Answer field name
  image_key: images           # Image field name
  max_prompt_length: 2048     # Maximum prompt length
  max_response_length: 2048   # Maximum response length
  format_prompt: ./examples/format_prompt/format_finlmm.jinja  # Prompt template
```

### 2. Algorithm Settings

```yaml
algorithm:
  adv_estimator: grpo         # Reinforcement learning algorithm
  use_kl_loss: true           # Whether to use KL loss
  kl_penalty: low_var_kl      # KL penalty type
  kl_coef: 1.0e-3             # KL coefficient
```

### 3. Worker Configuration

```yaml
worker:
  actor:
    global_batch_size: 128    # Global batch size
    model:
      model_path: Qwen/Qwen3-vl-2B-thinking  # Model path
  rollout:
    temperature: 1.0          # Sampling temperature
    top_p: 0.99               # Top-p sampling
  reward:
    reward_type: function     # Reward type
    score_function: ./examples/score_function/sf.py:compute_score  # Reward function
```



