# Reinforcement Learning with Human Feedback (RLHF)

A package for training language models to generate high-quality text using Reinforcement Learning from Human Feedback (RLHF) with GPT-4o as a reward model.

## Overview

This package provides tools for fine-tuning language models to generate text that aligns with human preferences using a form of Reinforcement Learning from Human Feedback (RLHF). It uses GPT-4o as a reward model to evaluate generated text and guide the training process.

The implementation uses the TRL (Transformer Reinforcement Learning) library and the PPO (Proximal Policy Optimization) algorithm for policy optimization.

> **Note about Cross-Platform Support**: This project supports multiple hardware platforms:
> - **NVIDIA GPUs**: Automatically detected and used via CUDA with optional optimizations for H100/A100
> - **Apple Silicon**: GPU acceleration via Metal Performance Shaders (MPS)
> - **CPU**: Fallback option for all platforms

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.25+
- TRL 0.17.0+
- OpenAI API key (for GPT-4o reward model)

### Installing

```bash
# Clone the repository
git clone https://github.com/SamFold/RL-Examples.git
cd RL-Examples

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start with Convenience Scripts

**For NVIDIA GPUs (H100/A100):**
```bash
# Edit H100_RUN_SCRIPT.sh to add your OpenAI API key
chmod +x H100_RUN_SCRIPT.sh  # Make executable
./H100_RUN_SCRIPT.sh
```

**For Apple Silicon (M1/M2/M3):**
```bash
# Edit MPS_RUN_SCRIPT.sh to add your OpenAI API key
chmod +x MPS_RUN_SCRIPT.sh  # Make executable
./MPS_RUN_SCRIPT.sh
```

### Manual Training

To train a model using GPT-4o as a reward model:

```bash
# Basic training command
python main.py --model_name lvwerra/gpt2-imdb --openai_api_key YOUR_API_KEY

# With device-specific optimizations enabled (for H100/A100 or Apple Silicon)
python main.py --model_name lvwerra/gpt2-imdb --optimize_device --openai_api_key YOUR_API_KEY
```

### Inference with a Trained Model

```bash
python main.py --inference --model_path path/to/trained/model --openai_api_key YOUR_API_KEY
```

## Command Line Arguments

- `--model_name`: Model to train (default: "lvwerra/gpt2-imdb")
- `--max_epochs`: Maximum training epochs (default: 200)
- `--batch_size`: Global batch size for all operations (default: 32)
- `--save_freq`: Checkpoint saving frequency (default: 50)
- `--openai_api_key`: OpenAI API key for GPT-4o reward model (required)
- `--parallel_reward`: Use parallel reward model for faster evaluation
- `--output_dir`: Directory for outputs (default: "trainer_output")
- `--device`: Training device ("cuda", "mps", or "cpu")
- `--optimize_device`: Enable hardware-specific optimizations for CUDA/MPS
- `--no_exploration`: Disable exploration (entropy bonus)
- `--inference`: Run in inference mode
- `--model_path`: Path to trained model for inference

## Key Features

- **GPT-4o Reward Model**: Uses OpenAI's GPT-4o as a reward function to evaluate text quality
- **Parallel Reward Processing**: Supports parallel OpenAI API calls for 10-20x speedup in reward computation
- **Separate Policy/Value Optimization**: Uses distinct optimization steps for policy and value networks
- **Reward Normalization**: Implements adaptive reward normalization for training stability
- **Reference Model Updates**: Uses Exponential Moving Average (EMA) for stable reference model updates
- **Generalized Advantage Estimation (GAE)**: Implements GAE for improved policy gradients
- **Entropy-Based Exploration**: Optional entropy bonus for exploration during training
- **KL Divergence Control**: Prevents policy from diverging too far from reference model
- **Hardware Optimizations**: Platform-specific optimizations for NVIDIA H100/A100 and Apple Silicon
- **Language Model Loss**: Configurable auxiliary language modeling loss component

## Project Structure

```
reinforment-learning-with-human-feedback/
├── main.py                     # Main script for training and inference
├── prepare_data.py             # Data preparation utilities
├── H100_RUN_SCRIPT.sh          # Convenience script for NVIDIA GPU training
├── MPS_RUN_SCRIPT.sh           # Convenience script for Apple Silicon
├── sentiment_rlhf/
│   ├── data/                   # Dataset utilities
│   ├── models/                 # Model loading utilities
│   │   └── model_loader.py     # Model loading implementation
│   ├── training/               # Training components
│   │   ├── ppo_trainer.py      # PPO trainer implementation
│   │   ├── reward_model.py     # Sequential reward model implementation
│   │   └── parallel_reward_model.py # Parallel reward model implementation
│   └── utils/                  # Utility functions
│       └── config.py           # Configuration utilities
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

## Examples

### Basic Training

```bash
# Basic training with default settings
python main.py --openai_api_key YOUR_API_KEY
```

### GPU Training Examples

**NVIDIA GPU with optimizations (best for H100/A100):**
```bash
python main.py --model_name lvwerra/gpt2-imdb --batch_size 32 \
  --max_epochs 20 --device cuda --optimize_device \
  --openai_api_key YOUR_API_KEY
```

**Apple Silicon with optimizations:**
```bash
python main.py --model_name lvwerra/gpt2-imdb --batch_size 32 \
  --max_epochs 20 --device mps --optimize_device \
  --openai_api_key YOUR_API_KEY
```

### Training with Parallel Reward Processing

```bash
python main.py --model_name lvwerra/gpt2-imdb --batch_size 32 \
  --parallel_reward --openai_api_key YOUR_API_KEY
```

### Training Without Exploration

```bash
python main.py --model_name lvwerra/gpt2-imdb --no_exploration \
  --openai_api_key YOUR_API_KEY
```

### Testing a Trained Model

```bash
python main.py --inference --model_path trainer_output/best-epoch-X \
  --openai_api_key YOUR_API_KEY
```

## Cloud Deployment

For faster training on cloud platforms:
- **NVIDIA GPUs**: See `NEBIUS_DEPLOYMENT_GUIDE.md` for detailed setup on Nebius Cloud
- **Google Cloud**: See `GCP_DEPLOYMENT_GUIDE.md` for GCP deployment instructions

## Citations

This implementation is based on the following resources:

- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

## License

MIT License