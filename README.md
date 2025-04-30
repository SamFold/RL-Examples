# Reinforcement Learning with Human Feedback (RLHF)

A package for training language models to generate high-quality text using Reinforcement Learning from Human Feedback (RLHF) with GPT-4o as a reward model.

## Overview

This package provides tools for fine-tuning language models to generate text that aligns with human preferences using a form of Reinforcement Learning from Human Feedback (RLHF). It uses GPT-4o as a reward model to evaluate generated text and guide the training process.

The implementation uses the TRL (Transformer Reinforcement Learning) library and the PPO (Proximal Policy Optimization) algorithm for policy optimization.

> **Note about Apple Silicon Support**: This project includes support for Apple Silicon GPUs through PyTorch's MPS (Metal Performance Shaders) backend. While most operations should work on MPS, some PyTorch operations may not be fully supported. If you encounter issues with MPS, you can fall back to CPU by using the `--device cpu` option.

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

### Training a Model

To train a model using GPT-4o as a reward model:

```bash
python main.py --model_name lvwerra/gpt2-imdb --openai_api_key YOUR_API_KEY
```

### Inference with a Trained Model

```bash
python main.py --inference --model_path path/to/trained/model --openai_api_key YOUR_API_KEY
```

## Command Line Arguments

- `--model_name`: The name of the model to train (default: "lvwerra/gpt2-imdb")
- `--max_epochs`: Maximum number of epochs to train (default: 200)
- `--batch_size`: Training batch size (default: 16)
- `--save_freq`: Frequency of saving checkpoints (default: 50)
- `--openai_api_key`: OpenAI API key for GPT-4o reward model (required)
- `--output_dir`: Directory to save outputs (default: "trainer_output")
- `--device`: Device to use for training ("cuda", "mps", or "cpu")
- `--no_exploration`: Disable exploration (entropy bonus) during training
- `--inference`: Run in inference mode instead of training
- `--model_path`: Path to a trained model for inference

## Key Features

- **GPT-4o Reward Model**: Uses OpenAI's GPT-4o as a reward function to evaluate text quality
- **Separate Policy/Value Optimization**: Uses distinct optimization steps for policy and value networks
- **Reward Normalization**: Implements adaptive reward normalization for training stability
- **Reference Model Updates**: Uses Exponential Moving Average (EMA) for stable reference model updates
- **Generalized Advantage Estimation (GAE)**: Implements GAE for improved policy gradients
- **Entropy-Based Exploration**: Optional entropy bonus for exploration during training
- **KL Divergence Control**: Prevents policy from diverging too far from reference model
- **MPS Support**: Support for Apple Silicon GPUs through PyTorch's MPS backend

## Project Structure

```
reinforment-learning-with-human-feedback/
├── main.py                     # Main script for training and inference
├── prepare_data.py             # Data preparation utilities
├── sentiment_rlhf/
│   ├── data/                   # Dataset utilities
│   ├── models/                 # Model loading utilities
│   │   └── model_loader.py     # Model loading implementation
│   ├── training/               # Training components
│   │   ├── ppo_trainer.py      # PPO trainer implementation
│   │   └── reward_model.py     # Reward model implementations
│   └── utils/                  # Utility functions
│       └── config.py           # Configuration utilities
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

## Examples

### Basic Training

```bash
python main.py --model_name lvwerra/gpt2-imdb --max_epochs 100 --output_dir my_trained_model --openai_api_key YOUR_API_KEY
```

### Using Apple Silicon (MPS) Acceleration

```bash
python main.py --model_name lvwerra/gpt2-imdb --device mps --openai_api_key YOUR_API_KEY
```

### Training Without Exploration (Entropy Bonus)

```bash
python main.py --model_name lvwerra/gpt2-imdb --no_exploration --openai_api_key YOUR_API_KEY
```

### Testing a Trained Model

```bash
python main.py --inference --model_path my_trained_model --openai_api_key YOUR_API_KEY
```

## Citations

This implementation is based on the following resources:

- [TRL - Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [InstructGPT: Training language models to follow instructions](https://arxiv.org/abs/2203.02155)

## License

MIT License