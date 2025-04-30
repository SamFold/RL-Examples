#!/bin/bash

# Script to run RLHF training on Nebius with H100 GPU

# Check if CUDA is available
nvidia-smi

# Activate virtual environment if needed
# source rlhf_env/bin/activate

# Set environment variables
export OPENAI_API_KEY="YOUR_API_KEY_HERE"

# Run with optimized parameters for H100
python main.py \
  --model_name lvwerra/gpt2-imdb \
  --batch_size 64 \
  --max_epochs 20 \
  --device cuda \
  --optimize_device \
  --mixed_precision \
  --precision_dtype bfloat16 \
  --openai_api_key $OPENAI_API_KEY \
  --parallel_reward \
  --output_dir trainer_output_h100

# View training history after completion
echo "Viewing training history:"
python main.py --view_history --history_file trainer_output_h100/training_results.csv

echo "Training complete!"