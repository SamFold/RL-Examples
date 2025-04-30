#!/bin/bash

# Script to run RLHF training on macOS with MPS (Apple Silicon)

# Set environment variables
export OPENAI_API_KEY="YOUR_API_KEY_HERE"

# Run with optimized parameters for Apple Silicon M2
python main.py \
  --model_name lvwerra/gpt2-imdb \
  --batch_size 16 \
  --max_epochs 200 \
  --device mps \
  --optimize_device \
  --openai_api_key $OPENAI_API_KEY \
  --parallel_reward \
  --output_dir trainer_output_mps

# View training history after completion
echo "Viewing training history:"
python main.py --view_history --history_file trainer_output_mps/training_results.csv

echo "Training complete!"