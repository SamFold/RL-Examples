# Nebius Cloud Deployment Guide for RLHF Training

This guide provides instructions for deploying the RLHF training pipeline on Nebius Cloud (formerly Yandex Cloud) to achieve faster training times and enable rapid hyperparameter exploration.

## Why Cloud GPU Training for RLHF

Your MacBook Air M2 is powerful for everyday tasks, but RLHF training benefits significantly from:
- Dedicated GPUs with CUDA support (not available on M-series chips)
- High RAM capacity for larger batch sizes
- Faster I/O for data loading
- Ability to run multiple experiments in parallel

## Nebius Setup Overview

### 1. Required Nebius Components

- **Compute Cloud**: For GPU-accelerated VMs
- **Object Storage**: For storing models and datasets
- **CLI**: For managing resources

### 2. VM Configuration

**GPU Instance Configuration**:
- Machine type: `standard-v2` (8 vCPUs, 32 GB memory)
- GPU: NVIDIA Tesla V100 or A100
- Boot disk: 200GB SSD
- Image: Ubuntu 22.04 LTS with CUDA drivers
- Estimated cost: Check Nebius pricing for current rates

**Cost-Saving Tip**: Use preemptible instances for significant cost reduction but with potential interruption risk (good for initial hyperparameter exploration).

## Environment Setup for Your Current Instance

Since you're already connected to your Nebius instance at `89.169.113.214`, let's set it up for RLHF training:

### Step 1: Install Required Software

```bash
# Update package list
sudo apt update

# Install Python development tools and other dependencies
sudo apt install -y python3-dev python3-pip git wget tmux htop nvidia-driver-535

# Verify CUDA installation
nvidia-smi

# Install Python packages
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio
```

### Step 2: Clone and Configure Your Repository

```bash
# Clone your repository
git clone https://github.com/SamFold/RL-Examples.git
cd RL-Examples

# Create virtual environment
python3 -m venv rlhf_env
source rlhf_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key for GPT-4o reward model
export OPENAI_API_KEY=your_api_key
echo 'export OPENAI_API_KEY=your_api_key' >> ~/.bashrc
```

### Step 3: Optimize Training Parameters

Edit `sentiment_rlhf/utils/config.py` to take advantage of GPU acceleration:

```python
# Example optimizations for GPU
config = {
    "batch_size": 32,  # Increased from 16
    "learning_rate": 1e-5,
    "max_epochs": 200,
    "num_ppo_updates": 8,
    "update_ref_freq": 50,
    "ref_ema_coef": 0.95
}
```

### Step 4: Set Up Object Storage (Optional)

If your dataset or models are large, you can use Nebius Object Storage:

```bash
# Install Nebius CLI if not already available
curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

# Initialize CLI
yc init

# Create a bucket
yc storage bucket create --name your-rlhf-bucket

# Upload data
yc storage cp --recursive ./data/ s3://your-rlhf-bucket/data/

# Download at runtime (add to your scripts)
# yc storage cp --recursive s3://your-rlhf-bucket/data/ ./data/
```

### Step 5: Start Training with TMUX

Using TMUX will keep your training running even if your connection drops:

```bash
# Start a new tmux session
tmux new -s rlhf-training

# Activate environment and start training
source rlhf_env/bin/activate
python main.py --model_name lvwerra/gpt2-imdb --batch_size 32 --device cuda --openai_api_key $OPENAI_API_KEY

# Detach from tmux session with Ctrl+B followed by D
# Reattach later with: tmux attach -t rlhf-training
```

### Step 6: Monitor Performance

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor memory usage
watch -n 1 free -h

# View training logs if using tmux
tmux attach -t rlhf-training
```

### Step 7: Download Results to Local Machine

From your local machine:

```bash
# Replace with your username and instance IP
scp -r user@89.169.113.214:~/reinforment-learning-with-human-feedback/trainer_output/* ./trainer_output/
```

## Multi-GPU Training (If Available)

If your instance has multiple GPUs, you can distribute training:

```python
# In your code, modify to use DataParallel
import torch.nn as nn
model = nn.DataParallel(model)
```

## Hyperparameter Optimization Strategies

### Option 1: Run Sequential Experiments

Simply modify config values between runs to iterate through hyperparameter combinations.

### Option 2: Run Parallel Experiments

1. Create multiple VM instances
2. Clone your repo to each instance
3. Run different hyperparameter configurations on each

### Option 3: Use Weight & Biases or MLflow

For tracking experiments:

```bash
# Install W&B
pip install wandb

# Initialize in your code
import wandb
wandb.init(project="rlhf-training")

# Log metrics
wandb.log({"reward": avg_reward, "loss": avg_loss})
```

## Expected Performance Improvement

| Configuration | Training Time (approx.) |
|---------------|------------------------|
| MacBook Air M2 | 35 minutes            |
| Nebius V100 GPU | 3-5 minutes          |
| Nebius A100 GPU | 1-3 minutes          |

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or gradient accumulation steps
- **VM Connection Issues**: Check network settings and SSH keys
- **Environment Errors**: Verify all dependencies are installed correctly
- **Performance Not Improved**: Ensure code is actually using GPU (check with `nvidia-smi`)

## Cost Management

- **Start/Stop Instances**: Stop the instance when not in use to save costs
```bash
# Use the Nebius CLI to stop your instance when not in use
yc compute instance stop your-instance-id
```

- **Use Preemptible Instances**: For non-critical training, preemptible instances can save up to 70%

- **Monitor Usage**: Regularly check the Nebius billing dashboard

## Security Notes

- **API Keys**: Never hardcode OpenAI API keys in your scripts
- **Firewall**: Restrict SSH access to your IP address only
- **Updates**: Keep the system updated with `sudo apt update && sudo apt upgrade`

## Conclusion

Following this guide should allow you to run your RLHF training significantly faster on Nebius Cloud compared to your local MacBook Air M2. This enables rapid iteration on hyperparameters and models, accelerating your research process.