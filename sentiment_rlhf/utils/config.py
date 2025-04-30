"""Configuration utilities for sentiment RLHF."""

from dataclasses import dataclass, field
from typing import Dict, Any
from trl import PPOConfig

@dataclass
class ModelConfig:
    """Model configuration for RLHF training."""
    model_name: str = "lvwerra/gpt2-imdb"


def default_model_config() -> ModelConfig:
    """Default factory for ModelConfig."""
    return ModelConfig()


def default_ppo_config() -> PPOConfig:
    """Default factory for PPOConfig."""
    return PPOConfig(
        learning_rate=1e-5,  # Reduced from 1.41e-5 for better stability
        per_device_train_batch_size=32,  # Increased for H100 GPU
        per_device_eval_batch_size=32,  # Increased for H100 GPU
    )


def default_sentiment_kwargs() -> Dict[str, Any]:
    """Default factory for sentiment kwargs."""
    return {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16
    }


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # Model configuration
    model_config: ModelConfig = field(default_factory=default_model_config)
    
    # PPO configuration
    ppo_config: PPOConfig = field(default_factory=default_ppo_config)
    
    # Dataset configuration
    dataset_name: str = "imdb"
    input_min_text_length: int = 2
    input_max_text_length: int = 8
    max_sequence_length: int = 1024
    
    # Output configuration
    output_min_length: int = 20
    output_max_length: int = 40
    
    # Training parameters
    max_epochs: int = 200
    save_freq: int = 50
    patience: int = 20
    early_stopping_threshold: float = 0.01
    
    # Reference model update strategy
    update_ref_freq: int = 50  # Much less frequent updates
    ref_ema_coef: float = 0.95  # Exponential moving average coefficient for reference model updates
    max_kl_target: float = 0.50  # Maximum allowed KL divergence before forced update
    
    # PPO specific parameters
    reward_coef: float = 0.5
    kl_penalty: float = 0.05
    lm_loss_coef: float = 0.00
    clip_epsilon: float = 0.2
    num_ppo_updates: int = 8
    entropy_coef: float = 0.01
    use_exploration: bool = True  # Whether to use entropy bonus for exploration
    
    # GAE parameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    
    # Optimization parameters
    use_lr_scheduler: bool = False  # Whether to use learning rate scheduling
    policy_grad_clip: float = 0.5  # Gradient clipping for policy network
    value_grad_clip: float = 0.5  # Gradient clipping for value network
    value_lr_multiplier: float = 2.0  # Multiplier for value head learning rate relative to policy
    
    # Sentiment configuration
    sentiment_kwargs: Dict[str, Any] = field(default_factory=default_sentiment_kwargs)


def get_default_config():
    """Get default configuration for RLHF training."""
    return RLHFConfig()
