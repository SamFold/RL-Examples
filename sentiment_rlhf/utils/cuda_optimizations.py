"""CUDA optimization utilities for H100 and other NVIDIA GPUs."""

import torch
from typing import Optional, Dict, Any, Union
from contextlib import nullcontext


class MixedPrecisionManager:
    """
    Manager for mixed precision training using PyTorch's native AMP.
    Handles automatic mixed precision (FP16/BF16) for better performance on H100 GPUs.
    """
    
    def __init__(
        self, 
        enabled: bool = True,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
    ):
        """
        Initialize mixed precision manager.
        
        Args:
            enabled: Whether to enable mixed precision.
            dtype: Precision type ('float16' or 'bfloat16'). BF16 is recommended for H100.
            device: Device to use ('cuda', 'cpu', etc.). If None, uses CUDA if available.
        """
        self.enabled = enabled
        self.original_dtype = None
        self.scaler = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set precision based on hardware and user choice
        if self.enabled:
            if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                # H100 (capability 9.0) and A100 (capability 8.0) support BF16 well
                self.mixed_dtype = torch.bfloat16
                print("Using BFloat16 mixed precision (optimal for H100/A100)")
            elif dtype == "float16" or (dtype == "bfloat16" and not torch.cuda.is_bf16_supported()):
                # Fall back to FP16 for other GPUs
                self.mixed_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
                print("Using Float16 mixed precision with gradient scaling")
            else:
                # Unsupported configuration, disable mixed precision
                self.enabled = False
                self.mixed_dtype = torch.float32
                print("Mixed precision not supported on this hardware with dtype", dtype)
        else:
            # Mixed precision disabled
            self.mixed_dtype = torch.float32
        
        # Apply CUDA optimizations
        if torch.cuda.is_available() and enabled:
            # Enable TF32 precision (specific to Ampere/H100 architecture)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cuDNN benchmarking and deterministic algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Log GPU information for reference
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA device: {device_name}")
            if "H100" in device_name:
                print("H100 GPU detected - all optimizations enabled")
                
    def __enter__(self):
        """Context manager entry point - switches to mixed precision."""
        if self.enabled:
            self.original_dtype = torch.get_default_dtype()
            # We don't actually change the default dtype, as that can cause issues
            # Instead we use torch.amp.autocast
            return self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - restores original precision."""
        if self.enabled and self.original_dtype is not None:
            # No need to restore since we're using autocast
            pass
            
    def get_autocast(self):
        """Get the autocast context for the current configuration."""
        if self.enabled:
            return torch.amp.autocast(device_type=self.device, dtype=self.mixed_dtype)
        else:
            # Return a dummy context manager when disabled
            return nullcontext()
        
    def backward(self, loss, optimizer=None):
        """
        Perform backward pass with gradient scaling if needed.
        
        Args:
            loss: The loss tensor to backpropagate.
            optimizer: Optional optimizer to step after backward.
            
        Returns:
            True if optimizer step was taken, False otherwise.
        """
        if not self.enabled or self.scaler is None:
            # Standard backward pass
            loss.backward()
            if optimizer is not None:
                optimizer.step()
            return True
            
        # Scaled backward pass (for float16)
        self.scaler.scale(loss).backward()
        
        if optimizer is not None:
            # Step with gradient scaling
            self.scaler.step(optimizer)
            self.scaler.update()
            return True
        
        return False
    
    def optimizer_step(self, optimizer, **kwargs):
        """
        Step optimizer with gradient scaling if needed.
        
        Args:
            optimizer: Optimizer to step.
            **kwargs: Additional arguments to pass to optimizer.step().
        """
        if not self.enabled or self.scaler is None:
            optimizer.step(**kwargs)
        else:
            self.scaler.step(optimizer)
            self.scaler.update()


def apply_trainer_optimizations(
    trainer,
    use_mixed_precision: bool = True,
    precision_dtype: str = "bfloat16",
    optimize_memory: bool = True
) -> Dict[str, Any]:
    """
    Apply CUDA optimizations to a trainer object.
    
    Args:
        trainer: The trainer object to optimize.
        use_mixed_precision: Whether to enable mixed precision training.
        precision_dtype: Precision type ('float16' or 'bfloat16').
        optimize_memory: Whether to enable memory optimizations.
        
    Returns:
        Dict of optimization settings applied.
    """
    # Create mixed precision manager
    mp_manager = MixedPrecisionManager(enabled=use_mixed_precision, dtype=precision_dtype, device=trainer.device)
    
    # Store mixed precision manager in trainer for later use
    trainer.mp_manager = mp_manager
    
    # Apply memory optimizations
    optimizations = {
        "mixed_precision": use_mixed_precision,
        "precision_dtype": precision_dtype,
        "memory_optimizations": {},
    }
    
    if optimize_memory and hasattr(trainer, "model") and trainer.device == "cuda":
        from torch.utils.checkpoint import checkpoint_sequential
        
        # Only apply if model is complex enough (more than 10M parameters)
        model_size = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        if model_size > 10_000_000:  # 10M parameters
            # Enable gradient checkpointing to save memory
            if hasattr(trainer.model, "gradient_checkpointing_enable"):
                trainer.model.gradient_checkpointing_enable()
                optimizations["memory_optimizations"]["gradient_checkpointing"] = True
    
    return optimizations


def get_optimal_cuda_settings(gpu_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get optimal CUDA settings based on GPU hardware.
    
    Args:
        gpu_name: Name of the GPU. If None, will be detected automatically.
        
    Returns:
        Dict with optimal settings.
    """
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    settings = {
        "mixed_precision": True,
        "precision_dtype": "bfloat16",
        "optimize_memory": True,
        "batch_size": 32,  # Default
    }
    
    if gpu_name:
        if "H100" in gpu_name:
            # Optimal H100 settings
            settings.update({
                "precision_dtype": "bfloat16",  # H100 has excellent BF16 support
                "batch_size": 64,  # H100 has 80GB memory, can handle larger batches
            })
        elif "A100" in gpu_name:
            # Optimal A100 settings
            settings.update({
                "precision_dtype": "bfloat16",  # A100 has good BF16 support
                "batch_size": 48,  # A100 has 40GB or 80GB memory
            })
        elif any(x in gpu_name for x in ["3090", "4090", "3080", "4080"]):
            # Settings for high-end GeForce
            settings.update({
                "precision_dtype": "float16",  # Better FP16 than BF16 on consumer GPUs
                "batch_size": 32,  # 24GB on 3090/4090
            })
            
    return settings