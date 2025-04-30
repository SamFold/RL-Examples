#!/usr/bin/env python
"""
Main script for sentiment RLHF model training.
"""
import argparse
import os
import json
import torch
import pandas as pd
from datetime import datetime

from sentiment_rlhf.data import build_dataset
from sentiment_rlhf.models import ModelLoader
from sentiment_rlhf.training import SentimentRLHFTrainer, create_reward_model
from sentiment_rlhf.training.parallel_reward_model import create_parallel_reward_model
from sentiment_rlhf.utils import get_default_config, get_optimal_cuda_settings, apply_trainer_optimizations
from sentiment_rlhf.utils.config import default_ppo_config, default_sentiment_kwargs


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Sentiment RLHF Training")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="lvwerra/gpt2-imdb",
                        help="The name of the model to train")
    
    # Training configuration
    parser.add_argument("--max_epochs", type=int, default=200,
                        help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    # We've removed the learning_rate parameter as it's now only configured in config.py
    parser.add_argument("--save_freq", type=int, default=50,
                        help="Frequency of saving checkpoints")
    
    # Reward model configuration
    parser.add_argument("--openai_api_key", type=str, required=True,
                        help="OpenAI API key for GPT-4o reward model")
    parser.add_argument("--parallel_reward", action="store_true",
                        help="Use parallel reward model for faster evaluation")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="trainer_output",
                        help="Directory to save outputs")
    
    # Device configuration
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (cuda, mps, or cpu)")
    parser.add_argument("--optimize_device", action="store_true",
                        help="Enable device-specific optimizations (H100 for CUDA, MPS for Mac)")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training for faster performance on H100/A100 GPUs")
    parser.add_argument("--precision_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                        help="Mixed precision data type to use (bfloat16 for H100/A100, float16 for other GPUs)")
    
    # Training behavior configuration
    parser.add_argument("--no_exploration", action="store_true",
                        help="Disable exploration (entropy bonus) during training")
    
    # Inference mode
    parser.add_argument("--inference", action="store_true",
                        help="Run in inference mode instead of training")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a trained model for inference")
    
    return parser.parse_args()


def setup_training(args):
    """
    Set up the training process.
    
    Args:
        args: The command line arguments.
        
    Returns:
        Tuple of trainer, tokenizer, and config.
    """
    # Set up config
    config = get_default_config()
    
    # Override config with arguments
    config.model_config.model_name = args.model_name
    config.batch_size = args.batch_size
    # Recreate ppo_config with updated batch size
    config.ppo_config = default_ppo_config(config.batch_size)
    config.ppo_config.output_dir = args.output_dir
    # Recreate sentiment_kwargs with updated batch size
    config.sentiment_kwargs = default_sentiment_kwargs(config.batch_size)
    config.max_epochs = args.max_epochs
    config.save_freq = args.save_freq
    
    # Set exploration setting
    if args.no_exploration:
        config.use_exploration = False
        print("Exploration (entropy bonus) disabled for training")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        # Convert dataclasses to dictionaries with custom handling for non-serializable objects
        def prepare_for_json(obj):
            if hasattr(obj, "__dict__"):
                return {k: prepare_for_json(v) for k, v in vars(obj).items()
                        if not k.startswith("_")}
            elif isinstance(obj, (list, tuple)):
                return [prepare_for_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: prepare_for_json(v) for k, v in obj.items()}
            elif hasattr(obj, "__dataclass_fields__"):
                return {k: prepare_for_json(getattr(obj, k)) for k in obj.__dataclass_fields__}
            else:
                # Try to make it JSON serializable, or convert to string
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, OverflowError):
                    return str(obj)
                
        config_dict = {
            "model_config": prepare_for_json(config.model_config),
            "ppo_config": prepare_for_json(config.ppo_config),
            "training": {k: prepare_for_json(v) for k, v in vars(config).items()
                         if k not in ["model_config", "ppo_config"]}
        }
        json.dump(config_dict, f, indent=2)
    
    print(f"Loading model: {config.model_config.model_name}")
    model_loader = ModelLoader(config.model_config.model_name)
    model, ref_model, tokenizer, generation_config = model_loader.load_model_for_training()
    
    print("Building dataset...")
    dataset = build_dataset(
        model_name=config.model_config.model_name,
        dataset_name=config.dataset_name,
        input_min_text_length=config.input_min_text_length,
        input_max_text_length=config.input_max_text_length,
        max_sequence_length=config.max_sequence_length
    )
    
    print("Setting up GPT-4o reward model...")
    if args.parallel_reward:
        print(f"Using parallel reward model with batch size {config.batch_size}")
        reward_model = create_parallel_reward_model(
            reward_type="gpt4",
            api_key=args.openai_api_key
        )
    else:
        print("Using sequential reward model")
        reward_model = create_reward_model(
            reward_type="gpt4",
            api_key=args.openai_api_key
        )
    
    print("Initializing trainer...")
    trainer = SentimentRLHFTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        reward_model=reward_model,
        device=args.device,
        optimize_device=args.optimize_device
    )
    
    return trainer, tokenizer, config


def run_inference(args):
    """
    Run inference with a trained model.
    
    Args:
        args: The command line arguments.
    """
    # Get default config to use batch size
    config = get_default_config()
    config.batch_size = args.batch_size
    
    if args.model_path is None:
        raise ValueError("Model path is required for inference mode")
    
    # Set up model and tokenizer
    from transformers import AutoTokenizer, pipeline
    from trl import AutoModelForCausalLMWithValueHead
    
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    model = model.to(device)
    
    # Define test prompts
    test_prompts = [
        "This movie was",
        "I thought the film was",
        "The director did",
        "The actors were",
        "The screenplay was",
        "Watching this movie made me feel",
        "The special effects were",
        "The plot of the movie was",
        "The soundtrack was",
        "The cinematography was"
    ]
    
    # Generate text
    model.eval()
    generations_by_prompt = []
    num_samples = 5  # Number of samples to generate per prompt
    
    print("Generating text...")
    for prompt in test_prompts:
        prompt_generations = []
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        for i in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_generations.append(generated_text)
        
        generations_by_prompt.append(prompt_generations)
    
    # Flatten for evaluation
    generations_flat = [gen for prompt_gens in generations_by_prompt for gen in prompt_gens]
    
    # Create GPT-4o reward model for evaluation
    from sentiment_rlhf.training import create_reward_model
    from sentiment_rlhf.training.parallel_reward_model import create_parallel_reward_model
    
    if args.parallel_reward:
        print(f"Using parallel reward model with batch size {config.batch_size}")
        reward_model = create_parallel_reward_model(
            reward_type="gpt4",
            api_key=args.openai_api_key
        )
    else:
        print("Using sequential reward model")
        reward_model = create_reward_model(
            reward_type="gpt4",
            api_key=args.openai_api_key
        )
    
    # Evaluate with GPT-4o
    print("Evaluating generations with GPT-4o...")
    scores_flat = reward_model(generations_flat).tolist()
    
    # Restructure scores to match generations
    scores_by_prompt = []
    idx = 0
    for prompt_gens in generations_by_prompt:
        scores_by_prompt.append(scores_flat[idx:idx+len(prompt_gens)])
        idx += len(prompt_gens)
    
    # Calculate averages per prompt
    prompt_averages = [sum(scores)/len(scores) for scores in scores_by_prompt]
    overall_average = sum(prompt_averages) / len(prompt_averages)
    
    # Print results with all samples
    print("\nGPT-4o evaluation results:")
    for i, (prompt, generations, scores, avg_score) in enumerate(zip(
        test_prompts, generations_by_prompt, scores_by_prompt, prompt_averages
    )):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Average score: {avg_score:.3f}")
        
        for j, (generation, score) in enumerate(zip(generations, scores)):
            print(f"\n  Sample {j+1} ({score:.3f}): {generation}")
        
        print("-" * 80)
    
    print(f"\nOverall average GPT-4o score: {overall_average:.3f}")


def main():
    """
    Main function for sentiment RLHF model training.
    """
    args = parse_arguments()
    
    # Print arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Get device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
            
            # Display CUDA device info
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Get optimal settings based on GPU hardware
            if args.optimize_device:
                optimal_settings = get_optimal_cuda_settings()
                print(f"Applying optimal CUDA settings for {torch.cuda.get_device_name(0)}")
                
                # Apply basic CUDA optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                # Enable mixed precision by default for H100/A100 if optimizing
                if "H100" in torch.cuda.get_device_name(0) or "A100" in torch.cuda.get_device_name(0):
                    args.mixed_precision = True
                    print(f"Mixed precision automatically enabled for {torch.cuda.get_device_name(0)}")
            
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            args.device = "mps"
            # Apply MPS optimizations if flag is set
            if args.optimize_device:
                print("Applying MPS-specific optimizations for Apple Silicon")
                # Apple Silicon specific optimizations could go here
        else:
            args.device = "cpu"
            
    print(f"Using device: {args.device}")
    
    # Report mixed precision status
    if args.mixed_precision and args.device == "cuda":
        print(f"Mixed precision training enabled using {args.precision_dtype}")
    elif args.mixed_precision and args.device != "cuda":
        print("Mixed precision requested but not supported on this device. Using full precision.")
        args.mixed_precision = False
    
    if args.inference:
        run_inference(args)
        return
    
    # Setup training
    trainer, tokenizer, config = setup_training(args)
    
    # Apply mixed precision optimizations if enabled
    if args.mixed_precision and args.device == "cuda":
        print("Applying mixed precision optimizations to trainer...")
        optimization_stats = apply_trainer_optimizations(
            trainer,
            use_mixed_precision=True,
            precision_dtype=args.precision_dtype,
            optimize_memory=args.optimize_device
        )
        print(f"Applied optimizations: {optimization_stats}")
    
    # Store reward model reference to use later
    reward_model = trainer.reward_model
    
    # Train model
    print("Starting training...")
    start_time = datetime.now()
    train_stats = trainer.train()
    end_time = datetime.now()
    
    # Print training stats
    print("\nTraining complete!")
    print(f"Training time: {end_time - start_time}")
    print(f"Best reward: {train_stats['best_reward']:.3f}")
    print(f"Best epoch: {train_stats['best_epoch']}")
    
    # Save training stats
    stats_path = os.path.join(args.output_dir, "train_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {k: v if not isinstance(v, list) else v[:10] for k, v in train_stats.items()}, 
            f, 
            indent=2
        )
    print(f"Training stats saved to {stats_path}")
    
    # Test prompts
    test_prompts = [
        "This movie was",
        "I thought the film was",
        "The director did",
        "The actors were",
        "The screenplay was"
    ]
    
    # Compare models
    print("\nComparing models...")
    comparison = trainer.compare_models(
        prompts=test_prompts,
        reward_model=reward_model
    )
    
    # Print comparison
    print(f"Reference model average GPT-4o score: {comparison['ref_avg']:.3f}")
    print(f"Trained model average GPT-4o score: {comparison['trained_avg']:.3f}")
    print(f"Average improvement: {comparison['avg_diff']:.3f}")
    
    # Print sample comparisons
    print("\nSample comparisons:")
    for i, (prompt, ref_reviews, trained_reviews, ref_scores, trained_scores, ref_avg, trained_avg) in enumerate(zip(
        comparison["prompts"],
        comparison["ref_reviews"],
        comparison["trained_reviews"],
        comparison["ref_scores"],
        comparison["trained_scores"],
        comparison["ref_prompt_avgs"],
        comparison["trained_prompt_avgs"]
    )):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Reference model (avg: {ref_avg:.3f}):")
        for j, (review, score) in enumerate(zip(ref_reviews, ref_scores)):
            print(f"  Sample {j+1} ({score:.3f}): {review}")
        
        print(f"\nTrained model (avg: {trained_avg:.3f}):")
        for j, (review, score) in enumerate(zip(trained_reviews, trained_scores)):
            print(f"  Sample {j+1} ({score:.3f}): {review}")
        
        print(f"Improvement: {trained_avg - ref_avg:.3f}")
        print("-" * 80)
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()