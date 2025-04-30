"""Model loading utilities for sentiment RLHF."""

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

class ModelLoader:
    """A class to load and prepare models for RLHF training."""
    
    def __init__(self, model_name):
        """
        Initialize the model loader.
        
        Args:
            model_name (str): Name of the base pretrained model.
        """
        self.model_name = model_name
    
    def load_model_for_training(self):
        """
        Load the model, tokenizer, and generation config for training.
        
        Returns:
            Tuple of (model, ref_model, tokenizer, generation_config)
        """
        # Load the base language model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Load the tokenizer for text processing
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set pad token to eos token for proper padding
        tokenizer.pad_token = tokenizer.eos_token

        # Create a generation config
        generation_config = GenerationConfig(
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Add generation config to base model
        base_model.generation_config = generation_config

        # Create the model with value head
        model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name)
        # Add generation config to model
        model.generation_config = generation_config
        # Make sure the base model prefix is set
        model.pretrained_model.base_model_prefix = base_model.base_model_prefix

        # Create a reference model - this stays fixed during training
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name)
        # Add generation config to reference model
        ref_model.generation_config = generation_config
        ref_model.pretrained_model.base_model_prefix = base_model.base_model_prefix
        
        return model, ref_model, tokenizer, generation_config
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory.
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Handle key mismatches if they occur
        try:
            # Try normal loading first
            model = AutoModelForCausalLMWithValueHead.from_pretrained(checkpoint_path)
        except RuntimeError as e:
            if "Missing key(s) in state_dict" in str(e):
                print("Detected key structure mismatch, attempting to fix...")
                from model_fix import load_model_with_key_fixing
                model = load_model_with_key_fixing(checkpoint_path)
            else:
                raise
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
