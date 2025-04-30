# Nebius Instance Setup Fix

It looks like there's an issue with the repository structure on your Nebius instance. The `sentiment_rlhf.data` module is missing. Follow these steps to fix the structure:

## 1. Create the missing data directory and files

```bash
# Create data directory
mkdir -p ~/source/RL-Examples/sentiment_rlhf/data
mkdir -p ~/source/RL-Examples/sentiment_rlhf/models

# Create __init__.py files
touch ~/source/RL-Examples/sentiment_rlhf/data/__init__.py
touch ~/source/RL-Examples/sentiment_rlhf/models/__init__.py
```

## 2. Create the dataset.py file

Create the file `~/source/RL-Examples/sentiment_rlhf/data/dataset.py` with the following content:

```python
"""Dataset utilities for sentiment RLHF."""

from datasets import load_dataset
from typing import Optional

def build_dataset(
    model_name: str,
    dataset_name: str = "imdb",
    input_min_text_length: int = 2,
    input_max_text_length: int = 8,
    max_sequence_length: int = 1024,
):
    """
    Build a dataset for sentiment RLHF.
    
    Args:
        model_name: The name of the model to train.
        dataset_name: The name of the dataset to use.
        input_min_text_length: Minimum length of input text in words.
        input_max_text_length: Maximum length of input text in words.
        max_sequence_length: Maximum sequence length.
        
    Returns:
        The processed dataset.
    """
    # Load IMDB dataset
    if dataset_name == "imdb":
        imdb_dataset = load_dataset("imdb", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Define preprocessing function
    def preprocess(examples):
        processed_examples = []
        for text in examples["text"]:
            # Create short prompts from the beginning of the text
            words = text.split()
            
            # Skip texts that are too short
            if len(words) < input_min_text_length:
                continue
            
            # Take only a few words to form a prompt
            prompt_length = min(len(words), input_max_text_length)
            prompt = " ".join(words[:prompt_length])
            
            # Store as dict
            processed_examples.append({
                "query": prompt + " ",  # Add space after prompt
                "input_ids": None,  # To be filled by tokenizer
                "label": examples["label"],
                "full_text": text
            })
        
        return processed_examples
    
    # Process the dataset
    processed_dataset = imdb_dataset.map(
        lambda examples: {"processed": preprocess(examples)},
        batched=True,
        remove_columns=["text"]
    )
    
    # Flatten the dataset
    flattened_data = []
    for example in processed_dataset:
        flattened_data.extend(example["processed"])
    
    print(f"Built dataset with {len(flattened_data)} examples")
    
    return flattened_data
```

## 3. Update the data/__init__.py file

Update the file `~/source/RL-Examples/sentiment_rlhf/data/__init__.py` with:

```python
"""Dataset module for sentiment RLHF."""

from .dataset import build_dataset

__all__ = ["build_dataset"]
```

## 4. Create the model_loader.py file

Create the file `~/source/RL-Examples/sentiment_rlhf/models/model_loader.py` with:

```python
"""Model loading utilities for sentiment RLHF."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

class ModelLoader:
    """Loader for RLHF models."""
    
    def __init__(self, model_name):
        """
        Initialize the model loader.
        
        Args:
            model_name: The name of the model to load.
        """
        self.model_name = model_name
    
    def load_model_for_training(self):
        """
        Load the model for RLHF training.
        
        Returns:
            Tuple of (model, ref_model, tokenizer, generation_config).
        """
        # Load model and tokenizer
        print(f"Loading model {self.model_name}...")
        model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"
        
        # Set default generation config
        generation_config = GenerationConfig.from_pretrained(self.model_name)
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Set models to evaluation mode (we'll enable training mode later)
        model.eval()
        ref_model.eval()
        
        return model, ref_model, tokenizer, generation_config
```

## 5. Update the models/__init__.py file

Update the file `~/source/RL-Examples/sentiment_rlhf/models/__init__.py` with:

```python
"""Models module for sentiment RLHF."""

from .model_loader import ModelLoader

__all__ = ["ModelLoader"]
```

## 6. Install required dependencies

Ensure all required packages are installed:

```bash
pip install datasets trl transformers torch
```

## 7. Try running again

After completing these steps, try running the main.py script again:

```bash
python main.py --model_name lvwerra/gpt2-imdb --batch_size 32 \
  --max_epochs 20 --device cuda --optimize_device \
  --openai_api_key YOUR_API_KEY
```