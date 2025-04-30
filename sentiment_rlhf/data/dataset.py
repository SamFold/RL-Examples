"""Dataset utilities for sentiment RLHF."""

from datasets import load_dataset
from transformers import AutoTokenizer
from trl.core import LengthSampler


def build_dataset(model_name, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8, max_sequence_length=1024):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        model_name (`str`):
            The name of the model to use for tokenization.
        dataset_name (`str`, optional):
            The name of the dataset to be loaded.
        input_min_text_length (`int`, optional):
            Minimum length for random truncation.
        input_max_text_length (`int`, optional):
            Maximum length for random truncation.
        max_sequence_length (`int`, optional):
            Maximum sequence length to prevent indexing errors.

    Returns:
        dataset: The processed dataset.
    """
    # Initialize the tokenizer from the model configuration
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set pad token to eos token (essential for proper batching)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load IMDB dataset from Hugging Face datasets
    ds = load_dataset(dataset_name, split="train")
    # Rename 'text' column to 'review' for clarity
    ds = ds.rename_columns({"text": "review"})
    # Filter for reviews longer than 200 characters (for meaningful content)
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # Create a sampler to randomly select text lengths between min and max
    # This adds variability to prevent model from overfitting to specific input lengths
    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    # Define tokenization function to apply to each sample
    def tokenize(sample):
        # First encode with truncation to ensure we don't exceed model's max length
        encoded = tokenizer.encode(sample["review"], truncation=True, max_length=max_sequence_length)
        # Then apply random length sampling from the truncated sequence
        length = min(input_size(), len(encoded))
        sample["input_ids"] = encoded[:length]
        # Store the decoded query for later use (to generate continuations)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # Apply tokenization to each example
    ds = ds.map(tokenize, batched=False)
    # Set format to PyTorch tensors
    ds.set_format(type="torch")
    return ds


def collator(data):
    """
    Data collator to batch examples together.
    This function combines individual data points into batch dictionaries.
    
    Args:
        data: List of examples.
        
    Returns:
        Batched dictionary.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])
