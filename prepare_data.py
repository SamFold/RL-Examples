from datasets import load_dataset
import pandas as pd
import os

# Create data directory
os.makedirs("data", exist_ok=True)

print("Downloading datasets...")

# Download and save the comparison dataset
print("Downloading comparison dataset...")
comparisons = load_dataset("CarperAI/openai_summarize_comparisons", split="test")
comparisons_df = comparisons.to_pandas()
comparisons_df.to_parquet("data/test.parquet")
print(f"Saved to data/test.parquet with {len(comparisons_df)} rows")

# Download and save the TLDR dataset
print("Downloading TLDR dataset...")
tldr = load_dataset("CarperAI/openai_summarize_tldr", split="test")
tldr_df = tldr.to_pandas()
tldr_df.to_parquet("data/test_policy.parquet")
print(f"Saved to data/test_policy.parquet with {len(tldr_df)} rows")

print("Data preparation complete!")