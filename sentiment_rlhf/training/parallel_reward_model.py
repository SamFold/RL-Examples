"""Parallel reward model implementation for sentiment RLHF using GPT-4o."""

import re
import aiohttp
import asyncio
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

from sentiment_rlhf.training.reward_model import BaseRewardModel


class ParallelGPT4RewardModel(BaseRewardModel):
    """Reward model using GPT-4o with parallel API calls."""
    
    def __init__(self, api_key, model="gpt-4o", batch_size=32, max_workers=16):
        """
        Initialize the parallel GPT-4o reward model.
        
        Args:
            api_key (str): OpenAI API key.
            model (str): Model to use (default: gpt-4o).
            batch_size (int): Maximum batch size for parallel processing.
            max_workers (int): Maximum number of concurrent workers.
        """
        self.api_key = api_key
        self.model = model
        self.reward_cache = {}  # Cache to avoid re-evaluating identical texts
        self.num_api_calls = 0  # Counter to track API usage
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # System prompt for GPT-4o
        self.system_prompt = """You are a helpful movie review evaluator. Your task is to assess movie reviews
        based on two equally important factors:
        1. Positivity of sentiment
        2. Coherence and grammatical correctness
        
        Rate each review on a scale from 0 to 10 where:
        - 0 = extremely negative AND/OR completely incoherent
        - 10 = extremely positive AND coherent/well-written
        
        IMPORTANT: A review must be BOTH positive AND coherent to receive a high score.
        A positive but incoherent review should score no higher than 5.
        A coherent but negative review should score no higher than 3.
        
        Your response should ONLY contain the numerical score as a float between 0 and 10.
        No explanation, no text, just a single number."""
    
    def __call__(self, texts):
        """
        Calculate GPT-4o rewards for a batch of texts in parallel.
        
        Args:
            texts: List of texts to evaluate.
            
        Returns:
            Tensor of reward values between 0 and 1.
        """
        # For single items, convert to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Fail fast if API key is not valid
        if not self.api_key or self.api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("ERROR: Must provide a valid OpenAI API key.")
        
        # Run the async evaluation with proper event loop handling
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in this thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        rewards = loop.run_until_complete(self._evaluate_batch(texts))
        return torch.tensor(rewards)
    
    async def _evaluate_batch(self, texts):
        """
        Evaluate a batch of texts in parallel.
        
        Args:
            texts: List of texts to evaluate.
            
        Returns:
            List of reward values.
        """
        # Check cache first for all texts
        uncached_texts = []
        uncached_indices = []
        rewards = [0.0] * len(texts)  # Pre-allocate results array
        
        # First pass - check cache
        for i, text in enumerate(texts):
            # Use first 200 chars as key to save memory
            cache_key = text[:200]
            if cache_key in self.reward_cache:
                rewards[i] = self.reward_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # If all texts were in cache, return immediately
        if not uncached_texts:
            return rewards
        
        # Process uncached texts in parallel batches
        print(f"Making {len(uncached_texts)} parallel API calls to OpenAI")
        start_time = time.time()
        
        # Set up the API session
        async with aiohttp.ClientSession() as session:
            # Create tasks for all uncached texts
            tasks = [
                self._get_gpt4_reward(session, text)
                for text in uncached_texts
            ]
            
            # Run all tasks and gather results
            uncached_rewards = await asyncio.gather(*tasks)
            
            # Store results in the rewards list and cache
            for i, (idx, reward) in enumerate(zip(uncached_indices, uncached_rewards)):
                rewards[idx] = reward
                # Cache the result using the first 200 chars as key
                self.reward_cache[uncached_texts[i][:200]] = reward
        
        end_time = time.time()
        print(f"Processed {len(uncached_texts)} texts in {end_time - start_time:.2f} seconds "
              f"({len(uncached_texts) / (end_time - start_time):.2f} texts/second)")
        
        return rewards
    
    async def _get_gpt4_reward(self, session, prompt_and_completion):
        """
        Asynchronously get GPT-4o reward for a single text.
        
        Args:
            session: aiohttp client session to use for API call
            prompt_and_completion: The text to evaluate
            
        Returns:
            Reward value between 0 and 1
        """
        try:
            # Make async API call
            response = await session.post(
                f"https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f'Rate this movie review: "{prompt_and_completion}"'}
                    ],
                    "temperature": 0.0  # Use 0 temperature for consistency
                }
            )
            
            # Parse the response JSON
            result = await response.json()
            
            # Check for API errors
            if "error" in result:
                raise ValueError(f"API Error: {result['error']['message']}")
                
            # Check for correct response format
            if "choices" not in result:
                raise ValueError(f"Unexpected API response format. Keys: {list(result.keys())}")
                
            # Parse the content from the message
            score_text = result["choices"][0]["message"]["content"].strip()
            
            # Convert to number
            try:
                score = float(score_text)
                # Clamp score to 0-10 range
                score = max(0, min(10, score))
            except ValueError:
                # If GPT-4o didn't return just a number, try to extract it
                matches = re.findall(r'\d+\.?\d*', score_text)
                if not matches:
                    raise ValueError(f"Could not parse numerical score from response: '{score_text}'")
                score = float(matches[0])
                score = max(0, min(10, score))
            
            # Normalize to 0-1 range
            score = score / 10.0
            
            # Count the API call
            self.num_api_calls += 1
            
            return score
            
        except Exception as e:
            # Log error and return a default value
            print(f"Error in async GPT-4o call: {str(e)}")
            return 0.5  # Return a neutral score on error


def create_parallel_reward_model(reward_type, **kwargs):
    """
    Create a parallel reward model based on the specified type.
    
    Args:
        reward_type (str): Type of reward model to create (only 'gpt4' is supported).
        **kwargs: Additional arguments for the reward model.
        
    Returns:
        A parallel reward model instance.
    """
    if reward_type == "gpt4":
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for GPT-4o reward model")
        
        # Always use the batch size from global config
        from sentiment_rlhf.utils import get_default_config
        default_config = get_default_config()
        max_workers = kwargs.get("max_workers", 16)
        
        return ParallelGPT4RewardModel(
            api_key=kwargs["api_key"],
            batch_size=default_config.batch_size,
            max_workers=max_workers
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. Only 'gpt4' is supported.")