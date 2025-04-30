"""Reward model implementation for sentiment RLHF using GPT-4o."""

import re
import requests
import time
import torch

class BaseRewardModel:
    """Base class for reward models."""
    
    def __call__(self, texts):
        """Calculate rewards for a batch of texts."""
        raise NotImplementedError("Subclasses must implement __call__")


class GPT4RewardModel(BaseRewardModel):
    """Reward model using GPT-4o."""
    
    def __init__(self, api_key, model="gpt-4o"):
        """
        Initialize the GPT-4o reward model.
        
        Args:
            api_key (str): OpenAI API key.
            model (str): Model to use (default: gpt-4o).
        """
        self.api_key = api_key
        self.model = model
        self.reward_cache = {}  # Cache to avoid re-evaluating identical texts
        self.num_api_calls = 0  # Counter to track API usage
    
    def __call__(self, texts):
        """
        Calculate GPT-4o rewards for a batch of texts.
        
        Args:
            texts: List of texts to evaluate.
            
        Returns:
            Tensor of reward values between 0 and 1.
        """
        rewards = []
        for text in texts:
            reward = self.get_gpt4_reward(text)
            rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def get_gpt4_reward(self, prompt_and_completion):
        """
        Ask GPT-4o to evaluate movie reviews on positivity and coherence.
        Returns a score between 0 and 10 normalized to 0-1.
        
        Args:
            prompt_and_completion: The text to evaluate.
            
        Returns:
            Reward value between 0 and 1.
        """
        # Check cache first (using the first 200 chars as key to save memory)
        cache_key = prompt_and_completion[:200]
        if cache_key in self.reward_cache:
            return self.reward_cache[cache_key]
        
        # Fail fast if API key is not valid
        if not self.api_key or self.api_key == "YOUR_OPENAI_API_KEY_HERE":
            raise ValueError("ERROR: Must provide a valid OpenAI API key.")
        
        try:
            # Prepare the system prompt for GPT-4o
            system_prompt = """You are a helpful movie review evaluator. Your task is to assess movie reviews
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
            
            # Call the OpenAI API
            response = requests.post(
                f"https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f'Rate this movie review: "{prompt_and_completion}"'}
                    ],
                    "temperature": 0.0  # Use 0 temperature for consistency
                }
            )
            
            # Parse the response
            result = response.json()
            
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
            
            # Cache the result
            self.reward_cache[cache_key] = score
            
            # Count the API call
            self.num_api_calls += 1
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
            return score
        except Exception as e:
            # Propagate error rather than silently failing
            raise Exception(f"Error getting GPT-4o reward: {str(e)}")


def create_reward_model(reward_type, **kwargs):
    """
    Create a reward model based on the specified type.
    
    Args:
        reward_type (str): Type of reward model to create (only 'gpt4' is supported).
        **kwargs: Additional arguments for the reward model.
        
    Returns:
        A reward model instance.
    """
    if reward_type == "gpt4":
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for GPT-4o reward model")
        return GPT4RewardModel(api_key=kwargs["api_key"])
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. Only 'gpt4' is supported.")
