"""PPO trainer implementation for sentiment RLHF."""

import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
import pandas as pd
from tqdm import tqdm

class SentimentRLHFTrainer:
    """Trainer for sentiment RLHF using PPO."""
    
    def __init__(self, config, model, ref_model, tokenizer, dataset, reward_model, device=None, optimize_device=False):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration.
            model: Model with value head.
            ref_model: Reference model for KL divergence.
            tokenizer: Tokenizer for text processing.
            dataset: Dataset for training.
            reward_model: Reward model for feedback.
            device: Device to use for training.
        """
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.reward_model = reward_model
        self.optimize_device = optimize_device
        
        # Set device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                # Apply CUDA optimizations if requested
                if self.optimize_device:
                    print("Applying CUDA-specific optimizations for trainer")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                # Apply MPS optimizations if requested
                if self.optimize_device:
                    print("Applying MPS-specific optimizations for trainer")
                    # Apple Silicon specific optimizations could go here
            else:
                self.device = "cpu"
                
        print(f"PPO Trainer using device: {self.device}")
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Ensure reference model exactly matches the initial policy model 
        # before training begins (only needed for proper initialization)
        for ref_param, param in zip(self.ref_model.parameters(), self.model.parameters()):
            ref_param.data.copy_(param.data)
        
        print("Reference model initialized with exact copy of initial policy")
        
        # Setup PPO trainer
        self._setup_ppo_trainer()
        
        # Configure generation parameters
        self.output_min_length = config.output_min_length
        self.output_max_length = config.output_max_length
        self.output_length_sampler = LengthSampler(self.output_min_length, self.output_max_length)
        
        # Set generation parameters
        self.generation_kwargs = {
            "min_length": self.output_min_length,
            "max_new_tokens": self.output_max_length,
            "top_k": 30,
            "top_p": 0.9,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
        }
        
        # Training metrics
        self.best_avg_reward = float('-inf')
        self.best_model_state = None
        self.best_epoch = 0
        
        # Reward normalization statistics
        self.reward_running_mean = 0.0
        self.reward_running_std = 1.0
        self.reward_count = 0
        self.train_step_count = 0
        
        # KL divergence tracking
        self.kl_divergences = []
        
        # PPO specific parameters
        self.reward_coef = config.reward_coef
        self.kl_penalty = config.kl_penalty
        self.lm_loss_coef = config.lm_loss_coef
        self.clip_epsilon = config.clip_epsilon
        self.num_ppo_updates = config.num_ppo_updates
        self.entropy_coef = config.entropy_coef
        self.use_exploration = config.use_exploration
        
        # Setup optimizers with separate optimization for policy and value networks
        self.learning_rate = config.ppo_config.learning_rate
        self.value_lr_multiplier = config.value_lr_multiplier
        
        # Use separate optimizers for policy and value networks
        self.policy_optimizer = optim.Adam(
            self.model.pretrained_model.parameters(),
            lr=self.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.model.v_head.parameters(),
            lr=self.learning_rate * self.value_lr_multiplier  # Value head can train faster
        )
        print(f"Using separate optimization with value head learning rate multiplier of {self.value_lr_multiplier}")
            
        # Setup learning rate schedulers if enabled
        self.policy_lr_scheduler = None
        self.value_lr_scheduler = None
        
        if config.use_lr_scheduler:
            self.policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy_optimizer, 
                T_max=config.max_epochs,
                eta_min=self.learning_rate / 10
            )
            self.value_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.value_optimizer, 
                T_max=config.max_epochs,
                eta_min=(self.learning_rate * self.value_lr_multiplier) / 10
            )
            print("Using learning rate scheduling")
            
        # Gradient clipping values
        self.policy_grad_clip = config.policy_grad_clip
        self.value_grad_clip = config.value_grad_clip
        
        # Detailed metrics tracking
        self.metrics = {
            "policy_losses": [],
            "value_losses": [],
            "kl_divs": self.kl_divergences,  # Reuse existing KL tracking
            "entropy": [],
            "lm_losses": [],
            "rewards": []
        }
    
    def _setup_ppo_trainer(self):
        """Setup the PPO trainer."""
        # Data collator function for batching
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])
        
        # Create a simple dummy model to use as a placeholder reward model
        from transformers import AutoModelForCausalLM
        reward_model = AutoModelForCausalLM.from_pretrained(self.config.model_config.model_name)
        reward_model.generation_config = self.model.generation_config
        
        # Create a dedicated value model (same architecture as the policy model)
        value_model = AutoModelForCausalLM.from_pretrained(self.config.model_config.model_name)
        value_model.generation_config = self.model.generation_config
        
        # Make sure config has required properties
        ppo_config = self.config.ppo_config
        if not hasattr(ppo_config, "stop_token_id"):
            ppo_config.stop_token_id = self.tokenizer.eos_token_id
            
        if not hasattr(ppo_config, "stop_token"):
            ppo_config.stop_token = None
        
        # Initialize the PPO trainer
        self.ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=reward_model,  # Placeholder - we compute rewards manually
            train_dataset=self.dataset,
            data_collator=collator,
            value_model=value_model
        )
    
    def update_reference_model_ema(self):
        """
        Update reference model using exponential moving average.
        Keeps ref_ema_coef percentage of old weights and (1-ref_ema_coef) of new weights.
        """
        with torch.no_grad():
            # Get model parameters
            model_params = dict(self.model.named_parameters())
            ref_params = dict(self.ref_model.named_parameters())
            
            # Update reference model parameters with EMA
            for name, param in self.ref_model.named_parameters():
                if name in model_params:
                    # EMA update: new_val = alpha * old_val + (1 - alpha) * current_val
                    param.data.mul_(self.config.ref_ema_coef)
                    param.data.add_(model_params[name].data * (1.0 - self.config.ref_ema_coef))
            
            # Also update buffers like batch norm statistics if present
            for name, buffer in self.ref_model.named_buffers():
                if name in dict(self.model.named_buffers()):
                    buffer.copy_(dict(self.model.named_buffers())[name])
        
        print(f"Updated reference model with EMA coefficient {self.config.ref_ema_coef}")
    
    def normalize_rewards(self, rewards):
        """
        Normalize rewards using running statistics with exponential moving average.
        
        Args:
            rewards: Tensor of rewards to normalize
            
        Returns:
            Normalized rewards tensor
        """
        # Convert to numpy for statistics if it's a tensor
        if isinstance(rewards, torch.Tensor):
            rewards_np = rewards.detach().cpu().numpy()
        else:
            rewards_np = rewards
            
        # Update running statistics
        batch_size = len(rewards_np)
        batch_mean = float(rewards_np.mean())
        batch_var = float(rewards_np.var()) if batch_size > 1 else 0.0
        
        # Calculate new count with decay
        decay = 0.99 if self.reward_count > 0 else 0.0
        new_count = decay * self.reward_count + batch_size
        
        # Update running mean
        weight_old = decay * self.reward_count / new_count
        weight_new = batch_size / new_count
        self.reward_running_mean = weight_old * self.reward_running_mean + weight_new * batch_mean
        
        # Update running variance with Welford's online algorithm
        delta = batch_mean - self.reward_running_mean
        new_var = weight_old * (self.reward_running_std**2) + weight_new * batch_var + weight_old * weight_new * delta**2
        self.reward_running_std = math.sqrt(max(new_var, 1e-8))
        
        # Update count
        self.reward_count = new_count
        
        # Normalize rewards (avoid division by zero)
        normalized_rewards = (rewards - self.reward_running_mean) / (self.reward_running_std + 1e-8)
        
        # Log statistics periodically
        self.train_step_count += 1
        if self.train_step_count % 10 == 0:
            print(f"Reward stats: mean={self.reward_running_mean:.4f}, std={self.reward_running_std:.4f}")
        
        return normalized_rewards
        
    def compute_gae(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of rewards [batch_size]
            values: Tensor of value predictions [batch_size]
            gamma: Discount factor (typically 0.99)
            lam: GAE lambda parameter (typically 0.95)
            
        Returns:
            advantages: Tensor of advantage estimates [batch_size]
        """
        # Ensure inputs are tensors
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.device)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self.device)
            
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        
        # For single-step case, advantage is just reward - value
        if batch_size == 1:
            return rewards - values
            
        # For batch case, we need to compute GAE
        # Use the fact that next value is current value for next element in batch
        last_advantage = 0
        for t in reversed(range(batch_size - 1)):
            # Calculate TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * values[t+1] - values[t]
            
            # Compute GAE recursively: A_t = delta_t + gamma * lambda * A_{t+1}
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
        
        # The last element just uses the TD error
        advantages[-1] = rewards[-1] - values[-1]
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def compute_log_probs(self, model, input_ids, response_mask=None):
        """Compute log probabilities of a sequence."""
        with torch.no_grad():
            outputs = model.pretrained_model(input_ids)
            logits = outputs[0][:, :-1, :]  # Remove last token prediction
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get the log prob of the actual next token
            target_ids = input_ids[:, 1:]  # Shift right
            gathered_logprobs = log_probs.gather(dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
            
            # Apply mask if provided (to focus on response tokens only)
            if response_mask is not None:
                gathered_logprobs = gathered_logprobs * response_mask
                # Sum only over response tokens
                seq_log_probs = gathered_logprobs.sum(dim=1) / (response_mask.sum(dim=1) + 1e-5)
            else:
                seq_log_probs = gathered_logprobs.sum(dim=1) / gathered_logprobs.size(1)
                
            return seq_log_probs
    
    def _compute_losses(self, seq, mask, advantage, old_log_prob, query_length, reward):
        """
        Compute all losses for a single sequence.
        
        Args:
            seq: Input sequence (query + response)
            mask: Response mask (0 for query tokens, 1 for response tokens)
            advantage: Advantage estimate for this sequence
            old_log_prob: Old log probability from before the update
            query_length: Length of the query part of the sequence
            reward: Target reward value for value function
            
        Returns:
            Tuple of (lm_loss, policy_loss, value_loss, entropy, kl_div)
        """
        # Forward pass for policy
        outputs = self.model.pretrained_model(seq)
        logits = outputs[0]
        
        # Create labels for language modeling
        input_ids = seq
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # shift left by 1
        labels[:, -1] = -100  # ignore last token
        
        # Set tokens before the response to -100 to ignore them in loss calculation
        labels[:, :query_length] = -100
        
        # Compute language modeling loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Compute new log probability
        new_log_prob = self.compute_log_probs(self.model, seq, mask)
        
        # PPO ratio and clipped objective
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -policy_reward.mean()
        
        # Value head loss
        hidden_states = self.model.pretrained_model(seq, output_hidden_states=True).hidden_states[-1]
        value_pred = self.model.v_head(hidden_states[:, -1]).squeeze()
        # Use normalized rewards for value loss (reward is already normalized)
        value_loss = F.mse_loss(value_pred, reward)
        
        # Entropy bonus
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Calculate KL divergence from reference model
        with torch.no_grad():
            ref_outputs = self.ref_model.pretrained_model(seq)
            ref_logits = ref_outputs[0]
        
        kl_div = F.kl_div(
            F.log_softmax(logits[:, :-1, :], dim=-1),
            F.softmax(ref_logits[:, :-1, :], dim=-1),
            reduction='none'
        ).sum(-1)
        
        # Apply mask and compute mean
        masked_kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-5)
        
        # Track KL divergence
        self.kl_divergences.append(masked_kl_div.item())
        
        return lm_loss, policy_loss, value_loss, entropy, masked_kl_div
    
    def _optimize_sequence(self, seq, mask, advantage, old_log_prob, query_length, reward):
        """
        Optimize a single sequence using separate policy and value optimizers.
        
        Args:
            seq: Input sequence (query + response)
            mask: Response mask (0 for query tokens, 1 for response tokens)
            advantage: Advantage estimate for this sequence
            old_log_prob: Old log probability from before the update
            query_length: Length of the query part of the sequence
            reward: Target reward value for value function
            
        Returns:
            Tuple of (policy_loss_value, value_loss_value, entropy_value, lm_loss_value, kl_value)
        """
        # Store loss values to return
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        lm_loss_value = 0.0
        kl_div_value = 0.0
        
        # 1. Policy Optimization - perform in a separate forward pass
        # ---------------------------------------------------------
        self.policy_optimizer.zero_grad()
        
        # Use mixed precision if available
        autocast = getattr(self, 'mp_manager', None).get_autocast() if self.use_mixed_precision else nullcontext()
        
        # Import nullcontext here for non-mixed precision case
        from contextlib import nullcontext
        
        # Forward pass for policy with optional mixed precision
        with autocast:
            outputs = self.model.pretrained_model(seq)
            logits = outputs[0]
        
        # Create labels for language modeling
        input_ids = seq
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # shift left by 1
        labels[:, -1] = -100  # ignore last token
        
        # Set tokens before the response to -100 to ignore them in loss calculation
        labels[:, :query_length] = -100
        
        # Compute language modeling loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        lm_loss_value = lm_loss.item()
        
        # Compute new log probability for policy
        new_log_prob = self.compute_log_probs(self.model, seq, mask)
        
        # PPO ratio and clipped objective
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_reward = torch.min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -policy_reward.mean()
        policy_loss_value = policy_loss.item()
        
        # Entropy bonus
        probs = F.softmax(shift_logits, dim=-1)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        entropy_value = entropy.item()
        
        # Calculate KL divergence from reference model
        with torch.no_grad():
            ref_outputs = self.ref_model.pretrained_model(seq)
            ref_logits = ref_outputs[0]
        
        kl_div = F.kl_div(
            F.log_softmax(logits[:, :-1, :], dim=-1),
            F.softmax(ref_logits[:, :-1, :], dim=-1),
            reduction='none'
        ).sum(-1)
        
        # Apply mask and compute mean for KL
        masked_kl_div = (kl_div * mask).sum() / (mask.sum() + 1e-5)
        kl_div_value = masked_kl_div.item()
        
        # Track KL divergence
        self.kl_divergences.append(masked_kl_div.item())
        
        # Combine policy-related losses
        policy_combined_loss = (
            self.lm_loss_coef * lm_loss +         # Removed Language modeling component
            1.0 * policy_loss +     # PPO policy gradient component
            self.kl_penalty * masked_kl_div  # KL regularization component
        )
        
        # Add entropy bonus if exploration is enabled
        if self.use_exploration:
            policy_combined_loss -= self.entropy_coef * entropy
            
        # Backward pass for policy (only affects pretrained_model)
        if self.use_mixed_precision:
            # Use mixed precision backward
            self.mp_manager.backward(policy_combined_loss)
        else:
            # Standard backward
            policy_combined_loss.backward()
            
            # Clip policy gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.pretrained_model.parameters(), 
                max_norm=self.policy_grad_clip
            )
            
            # Update policy network
            self.policy_optimizer.step()
        
        # 2. Value Network Optimization (separate) - do in a completely new forward pass
        # ---------------------------------------------------------------------
        self.value_optimizer.zero_grad()
        
        # Do a separate forward pass for value head to avoid graph issues
        with torch.no_grad():
            hidden_states = self.model.pretrained_model(seq, output_hidden_states=True).hidden_states[-1]
        
        # Detach hidden states to avoid backward through policy network
        hidden_states_detached = hidden_states.detach()
        
        # Use mixed precision if available for value head computation
        with autocast:
            # Compute value prediction
            value_pred = self.model.v_head(hidden_states_detached[:, -1]).squeeze()
            
            # Calculate value loss with normalized reward
            value_loss = F.mse_loss(value_pred, reward)
        
        value_loss_value = value_loss.item()
        
        # Scale value loss (same as in original code) (adjusted this from 0.5)
        scaled_value_loss = value_loss
        
        # Backward pass for value network with mixed precision if enabled
        if self.use_mixed_precision:
            # Use mixed precision backward
            self.mp_manager.backward(scaled_value_loss, self.value_optimizer)
        else:
            # Standard backward
            scaled_value_loss.backward()
            
            # Clip value gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.v_head.parameters(), 
                max_norm=self.value_grad_clip
            )
            
            # Update value network
            self.value_optimizer.step()
        
        return (
            policy_loss_value,
            value_loss_value,
            entropy_value,
            lm_loss_value,
            kl_div_value
        )
    
    # _optimize_sequence_combined has been removed since we're only using separate optimization
    
    def train(self):
        """Run the training loop."""
        # Max epochs and early stopping parameters
        max_epochs = self.config.max_epochs
        patience = self.config.patience
        early_stopping_threshold = self.config.early_stopping_threshold
        epochs_no_improve = 0
        update_ref_freq = self.config.update_ref_freq
        reward_history = []
        
        # Store model device
        model_device = next(self.model.parameters()).device
        
        # Check for mixed precision capability
        # If the mixed precision manager was set by apply_trainer_optimizations, use it
        self.use_mixed_precision = hasattr(self, 'mp_manager')
        
        # Print training configuration
        print(f"Training configuration:")
        print(f"  Device: {self.device}")
        print(f"  Exploration (entropy bonus): {'Enabled' if self.use_exploration else 'Disabled'}")
        print(f"  KL penalty: {self.kl_penalty}")
        print(f"  Policy learning rate: {self.learning_rate}")
        print(f"  Value learning rate: {self.learning_rate * self.value_lr_multiplier}")
        
        # Print mixed precision status if enabled
        if self.use_mixed_precision:
            print(f"  Mixed precision: Enabled ({self.mp_manager.mixed_dtype})")
        print()
        
        # Print header
        print(f"{'Epoch':<6} {'RawReward':<10} {'NormReward':<10} {'Loss':<10} {'API Calls':<10} {'Cache Size':<10}")
        print("-" * 70)
        
        # Main training loop
        for epoch, batch in enumerate(self.ppo_trainer.dataloader):
            if epoch >= max_epochs:
                break
                
            # Track previous API calls for GPT-4 reward model
            if hasattr(self.reward_model, "num_api_calls"):
                prev_api_calls = self.reward_model.num_api_calls
            else:
                prev_api_calls = 0
                
            # Extract input tensors from the batch
            query_tensors = batch["input_ids"]
            
            # Step 1: Get response from GPT-2 (the policy model)
            response_tensors = []
            for query in query_tensors:
                # Create the query tensor with proper device
                query_tensor = query.unsqueeze(0).to(model_device)
                attention_mask = torch.ones_like(query_tensor, device=model_device)
                
                try:
                    response = self.model.generate(
                        input_ids=query_tensor,
                        attention_mask=attention_mask,
                        **self.generation_kwargs
                    )
                    
                    # Extract newly generated tokens
                    gen_len = min(response.size(1) - query_tensor.size(1), self.output_max_length)
                    response_tensors.append(response.squeeze()[query_tensor.size(1):query_tensor.size(1) + gen_len])
                except Exception as e:
                    print(f"Generation error: {e}")
                    continue
            
            # Skip if we couldn't generate any responses
            if len(response_tensors) == 0:
                continue
            
            # Decode responses for later inspection
            batch["response"] = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # Combine original query and generated response for sentiment analysis
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            
            # Get raw rewards from the reward model
            raw_rewards = self.reward_model(texts).to(model_device)
            
            # Store raw reward average for logging
            batch_raw_reward_avg = raw_rewards.mean().item()
            
            # Normalize rewards for training stability
            reward_tensor = self.normalize_rewards(raw_rewards)
            
            # Store full sequences for PPO updates
            full_sequences = []
            response_masks = []
            for query, response in zip(query_tensors, response_tensors):
                # Create full sequence (query + response)
                full_seq = torch.cat([query.to(model_device), response.to(model_device)], dim=0).unsqueeze(0)
                full_sequences.append(full_seq)
                
                # Create response mask (0 for query tokens, 1 for response tokens)
                mask = torch.zeros_like(full_seq[:, :-1])
                mask[:, len(query):] = 1
                response_masks.append(mask)
            
            # Compute value predictions before updating
            with torch.no_grad():
                value_preds = []
                for seq in full_sequences:
                    # Forward pass through value head
                    hidden_states = self.model.pretrained_model(seq, output_hidden_states=True).hidden_states[-1]
                    # Use the final token's hidden state for value prediction
                    value = self.model.v_head(hidden_states[:, -1]).squeeze()
                    value_preds.append(value)
                
                value_preds = torch.stack(value_preds)
            
            # Compute advantages using GAE with parameters from config
            advantages = self.compute_gae(
                reward_tensor, 
                value_preds, 
                gamma=self.config.gamma, 
                lam=self.config.gae_lambda
            )
            
            # Compute old log probs for importance sampling
            old_log_probs = []
            for i, (seq, mask) in enumerate(zip(full_sequences, response_masks)):
                old_log_prob = self.compute_log_probs(self.model, seq, mask)
                old_log_probs.append(old_log_prob)
            
            old_log_probs = torch.cat(old_log_probs)
            
            # PPO multiple update steps on the same batch of data
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            total_lm_loss = 0
            
            for ppo_step in range(self.num_ppo_updates):
                # Initialize loss tracking for this step
                step_policy_loss = 0
                step_value_loss = 0
                step_entropy = 0
                step_lm_loss = 0
                
                # Process each sequence in the batch
                for i, (seq, mask, advantage, old_log_prob) in enumerate(zip(full_sequences, response_masks, advantages, old_log_probs)):
                    # Process this training example with separate optimization for policy and value
                    pl, vl, ent, lml, kl = self._optimize_sequence(
                        seq, mask, advantage, old_log_prob, len(query_tensors[i]), reward_tensor[i]
                    )
                    
                    # Accumulate metrics
                    step_policy_loss += pl
                    step_value_loss += vl
                    step_entropy += ent
                    step_lm_loss += lml
                
                # Update learning rate schedulers if enabled
                if self.config.use_lr_scheduler:
                    if self.policy_lr_scheduler is not None:
                        self.policy_lr_scheduler.step()
                    if self.value_lr_scheduler is not None:
                        self.value_lr_scheduler.step()
                
                # Calculate average losses for this PPO step
                avg_policy_loss = step_policy_loss / len(full_sequences) if full_sequences else 0
                avg_value_loss = step_value_loss / len(full_sequences) if full_sequences else 0
                avg_entropy = step_entropy / len(full_sequences) if full_sequences else 0
                avg_lm_loss = step_lm_loss / len(full_sequences) if full_sequences else 0
                
                # Track metrics
                self.metrics["policy_losses"].append(avg_policy_loss)
                self.metrics["value_losses"].append(avg_value_loss)
                self.metrics["entropy"].append(avg_entropy)
                self.metrics["lm_losses"].append(avg_lm_loss)
                
                # Update totals for epoch average
                total_policy_loss += avg_policy_loss
                total_value_loss += avg_value_loss
                total_entropy += avg_entropy
                total_lm_loss += avg_lm_loss
            
            # Calculate current KL divergence (from the most recent update)
            current_kl = self.kl_divergences[-1] if self.kl_divergences else 0
            
            # Update reference model based on schedule OR if KL divergence gets too large
            if (epoch > 0 and epoch % self.config.update_ref_freq == 0) or current_kl > self.config.max_kl_target:
                if current_kl > self.config.max_kl_target:
                    print(f"KL divergence ({current_kl:.4f}) exceeded target ({self.config.max_kl_target:.4f}). Forcing reference model update.")
                self.update_reference_model_ema()
                print(f"Updated reference model at epoch {epoch} (EMA coefficient: {self.config.ref_ema_coef})")
                
            # Calculate new API calls made in this epoch
            if hasattr(self.reward_model, "num_api_calls"):
                new_api_calls = self.reward_model.num_api_calls - prev_api_calls
                cache_size = len(self.reward_model.reward_cache)
            else:
                new_api_calls = 0
                cache_size = 0
                
            # Print progress information
            avg_raw_reward = batch_raw_reward_avg
            avg_norm_reward = torch.mean(reward_tensor).item()
            reward_history.append(avg_raw_reward)  # Store raw rewards in history
            self.metrics["rewards"].append(avg_raw_reward)  # Also track in metrics
            
            # Calculate average loss for logging
            avg_policy_loss = total_policy_loss / self.num_ppo_updates
            
            print(f"{epoch:<6} {avg_raw_reward:<10.3f} {avg_norm_reward:<10.3f} {avg_policy_loss:<10.3f} {new_api_calls:<10} {cache_size:<10}")
            
            # Log detailed metrics periodically
            if epoch % 5 == 0:
                print(f"\nEpoch {epoch} detailed metrics:")
                print(f"  Policy loss: {total_policy_loss/self.num_ppo_updates:.4f}")
                print(f"  Value loss: {total_value_loss/self.num_ppo_updates:.4f}")
                print(f"  KL divergence: {current_kl:.4f}")
                print(f"  Entropy: {total_entropy/self.num_ppo_updates:.4f}")
                print(f"  LM loss: {total_lm_loss/self.num_ppo_updates:.4f}\n")
            
            # Track best model based on raw reward (not normalized)
            if avg_raw_reward > self.best_avg_reward + early_stopping_threshold:
                self.best_avg_reward = avg_raw_reward
                self.best_model_state = {
                    'model': self.model.state_dict(),
                    'epoch': epoch,
                    'reward': avg_raw_reward
                }
                self.best_epoch = epoch
                print(f"★ New best model with reward: {avg_raw_reward:.3f} ★")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            # Early stopping check
            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break
                
            # Show sample generations every 10 epochs
            if epoch % 10 == 0 and len(texts) > 0:
                print("\nSample generations:")
                num_samples = min(3, len(texts))  # Show up to 3 samples
                for i in range(num_samples):
                    print(f"Sample {i+1}:")
                    print(f"Text: {texts[i]}")
                    print(f"Reward: {reward_tensor[i].item():.3f}")
                    print("-" * 60)
                
            # Save model checkpoints
            if epoch % self.config.save_freq == 0 and epoch > 0:
                save_dir = os.path.join(self.config.ppo_config.output_dir, f"epoch-{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                from model_fix import save_model_with_proper_keys
                save_model_with_proper_keys(self.model, save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"Model saved to {save_dir}")
                
                # Save best model if available
                if self.best_model_state is not None:
                    best_save_dir = os.path.join(self.config.ppo_config.output_dir, f"best-epoch-{self.best_model_state['epoch']}")
                    os.makedirs(best_save_dir, exist_ok=True)
                    
                    # Fix the state dict keys before saving
                    fixed_state_dict = {}
                    for key, value in self.best_model_state['model'].items():
                        # Add the pretrained_model prefix to model weights
                        if not key.startswith('pretrained_model.') and key.startswith('transformer'):
                            fixed_state_dict[f'pretrained_model.{key}'] = value
                        else:
                            fixed_state_dict[key] = value
                    
                    # Create new model and load the fixed state dict
                    from trl import AutoModelForCausalLMWithValueHead
                    best_model_copy = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_config.model_name)
                    best_model_copy.load_state_dict(fixed_state_dict, strict=False)
                    save_model_with_proper_keys(best_model_copy, best_save_dir)
                    self.tokenizer.save_pretrained(best_save_dir)
                    print(f"Best model saved to {best_save_dir} (reward: {self.best_model_state['reward']:.3f})")
        
        # Return training statistics
        return {
            "best_reward": self.best_avg_reward,
            "best_epoch": self.best_epoch,
            "reward_history": reward_history,
            "epochs_trained": epoch + 1,
            "reward_stats": {
                "final_mean": self.reward_running_mean,
                "final_std": self.reward_running_std,
                "sample_count": self.reward_count
            },
            "kl_divergence_history": self.kl_divergences,
            "detailed_metrics": self.metrics
        }
    
    def compare_models(self, prompts, reward_model):
        """Compare original model with the trained model using GPT-4o rewards."""
        
        # Use the best model if available, otherwise use the current model
        if self.best_model_state is not None:
            from trl import AutoModelForCausalLMWithValueHead
            best_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.config.model_config.model_name)
            fixed_state_dict = {}
            for key, value in self.best_model_state['model'].items():
                if not key.startswith('pretrained_model.') and key.startswith('transformer'):
                    fixed_state_dict[f'pretrained_model.{key}'] = value
                else:
                    fixed_state_dict[key] = value
            best_model.load_state_dict(fixed_state_dict, strict=False)
            best_model.to(self.device)
            trained_model = best_model
        else:
            trained_model = self.model
            
        # Generate reviews with both original model and trained model (5 samples per prompt)
        print("Generating reviews with reference model...")
        original_reviews_nested = self.generate_reviews(self.ref_model, prompts)
        
        print("Generating reviews with trained model...")
        trained_reviews_nested = self.generate_reviews(trained_model, prompts)
        
        # Flatten nested reviews for evaluation
        original_reviews_flat = [review for prompt_reviews in original_reviews_nested for review in prompt_reviews]
        trained_reviews_flat = [review for prompt_reviews in trained_reviews_nested for review in prompt_reviews]
        
        # Evaluate rewards using GPT-4o
        print("Evaluating reference model reviews with GPT-4o...")
        original_scores_flat = reward_model(original_reviews_flat).tolist()
        
        print("Evaluating trained model reviews with GPT-4o...")
        trained_scores_flat = reward_model(trained_reviews_flat).tolist()
        
        # Reshape scores to match the nested structure of reviews
        original_scores = []
        trained_scores = []
        idx = 0
        for prompt_reviews in original_reviews_nested:
            original_scores.append(original_scores_flat[idx:idx+len(prompt_reviews)])
            idx += len(prompt_reviews)
        
        idx = 0
        for prompt_reviews in trained_reviews_nested:
            trained_scores.append(trained_scores_flat[idx:idx+len(prompt_reviews)])
            idx += len(prompt_reviews)
        
        # Calculate prompt-level averages
        original_prompt_avgs = [sum(scores)/len(scores) for scores in original_scores]
        trained_prompt_avgs = [sum(scores)/len(scores) for scores in trained_scores]
        
        # Calculate overall averages
        ref_avg = sum(original_prompt_avgs) / len(original_prompt_avgs) if original_prompt_avgs else 0
        trained_avg = sum(trained_prompt_avgs) / len(trained_prompt_avgs) if trained_prompt_avgs else 0
        avg_diff = trained_avg - ref_avg
        
        # Return comparison results
        return {
            "prompts": prompts,
            "ref_reviews": original_reviews_nested,
            "trained_reviews": trained_reviews_nested,
            "ref_scores": original_scores,
            "trained_scores": trained_scores,
            "ref_prompt_avgs": original_prompt_avgs,
            "trained_prompt_avgs": trained_prompt_avgs,
            "score_differences": [t - o for t, o in zip(trained_prompt_avgs, original_prompt_avgs)],
            "ref_avg": ref_avg,
            "trained_avg": trained_avg,
            "avg_diff": avg_diff
        }
    
    def generate_reviews(self, model, prompts, max_length=50, num_samples=5):
        """
        Generate text from a model.
        
        Args:
            model: The model to generate text with
            prompts: List of prompts to generate from
            max_length: Maximum length of generated text
            num_samples: Number of samples to generate for each prompt
            
        Returns:
            List of lists, where each inner list contains generated texts for a prompt
        """
        model.eval()  # Set model to evaluation mode
        all_reviews = []
        
        for prompt in prompts:
            prompt_reviews = []
            # Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate multiple examples for this prompt
            for _ in range(num_samples):
                # Generate text with the model
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prompt_reviews.append(generated_text)
            
            all_reviews.append(prompt_reviews)
        
        return all_reviews