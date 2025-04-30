"""Training results logger for RLHF experiments."""

import os
import json
from datetime import datetime
import pandas as pd


class TrainingLogger:
    """Logs training parameters and results to a file for experiment tracking."""
    
    def __init__(self, log_file="training_results.csv"):
        """
        Initialize the training logger.
        
        Args:
            log_file: Path to the CSV file for tracking results.
        """
        self.log_file = log_file
    
    def log_training_run(self, params, results):
        """
        Log a training run with its parameters and results.
        
        Args:
            params: Dictionary of training parameters
            results: Dictionary of training results
        """
        # Create a single row record
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record = {
            "timestamp": timestamp,
            
            # Key parameters
            "batch_size": params.get("batch_size"),
            "learning_rate": params.get("learning_rate"),
            "lm_loss_coef": params.get("lm_loss_coef"),
            "num_ppo_updates": params.get("num_ppo_updates"),
            "use_exploration": params.get("use_exploration"),
            
            # KL divergence parameters
            "max_kl_target": params.get("max_kl_target"),
            "kl_penalty": params.get("kl_penalty"),
            "update_ref_freq": params.get("update_ref_freq"),
            "ref_ema_coef": params.get("ref_ema_coef"),
            
            "value_lr_multiplier": params.get("value_lr_multiplier"),
            "mixed_precision": params.get("mixed_precision"),
            
            # Training results
            "best_reward": results.get("best_reward"),
            "best_epoch": results.get("best_epoch"),
            "total_epochs": results.get("epochs_trained"),
            "ref_avg_score": results.get("ref_avg"),
            "trained_avg_score": results.get("trained_avg"),
            "improvement": results.get("avg_diff"),
            
            # KL divergence statistics
            "kl_max": results.get("kl_max"),
            "kl_min": results.get("kl_min"),
            "kl_avg": results.get("kl_avg"),
            "kl_final": results.get("kl_final"),
            
            # Optional additional info
            "model_name": params.get("model_name"),
            "device": params.get("device"),
            "output_dir": params.get("output_dir"),
        }
        
        # Load existing log or create new one
        if os.path.exists(self.log_file):
            try:
                df = pd.read_csv(self.log_file)
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing log file: {e}")
                df = pd.DataFrame([record])
        else:
            df = pd.DataFrame([record])
        
        # Save to CSV
        df.to_csv(self.log_file, index=False)
        print(f"Training results logged to {self.log_file}")
        
        # Also save detailed results as JSON for this specific run
        run_id = f"run_{timestamp.replace(' ', '_').replace(':', '-')}"
        detailed_log_file = f"{os.path.splitext(self.log_file)[0]}_{run_id}.json"
        
        # Combine params and results
        detailed_record = {
            "parameters": params,
            "results": results
        }
        
        with open(detailed_log_file, "w") as f:
            json.dump(detailed_record, f, indent=2, default=str)
        
        print(f"Detailed results saved to {detailed_log_file}")
        
        return record
    
    def get_summary(self, n_recent=5):
        """
        Get a summary of recent training runs.
        
        Args:
            n_recent: Number of most recent runs to include
            
        Returns:
            DataFrame with recent training runs
        """
        if not os.path.exists(self.log_file):
            return "No training logs found."
            
        try:
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return "No training logs found."
                
            # Sort by timestamp and get the n most recent runs
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False).head(n_recent)
            
            # Reset index for better display
            df = df.reset_index(drop=True)
            
            return df
        except Exception as e:
            return f"Error reading training logs: {e}"
            
    def get_best_runs(self, metric="improvement", n_best=3):
        """
        Get the best performing training runs based on specified metric.
        
        Args:
            metric: Metric to sort by (improvement, best_reward, etc.)
            n_best: Number of best runs to return
            
        Returns:
            DataFrame with best training runs
        """
        if not os.path.exists(self.log_file):
            return "No training logs found."
            
        try:
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return "No training logs found."
            
            # Sort by the specified metric in descending order
            if metric in df.columns:
                df = df.sort_values(metric, ascending=False).head(n_best)
                df = df.reset_index(drop=True)
                return df
            else:
                return f"Metric '{metric}' not found in training logs."
        except Exception as e:
            return f"Error reading training logs: {e}"