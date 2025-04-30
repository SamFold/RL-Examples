#!/bin/bash

# Script to view training history without running a new training session

if [ $# -eq 0 ]; then
  # No arguments provided, use default file
  HISTORY_FILE="trainer_output/training_results.csv"
  METRIC="improvement"
else
  # Use provided file path
  HISTORY_FILE="$1"
  
  if [ $# -eq 2 ]; then
    # Use provided metric
    METRIC="$2"
  else
    # Default metric
    METRIC="improvement"
  fi
fi

# Print header
echo "Viewing training history from: $HISTORY_FILE"
echo "Sorting by metric: $METRIC"
echo "--------------------------------------------------------"

# Run the history viewer
python main.py --view_history --history_file "$HISTORY_FILE" --history_metric "$METRIC"

echo "--------------------------------------------------------"
echo "To view with different metrics, use: ./view_history.sh [file_path] [metric]"
echo "Available metrics: improvement, best_reward, trained_avg_score, etc."