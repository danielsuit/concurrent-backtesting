"""Model loading and utilities"""
import tensorflow as tf
import numpy as np
from typing import List, Dict, Any


def load_model(path: str) -> tf.keras.Model:
    """
    Load a Keras model from file.
    
    Args:
        path: Path to model file
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(path)


def run_predictions(strategy, features_lstm: np.ndarray, features_linear: np.ndarray, 
                   num_samples: int) -> List[Dict[str, Any]]:
    """
    Run concurrent predictions on samples.
    
    Args:
        strategy: ConcurrentMLStrategy instance
        features_lstm: LSTM features array
        features_linear: Linear features array
        num_samples: Number of samples to process
    
    Returns:
        List of prediction results
    """
    max_idx = min(len(features_lstm), len(features_linear))
    num_samples = min(num_samples, max_idx)
    
    # Randomly sample indices spread across the dataset
    np.random.seed(42)  # for reproducibility
    sample_indices = np.random.choice(max_idx, size=num_samples, replace=False)
    sample_indices = np.sort(sample_indices)  # sort for cleaner output
    
    print(f"Running predictions on {num_samples} random samples across dataset...")
    
    results = []
    for count, i in enumerate(sample_indices):
        features_lstm_sample = features_lstm[i:i+1]
        features_linear_sample = features_linear[i:i+1]
        
        result = strategy.predict_concurrent(features_lstm_sample, features_linear_sample)
        result['sample_idx'] = i
        results.append(result)
        
        if (count + 1) % max(1, num_samples // 5) == 0:
            print(f"Completed {count + 1}/{num_samples} predictions")
    
    return results, sample_indices
