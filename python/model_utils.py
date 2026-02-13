"""Model loading and utilities"""
import tensorflow as tf
import numpy as np
import joblib
from typing import List, Dict, Any


class SklearnModelWrapper:
    """Wraps an sklearn model (with optional scalers) to have a Keras-compatible .predict() interface."""
    
    def __init__(self, model, scaler=None, y_scaler=None):
        self.model = model
        self.scaler = scaler
        self.y_scaler = y_scaler
    
    def predict(self, X, verbose=0):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        if self.y_scaler is not None:
            y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        return y_pred.reshape(-1, 1)


def load_model(path: str):
    """
    Load a model from file. Supports .keras (TF) and .joblib (sklearn).
    
    Args:
        path: Path to model file
    
    Returns:
        Loaded model with .predict() interface
    """
    if path.endswith('.joblib'):
        data = joblib.load(path)
        if isinstance(data, dict):
            return SklearnModelWrapper(
                model=data['model'],
                scaler=data.get('scaler'),
                y_scaler=data.get('y_scaler')
            )
        else:
            return SklearnModelWrapper(model=data)
    else:
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
