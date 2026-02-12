"""Concurrent model execution for multi-model inference"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any


class ConcurrentMLStrategy:
    """Execute multiple ML models in parallel"""
    
    def __init__(self, model1, model2, n_workers=2):
        """
        Initialize concurrent strategy.
        
        Args:
            model1: First model (e.g., LSTM)
            model2: Second model (e.g., Linear)
            n_workers: Number of parallel workers
        """
        self.model1 = model1
        self.model2 = model2
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
    
    def predict_concurrent(self, features_lstm: Any, features_linear: Any) -> Dict:
        """
        Run both models in parallel.
        
        Args:
            features_lstm: Input features for model1
            features_linear: Input features for model2
        
        Returns:
            Dictionary with model1, model2 predictions and latency
        """
        start = time.time()
        
        # Submit both predictions to thread pool
        future1 = self.executor.submit(self._predict_model1, features_lstm)
        future2 = self.executor.submit(self._predict_model2, features_linear)
        
        # Wait for both to complete
        pred1 = future1.result()
        pred2 = future2.result()
        
        latency = time.time() - start
        
        return {
            'model1': pred1,
            'model2': pred2,
            'latency_ms': latency * 1000
        }
    
    def _predict_model1(self, features: Any) -> Any:
        """Predict using model1 (LSTM)"""
        return self.model1.predict(features, verbose=0)[0]
    
    def _predict_model2(self, features: Any) -> Any:
        """Predict using model2 (Linear)"""
        return self.model2.predict(features, verbose=0)[0]
    
    def __del__(self):
        """Clean up thread pool"""
        self.executor.shutdown(wait=True)
