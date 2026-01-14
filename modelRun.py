import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor
class ConcurrentMLStrategy:
    def __init__(self, model1, model2, n_workers=2):
        self.model1 = model1
        self.model2 = model2
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        
    def predict_concurrent(self, features_lstm, features_linear):
        """Run both models in parallel"""
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
    
    def _predict_model1(self, features):
        # LSTM model expects (batch_size, 30, 6)
        return self.model1.predict(features, verbose=0)[0]
    
    def _predict_model2(self, features):
        # Regularized Linear model expects (batch_size, 4)
        return self.model2.predict(features, verbose=0)[0]

def load_model(path):
    return tf.keras.models.load_model(path)
def load_data(path):
    df = pd.read_csv(path, parse_dates=['Dates'])
    df.sort_values('Dates', inplace=True)
    return df
def preprocess_data(df, feature_cols, target_col, sequence_length=30):
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Create sequences for time series models
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)
def run_model_comparison(data_path, model1_path, model2_path):
    # Load data
    df = load_data(data_path)
    feature_cols_lstm = ['Open', 'Close', 'High', 'Low', 'Volume', 'Number Ticks']  # All 6 features for LSTM
    feature_cols_linear = ['Open', 'Close', 'High', 'Low']  # First 4 features for linear model
    target_col = 'Close'
    
    # Preprocess data for LSTM
    X_lstm, y = preprocess_data(df, feature_cols_lstm, target_col, sequence_length=30)
    
    # Preprocess data for linear model - use only last values from the LSTM sequence
    X_linear = X_lstm[:, -1, :4]  # Take last timestep and first 4 features
    
    # Load models
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # Initialize concurrent strategy
    strategy = ConcurrentMLStrategy(model1, model2)
    
    # Run predictions concurrently
    results = []
    for i in range(len(X_lstm)):
        features_lstm = X_lstm[i:i+1]  # Shape: (1, 30, 6)
        features_linear = X_linear[i:i+1]  # Shape: (1, 4)
        result = strategy.predict_concurrent(features_lstm, features_linear)
        results.append(result)
    
    return results
if __name__ == "__main__":
    data_path = "../../../desktop/quant/hist/aaplIntra.csv"
    model1_path = "models/lstm.keras"
    model2_path = "models/regularizedLinear.keras"
    
    results = run_model_comparison(data_path, model1_path, model2_path)
    
    # Print results
    for res in results:
        print(f"Model1 Prediction: {res['model1']}, Model2 Prediction: {res['model2']}, Latency: {res['latency_ms']:.2f} ms")