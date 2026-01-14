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

def preprocess_linear_data(df, sequence_length=30):
    """Preprocess data for linear model (log-returns)"""
    df_copy = df.copy()
    df_copy["log_close"] = np.log(df_copy["Close"])
    df_copy["y"] = df_copy["log_close"].shift(-1) - df_copy["log_close"]
    
    # Simple lag features
    for k in [1, 2, 3, 5]:
        df_copy[f"ret_lag_{k}"] = df_copy["log_close"].diff(k)
    
    # Drop NAs
    df_copy = df_copy.dropna()
    
    feature_cols = [c for c in df_copy.columns if c.startswith("ret_lag_")]
    X = df_copy[feature_cols].values.astype('float32')
    y = df_copy["y"].values.astype('float32')
    
    return X, y
def run_model_comparison(data_path, model1_path, model2_path, max_samples=10):
    # Load data
    df = load_data(data_path)
    
    # LSTM model preprocessing
    feature_cols_lstm = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number Ticks']
    X_lstm, y_direction = preprocess_data(df, feature_cols_lstm, 'Close', sequence_length=30)
    
    # Linear model preprocessing
    X_linear, y_logret = preprocess_linear_data(df)
    
    # Load models
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # Initialize concurrent strategy
    strategy = ConcurrentMLStrategy(model1, model2)
    
    # Run predictions concurrently (limit to max_samples for testing)
    results = []
    num_samples = min(max_samples, len(X_lstm))
    print(f"Running predictions on {num_samples} samples...")
    for i in range(num_samples):
        features_lstm = X_lstm[i:i+1]  # Shape: (1, 30, 6)
        
        # For linear model, match the index properly
        if i < len(X_linear):
            features_linear = X_linear[i:i+1]  # Shape: (1, 4)
            actual_logret = y_logret[i]
        else:
            continue
            
        result = strategy.predict_concurrent(features_lstm, features_linear)
        result['actual_direction'] = y_direction[i]  # 0 or 1
        result['actual_logret'] = actual_logret  # continuous
        result['actual_price'] = df['Close'].iloc[i+30]  # actual price at that point
        results.append(result)
        if (i + 1) % max(1, num_samples // 5) == 0:
            print(f"Completed {i + 1}/{num_samples} predictions")
    
    return results
if __name__ == "__main__":
    data_path = "../../../desktop/quant/hist/aaplIntra.csv"
    model1_path = "models/lstm.keras"
    model2_path = "models/regularizedLinear.keras"
    
    results = run_model_comparison(data_path, model1_path, model2_path, max_samples=5)
    
    # Print results
    print("\n" + "-"*100)
    print("MODEL PREDICTIONS COMPARISON")
    print("-"*100)
    print(f"{'Sample':<8} {'Price':<12} {'Direction':<12} {'LSTM Pred':<12} {'Log-Ret':<12} {'Log-Ret Pred':<14} {'Latency':<10}")
    print(f"{'':8} {'(Actual)':<12} {'(Actual)':<12} {'(0-1)':<12} {'(Actual)':<12} {'(Predicted)':<14}")
    print("-"*100)
    
    lstm_correct = 0
    
    for i, res in enumerate(results):
        price = res['actual_price']
        direction = int(res['actual_direction'])
        actual_logret = res['actual_logret']
        
        lstm_pred = res['model1'].item() if isinstance(res['model1'], np.ndarray) else res['model1']
        linear_pred = res['model2'].item() if isinstance(res['model2'], np.ndarray) else res['model2']
        
        # LSTM predicted direction (round to 0 or 1)
        lstm_direction_pred = round(lstm_pred)
        if lstm_direction_pred == direction:
            lstm_correct += 1
        
        print(f"{i+1:<8} ${price:<11.4f} {direction:<12} {lstm_pred:<12.4f} {actual_logret:<12.6f} {linear_pred:<14.6f} {res['latency_ms']:<9.2f}ms")
    
    print("-"*100)
    print(f"\nModel Performance Summary:")
    print(f"LSTM Model - Direction Prediction Accuracy: {lstm_correct}/{len(results)} ({100*lstm_correct/len(results):.1f}%)")
    print(f"Linear Model - Predicts log-returns for next bar")