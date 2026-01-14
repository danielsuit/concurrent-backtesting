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
        
        # submit both predictions to thread pool
        future1 = self.executor.submit(self._predict_model1, features_lstm)
        future2 = self.executor.submit(self._predict_model2, features_linear)
        
        # wait for both to complete
        pred1 = future1.result()
        pred2 = future2.result()
        
        latency = time.time() - start
        
        return {
            'model1': pred1,
            'model2': pred2,
            'latency_ms': latency * 1000
        }
    
    def _predict_model1(self, features):
        return self.model1.predict(features, verbose=0)[0]
    
    def _predict_model2(self, features):
        return self.model2.predict(features, verbose=0)[0]

def load_model(path):
    return tf.keras.models.load_model(path)
def load_data(path):
    df = pd.read_csv(path, parse_dates=['Dates'])
    df.sort_values('Dates', inplace=True)
    return df
def preprocess_data(df, feature_cols, sequence_length=60):
    """Preprocess data for LSTM model (price prediction)"""
    X = df[feature_cols].values.astype('float32')
    close = df['Close'].values.astype('float32')
    
    # next bar price as target
    y = np.roll(close, -1)
    
    # drop last bar (no valid target)
    X = X[:-1]
    y = y[:-1]
    
    # create sequences for time series models
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length-1])
    
    return np.array(X_sequences), np.array(y_sequences)

def preprocess_linear_data(df, sequence_length=30):
    """Preprocess data for linear model (log-returns)"""
    df_copy = df.copy()
    df_copy["log_close"] = np.log(df_copy["Close"])
    df_copy["y"] = df_copy["log_close"].shift(-1) - df_copy["log_close"]
    
    #  lag features
    for k in [1, 2, 3, 5]:
        df_copy[f"ret_lag_{k}"] = df_copy["log_close"].diff(k)
    
    # drop NAs
    df_copy = df_copy.dropna()
    
    feature_cols = [c for c in df_copy.columns if c.startswith("ret_lag_")]
    X = df_copy[feature_cols].values.astype('float32')
    y = df_copy["y"].values.astype('float32')
    
    return X, y
def run_model_comparison(data_path, model1_path, model2_path, max_samples=10):
    df = load_data(data_path)
    
    # LSTM model preprocessing (predicts next-bar price)
    feature_cols_lstm = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number Ticks']
    X_lstm, y_price = preprocess_data(df, feature_cols_lstm, sequence_length=60)
    
    # linear model preprocessing
    X_linear, y_logret = preprocess_linear_data(df)
    
    # load models
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # init concurrent strategy
    strategy = ConcurrentMLStrategy(model1, model2)
    
    # run predictions concurrently on random samples
    results = []
    
    # get valid indices (must have both LSTM and linear data)
    max_idx = min(len(X_lstm), len(X_linear))
    num_samples = min(max_samples, max_idx)
    
    # randomly sample indices spread across the dataset
    np.random.seed(42)  # for reproducibility
    sample_indices = np.random.choice(max_idx, size=num_samples, replace=False)
    sample_indices = np.sort(sample_indices)  # sort for cleaner output
    
    print(f"Running predictions on {num_samples} random samples across dataset...")
    for count, i in enumerate(sample_indices):
        features_lstm = X_lstm[i:i+1]
        features_linear = X_linear[i:i+1]
        actual_logret = y_logret[i]
            
        result = strategy.predict_concurrent(features_lstm, features_linear)
        result['actual_price'] = y_price[i]  # actual next-bar price
        result['actual_logret'] = actual_logret  # continuous
        result['sample_idx'] = i  # track which index was sampled
        results.append(result)
        if (count + 1) % max(1, num_samples // 5) == 0:
            print(f"Completed {count + 1}/{num_samples} predictions")
    
    return results
if __name__ == "__main__":
    data_path = "../../../desktop/quant/hist/aaplIntra.csv"
    model1_path = "models/lstm.keras"
    model2_path = "models/regularizedLinear.keras"
    
    results = run_model_comparison(data_path, model1_path, model2_path, max_samples=5)
    print("\n" + "-"*120)
    print("MODEL PREDICTIONS COMPARISON (Random Samples)")
    print("-"*120)
    print(f"{'Sample':<8} {'Index':<10} {'Actual Price':<14} {'LSTM Pred':<14} {'Price Error':<14} {'Log-Ret':<12} {'Log-Ret Pred':<14} {'Latency':<10}")
    print("-"*120)
    
    price_errors = []
    
    for i, res in enumerate(results):
        actual_price = res['actual_price']
        actual_logret = res['actual_logret']
        sample_idx = res['sample_idx']
        
        lstm_pred = res['model1'].item() if isinstance(res['model1'], np.ndarray) else res['model1']
        linear_pred = res['model2'].item() if isinstance(res['model2'], np.ndarray) else res['model2']
        
        price_error = lstm_pred - actual_price
        price_errors.append(abs(price_error))
        
        print(f"{i+1:<8} {sample_idx:<10} ${actual_price:<13.4f} ${lstm_pred:<13.4f} {price_error:>+13.4f} {actual_logret:<12.6f} {linear_pred:<14.6f} {res['latency_ms']:<9.2f}ms")
    
    print("-"*120)
    print(f"\nModel Performance Summary:")
    print(f"LSTM Model - Mean Absolute Error: ${np.mean(price_errors):.4f}")
    print(f"Linear Model - Predicts log-returns for next bar")