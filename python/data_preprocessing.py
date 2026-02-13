"""Data loading and preprocessing for backtesting"""
import numpy as np
import pandas as pd
from typing import Tuple, List


def load_data(path: str) -> pd.DataFrame:
    """
    Load market data from CSV.
    
    Args:
        path: Path to CSV file with market data
    
    Returns:
        DataFrame with sorted data by date
    """
    df = pd.read_csv(path, parse_dates=['Dates'])
    df.sort_values('Dates', inplace=True)
    return df


def preprocess_lstm_data(df: pd.DataFrame, feature_cols: List[str], sequence_length: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for LSTM model (price prediction).
    
    Args:
        df: DataFrame with market data
        feature_cols: List of column names to use as features
        sequence_length: Length of sequences for LSTM
    
    Returns:
        Tuple of (X_sequences, y_sequences) numpy arrays
    """
    X = df[feature_cols].values.astype('float32')
    close = df['Close'].values.astype('float32')
    
    # Next bar price as target
    y = np.roll(close, -1)
    
    # Drop last bar (no valid target)
    X = X[:-1]
    y = y[:-1]
    
    # Create sequences for time series models
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length-1])
    
    return np.array(X_sequences), np.array(y_sequences)


def preprocess_linear_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for linear model (price prediction).
    Must match the feature engineering used during training (15 features).
    
    Args:
        df: DataFrame with market data
    
    Returns:
        Tuple of (X, y) numpy arrays with features and targets
    """
    df_copy = df.copy()
    
    # 1-step ahead Close price target
    df_copy["y"] = df_copy["Close"].shift(-1)
    
    # Lag returns: 1, 2, 3, 5, 10, 20
    for k in [1, 2, 3, 5, 10, 20]:
        df_copy[f"ret_lag_{k}"] = df_copy["Close"].pct_change(k)
    
    # Volatility features: rolling std of returns at windows 5, 10, 20
    daily_ret = df_copy["Close"].pct_change()
    for w in [5, 10, 20]:
        df_copy[f"volatility_{w}"] = daily_ret.rolling(w).std()
    
    # Volume features
    df_copy["volume_norm"] = df_copy["Volume"] / df_copy["Volume"].rolling(20).mean()
    df_copy["volume_change"] = df_copy["Volume"].pct_change()
    
    # Close relative to moving averages
    for w in [5, 10, 20]:
        df_copy[f"close_ma_{w}"] = df_copy["Close"] / df_copy["Close"].rolling(w).mean()
    
    # High-Low range
    df_copy["hl_range"] = (df_copy["High"] - df_copy["Low"]) / df_copy["Close"]
    
    # Drop NAs
    df_copy = df_copy.dropna()
    
    feature_cols = [c for c in df_copy.columns 
                    if c.startswith(("ret_lag_", "volatility_", "volume_", "close_ma_")) or c == "hl_range"]
    X = df_copy[feature_cols].values.astype('float32')
    y = df_copy["y"].values.astype('float32')
    
    return X, y
