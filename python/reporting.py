"""Reporting and output utilities"""
import numpy as np
from typing import List, Dict, Any


def print_prediction_comparison(results: List[Dict[str, Any]], lstm_predictions: np.ndarray,
                               linear_predictions: np.ndarray, lstm_actuals: np.ndarray,
                               linear_actuals: np.ndarray):
    """
    Print formatted comparison of model predictions.
    
    Args:
        results: List of prediction result dicts
        lstm_predictions: LSTM predictions array
        linear_predictions: Linear predictions array
        lstm_actuals: LSTM actual values array
        linear_actuals: Linear actual values array
    """
    print("\n" + "-"*130)
    print("MODEL PREDICTIONS COMPARISON (Random Samples) - Both Models Predict Next-Bar Price")
    print("-"*130)
    print(f"{'Sample':<8} {'Index':<10} {'Actual Price':<14} {'LSTM Pred':<14} {'LSTM Error':<14} {'Linear Pred':<14} {'Linear Error':<14} {'Latency':<10}")
    print("-"*130)
    
    lstm_errors = []
    linear_errors = []
    
    # Show first 10 samples for brevity
    display_samples = min(10, len(results))
    for i, res in enumerate(results[:display_samples]):
        actual_price = res['actual_price']
        actual_linear = res['actual_linear']
        sample_idx = res['sample_idx']
        lstm_pred = res['model1'].item() if isinstance(res['model1'], np.ndarray) else res['model1']
        linear_pred = res['model2'].item() if isinstance(res['model2'], np.ndarray) else res['model2']
        lstm_error = lstm_pred - actual_price
        linear_error = linear_pred - actual_linear
        lstm_errors.append(abs(lstm_error))
        linear_errors.append(abs(linear_error))
        print(f"{i+1:<8} {sample_idx:<10} ${actual_price:<13.4f} ${lstm_pred:<13.4f} {lstm_error:>+13.4f} ${linear_pred:<13.4f} {linear_error:>+13.4f} {res['latency_ms']:<9.2f}ms")
    
    if len(results) > display_samples:
        print(f"... ({len(results) - display_samples} more samples)")
    
    print("-"*130)
    print(f"\nBasic Model Performance Summary:")
    print(f"LSTM Model   - Mean Absolute Error: ${np.mean([abs(lstm_predictions[i] - lstm_actuals[i]) for i in range(len(results))]):.4f}")
    print(f"Linear Model - Mean Absolute Error: ${np.mean([abs(linear_predictions[i] - linear_actuals[i]) for i in range(len(results))]):.4f}")
