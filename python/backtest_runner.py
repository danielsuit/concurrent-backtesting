"""Main concurrent backtesting engine"""
import warnings
import numpy as np
import tensorflow as tf

from data_preprocessing import load_data, preprocess_lstm_data, preprocess_linear_data
from model_utils import load_model, run_predictions
from model_executor import ConcurrentMLStrategy
from risk_analytics import ResultAggregator
from reporting import print_prediction_comparison

warnings.filterwarnings('ignore')


def run_model_comparison(data_path: str, model1_path: str, model2_path: str, max_samples: int = 10):
    """
    Run concurrent model comparison and backtesting.
    
    Args:
        data_path: Path to market data CSV
        model1_path: Path to LSTM model
        model2_path: Path to linear model
        max_samples: Maximum number of samples to test
    
    Returns:
        Tuple of (results, price_series) for analysis
    """
    # Load and preprocess data
    print("Loading data...")
    df = load_data(data_path)
    
    # LSTM model preprocessing (predicts next-bar price)
    feature_cols_lstm = ['Open', 'High', 'Low', 'Close', 'Volume', 'Number Ticks']
    X_lstm, y_price = preprocess_lstm_data(df, feature_cols_lstm, sequence_length=600)
    
    # Linear model preprocessing (now predicts next-bar price)
    X_linear, y_linear = preprocess_linear_data(df)
    
    # Load models
    print("Loading models...")
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # Initialize concurrent strategy
    strategy = ConcurrentMLStrategy(model1, model2)
    
    # Run predictions concurrently on random samples
    results, sample_indices = run_predictions(strategy, X_lstm, X_linear, max_samples)
    
    # Attach actual values to results
    for i, res in enumerate(results):
        sample_idx = res['sample_idx']
        res['actual_price'] = y_price[sample_idx]
        res['actual_linear'] = y_linear[sample_idx]
    
    # Extract price series for analytics
    price_series = df['Close'].values[sample_indices].astype('float32')
    
    return results, price_series


def main():
    """Main execution function"""
    # Configuration
    data_path = "../../../../desktop/quant/hist/aaplIntra.csv"
    model1_path = "../training/models/lstm.keras"
    model2_path = "../training/models/elasticNet.joblib"
    
    # Run model comparison
    results, price_series = run_model_comparison(data_path, model1_path, model2_path, max_samples=100)
    
    # Extract predictions and actuals for analytics
    lstm_predictions = np.array([
        res['model1'].item() if isinstance(res['model1'], np.ndarray) else res['model1'] 
        for res in results
    ])
    linear_predictions = np.array([
        res['model2'].item() if isinstance(res['model2'], np.ndarray) else res['model2'] 
        for res in results
    ])
    lstm_actuals = np.array([res['actual_price'] for res in results])
    linear_actuals = np.array([res['actual_linear'] for res in results])
    
    # ==========================================================================
    # BASIC PREDICTION COMPARISON OUTPUT
    # ==========================================================================
    print_prediction_comparison(results, lstm_predictions, linear_predictions, lstm_actuals, linear_actuals)
    
    # Average latency
    avg_latency = np.mean([res['latency_ms'] for res in results])
    print(f"\nAverage Inference Latency: {avg_latency:.2f}ms")
    
    # ==========================================================================
    # VECTORIZED RISK ANALYTICS & RESEARCH-GRADE DIAGNOSTICS
    # ==========================================================================
    
    # Initialize result aggregator
    aggregator = ResultAggregator()
    
    # Add both strategy variants for comparison
    aggregator.add_strategy_results(
        name="LSTM_Model",
        predictions=lstm_predictions,
        actuals=lstm_actuals,
        prices=price_series
    )
    
    aggregator.add_strategy_results(
        name="ElasticNet_Linear",
        predictions=linear_predictions,
        actuals=linear_actuals,
        prices=price_series
    )
    
    # Generate and print comprehensive diagnostics
    aggregator.print_diagnostics_summary()


if __name__ == "__main__":
    main()
