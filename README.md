# concurrent-backtesting

Concurrent backtesting engine for comparing and blending two price forecasting models on intraday equity data:

1. LSTM sequence model
2. Regularized linear model (Elastic Net / L2-style regularization)

The project runs both models in parallel, generates trading signals from their next-step price predictions, and evaluates strategy quality with research-style risk metrics.

## What this project does

- Uses historical intraday stock data (10-second bars, 6-month window in the current setup).
- Loads trained models from `training/models`.
- Executes model inference concurrently:
	- Python path: `ThreadPoolExecutor`
	- C++ path: `std::async`
- Simulates strategy behavior from predictions.
- Reports diagnostics such as:
	- Sharpe ratio
	- Sortino ratio
	- Maximum drawdown
	- VaR (loss percentile)
	- Prediction error metrics (MAE, RMSE, etc.)

## Repository layout

- `training/`
	- Model training notebooks (`lstm.ipynb`, regularized linear variants, Elastic Net)
	- Export script: `export_models.py`
	- Exported artifacts for Python and C++ runtimes (`.keras`, `.joblib`, `.json`)
- `python/`
	- Concurrent backtest runner (`backtest_runner.py`)
	- Model execution utilities (`model_executor.py`, `model_utils.py`)
	- Data prep and analytics (`data_preprocessing.py`, `risk_analytics.py`, `reporting.py`)
- `cpp/`
	- High-performance concurrent inference/backtesting executable
	- Core source in `cpp/src/`
	- Build configuration in `cpp/Makefile`

## End-to-end workflow

1. Train models in notebooks under `training/`.
2. Export weights and metadata for runtime use with `training/export_models.py`.
3. Run backtests:
	 - Python for fast iteration and analysis
	 - C++ for lower-latency, optimized execution
4. Compare standalone model performance and blended portfolio mixes.

## Running the project

### 1) Export model artifacts

From `cpp/`:

```bash
make export_models
```

### 2) Run Python concurrent backtest

From `python/`:

```bash
python3 backtest_runner.py
```

This path loads `.keras` and `.joblib` models and prints:

- Sample-by-sample prediction comparison
- Inference latency stats
- Aggregated risk/diagnostic summary

### 3) Build and run C++ engine

From `cpp/`:

```bash
make
./model_run --help
```

Example run:

```bash
./model_run \
	--data /path/to/aaplIntra.csv \
	--linear-model ../training/models/elasticNet.json \
	--lstm-model ../training/models/lstm.json \
	--samples 100 \
	--subsample 4
```

The C++ runner includes:

- Parallel model inference
- 100-way blended portfolio evaluation (linear/lstm weight mixes)
- Ranked diagnostics for top blends

## Performance notes

Current optimization work (see `optimization.md`) includes:

- Compiler tuning (`-O3 -march=native -flto`, etc.)
- LSTM loop/memory access improvements
- Sequence subsampling via `--subsample`
- Vector pre-allocation and reduced allocation churn

## Data and config assumptions

- Current scripts include hard-coded default paths for data/model artifacts.
- Expected market data schema includes OHLCV-style fields and tick-related columns.
- LSTM preprocessing currently uses long lookback windows (sequence length 600).

If you want this to be more portable, the next improvement is moving all paths and runtime settings to CLI flags or a config file shared by both Python and C++ entry points.
