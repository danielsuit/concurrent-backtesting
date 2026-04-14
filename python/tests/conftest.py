"""Shared fixtures for the backtesting test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    """Seeded RNG for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_returns(rng):
    """500-point synthetic daily return series with realistic properties."""
    return rng.normal(0.0005, 0.015, size=500)


@pytest.fixture
def synthetic_equity(synthetic_returns):
    """Equity curve derived from synthetic_returns, starting at 1.0."""
    return np.cumprod(1 + synthetic_returns)


@pytest.fixture
def synthetic_predictions(rng):
    """Pair of (predictions, actuals) with known correlation."""
    actuals = np.cumsum(rng.normal(0, 1, size=200)) + 100
    noise = rng.normal(0, 0.5, size=200)
    predictions = actuals + noise
    return predictions, actuals


@pytest.fixture
def ohlcv_dataframe():
    """Minimal OHLCV DataFrame suitable for feature engineering tests."""
    n = 120
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 50000, size=n).astype(float)
    ticks = rng.integers(10, 500, size=n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame({
        "Dates": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Number Ticks": ticks,
    })
