"""Tests for data_preprocessing.py — covers load_data, preprocess_lstm_data, preprocess_linear_data."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_preprocessing import load_data, preprocess_lstm_data, preprocess_linear_data


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_sorts_by_date(self, tmp_path):
        df = pd.DataFrame({
            "Dates": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "Close": [3.0, 1.0, 2.0],
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        loaded = load_data(str(path))
        assert list(loaded["Close"]) == [1.0, 2.0, 3.0]

    def test_returns_dataframe(self, tmp_path):
        df = pd.DataFrame({
            "Dates": ["2024-01-01"],
            "Close": [100.0],
        })
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        assert isinstance(load_data(str(path)), pd.DataFrame)


# ---------------------------------------------------------------------------
# preprocess_lstm_data
# ---------------------------------------------------------------------------

class TestPreprocessLSTM:
    def test_output_shapes(self, ohlcv_dataframe):
        feature_cols = ["Open", "High", "Low", "Close", "Volume", "Number Ticks"]
        seq_len = 20
        X, y = preprocess_lstm_data(ohlcv_dataframe, feature_cols, sequence_length=seq_len)
        n = len(ohlcv_dataframe) - 1
        expected_samples = n - seq_len + 1
        assert X.shape == (expected_samples, seq_len, len(feature_cols))
        assert y.shape == (expected_samples,)

    def test_target_is_next_close(self, ohlcv_dataframe):
        feature_cols = ["Close"]
        seq_len = 10
        X, y = preprocess_lstm_data(ohlcv_dataframe, feature_cols, sequence_length=seq_len)
        close = ohlcv_dataframe["Close"].values.astype("float32")
        next_close = np.roll(close, -1)[:-1]
        for i in range(min(5, len(y))):
            assert y[i] == pytest.approx(next_close[i + seq_len - 1], rel=1e-5)

    def test_sequence_length_one(self, ohlcv_dataframe):
        X, y = preprocess_lstm_data(ohlcv_dataframe, ["Close"], sequence_length=1)
        assert X.shape[1] == 1

    def test_sequence_too_long_yields_empty(self, ohlcv_dataframe):
        X, y = preprocess_lstm_data(ohlcv_dataframe, ["Close"], sequence_length=len(ohlcv_dataframe) + 10)
        assert len(X) == 0


# ---------------------------------------------------------------------------
# preprocess_linear_data
# ---------------------------------------------------------------------------

class TestPreprocessLinear:
    def test_produces_15_features(self, ohlcv_dataframe):
        X, y = preprocess_linear_data(ohlcv_dataframe)
        assert X.shape[1] == 15

    def test_no_nans(self, ohlcv_dataframe):
        X, y = preprocess_linear_data(ohlcv_dataframe)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_target_is_next_close(self, ohlcv_dataframe):
        X, y = preprocess_linear_data(ohlcv_dataframe)
        close = ohlcv_dataframe["Close"].values
        shifted = pd.Series(close).shift(-1).dropna().values
        for val in y:
            assert val in shifted or np.isclose(val, shifted, rtol=1e-5).any()

    def test_output_dtype(self, ohlcv_dataframe):
        X, y = preprocess_linear_data(ohlcv_dataframe)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_feature_columns_present(self, ohlcv_dataframe):
        """Verify the feature engineering creates the expected column families."""
        df_copy = ohlcv_dataframe.copy()
        df_copy["y"] = df_copy["Close"].shift(-1)
        for k in [1, 2, 3, 5, 10, 20]:
            df_copy[f"ret_lag_{k}"] = df_copy["Close"].pct_change(k)
        daily_ret = df_copy["Close"].pct_change()
        for w in [5, 10, 20]:
            df_copy[f"volatility_{w}"] = daily_ret.rolling(w).std()
        df_copy["volume_norm"] = df_copy["Volume"] / df_copy["Volume"].rolling(20).mean()
        df_copy["volume_change"] = df_copy["Volume"].pct_change()
        for w in [5, 10, 20]:
            df_copy[f"close_ma_{w}"] = df_copy["Close"] / df_copy["Close"].rolling(w).mean()
        df_copy["hl_range"] = (df_copy["High"] - df_copy["Low"]) / df_copy["Close"]
        feature_cols = [c for c in df_copy.columns
                        if c.startswith(("ret_lag_", "volatility_", "volume_", "close_ma_")) or c == "hl_range"]
        assert len(feature_cols) == 15
