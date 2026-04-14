"""Tests for model_executor.py — ConcurrentMLStrategy thread pool execution."""
import time
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_executor import ConcurrentMLStrategy


class _DummyModel:
    """Minimal model stub that returns a predictable value."""
    def __init__(self, value, delay=0):
        self._value = value
        self._delay = delay

    def predict(self, X, verbose=0):
        if self._delay:
            time.sleep(self._delay)
        return np.array([[self._value]])


class TestConcurrentMLStrategy:
    def test_returns_both_predictions(self):
        m1 = _DummyModel(42.0)
        m2 = _DummyModel(99.0)
        strat = ConcurrentMLStrategy(m1, m2)
        result = strat.predict_concurrent(np.zeros((1, 5)), np.zeros((1, 3)))
        assert result["model1"] == pytest.approx(42.0)
        assert result["model2"] == pytest.approx(99.0)

    def test_latency_is_reported(self):
        strat = ConcurrentMLStrategy(_DummyModel(1.0), _DummyModel(2.0))
        result = strat.predict_concurrent(np.zeros((1, 5)), np.zeros((1, 3)))
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_concurrent_is_faster_than_sequential(self):
        """Two 50ms predictions run concurrently should take well under 100ms total."""
        m1 = _DummyModel(1.0, delay=0.05)
        m2 = _DummyModel(2.0, delay=0.05)
        strat = ConcurrentMLStrategy(m1, m2, n_workers=2)
        result = strat.predict_concurrent(np.zeros((1, 2)), np.zeros((1, 2)))
        assert result["latency_ms"] < 90

    def test_deterministic_across_calls(self):
        strat = ConcurrentMLStrategy(_DummyModel(3.14), _DummyModel(2.72))
        r1 = strat.predict_concurrent(np.zeros((1, 1)), np.zeros((1, 1)))
        r2 = strat.predict_concurrent(np.zeros((1, 1)), np.zeros((1, 1)))
        assert r1["model1"] == r2["model1"]
        assert r1["model2"] == r2["model2"]

    def test_shutdown_is_idempotent(self):
        strat = ConcurrentMLStrategy(_DummyModel(1.0), _DummyModel(2.0))
        del strat  # triggers __del__ / shutdown


tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed — skipping model_utils tests")


class TestSklearnModelWrapper:
    """Test the SklearnModelWrapper used by model_utils.load_model."""
    def test_predict_without_scalers(self):
        from model_utils import SklearnModelWrapper

        class _BareModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        w = SklearnModelWrapper(model=_BareModel())
        out = w.predict(np.array([[1, 2, 3]]))
        assert out.shape == (1, 1)
        assert out[0, 0] == pytest.approx(6.0)

    def test_predict_with_scaler(self):
        from model_utils import SklearnModelWrapper

        class _FakeScaler:
            def transform(self, X):
                return X * 2

        class _BareModel:
            def predict(self, X):
                return np.sum(X, axis=1)

        w = SklearnModelWrapper(model=_BareModel(), scaler=_FakeScaler())
        out = w.predict(np.array([[1, 2, 3]]))
        assert out[0, 0] == pytest.approx(12.0)

    def test_predict_with_y_scaler(self):
        from model_utils import SklearnModelWrapper

        class _FakeYScaler:
            def inverse_transform(self, y):
                return y + 100

        class _BareModel:
            def predict(self, X):
                return np.array([1.0])

        w = SklearnModelWrapper(model=_BareModel(), y_scaler=_FakeYScaler())
        out = w.predict(np.array([[0]]))
        assert out[0, 0] == pytest.approx(101.0)
