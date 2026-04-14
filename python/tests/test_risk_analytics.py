"""Tests for risk_analytics.py — covers RiskAnalytics, StrategyMetrics, and ResultAggregator."""
import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_analytics import RiskAnalytics, StrategyMetrics, ResultAggregator


# ---------------------------------------------------------------------------
# StrategyMetrics
# ---------------------------------------------------------------------------

class TestStrategyMetrics:
    def test_equity_curve_is_cumprod(self):
        rets = np.array([0.01, -0.02, 0.03])
        m = StrategyMetrics(name="t", returns=rets,
                            predictions=np.zeros(3), actuals=np.zeros(3))
        expected = np.cumprod(1 + rets)
        np.testing.assert_allclose(m.equity_curve, expected)

    def test_cumulative_returns_matches_equity(self):
        rets = np.array([0.05, -0.03, 0.02])
        m = StrategyMetrics(name="t", returns=rets, predictions=np.zeros(3), actuals=np.zeros(3))
        np.testing.assert_allclose(m.cumulative_returns, m.equity_curve - 1)

    def test_n_samples(self):
        m = StrategyMetrics(name="t", returns=np.zeros(7), predictions=np.zeros(7), actuals=np.zeros(7))
        assert m.n_samples == 7

    def test_dtype_coercion(self):
        m = StrategyMetrics(name="t", returns=[1, 2, 3],
                            predictions=[4, 5, 6], actuals=[7, 8, 9])
        assert m.returns.dtype == np.float64


# ---------------------------------------------------------------------------
# RiskAnalytics.compute_return_distribution
# ---------------------------------------------------------------------------

class TestReturnDistribution:
    def test_empty_returns(self):
        assert RiskAnalytics.compute_return_distribution(np.array([])) == {}

    def test_single_return(self):
        dist = RiskAnalytics.compute_return_distribution(np.array([0.05]))
        assert dist["mean"] == pytest.approx(0.05)
        assert dist["std"] == 0.0

    def test_known_series(self):
        rets = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
        dist = RiskAnalytics.compute_return_distribution(rets)
        assert dist["mean"] == pytest.approx(np.mean(rets))
        assert dist["std"] == pytest.approx(np.std(rets, ddof=1))
        assert dist["min"] == pytest.approx(-0.02)
        assert dist["max"] == pytest.approx(0.02)
        assert 0 <= dist["positive_returns_pct"] <= 100

    def test_positive_negative_percentages_sum(self, synthetic_returns):
        dist = RiskAnalytics.compute_return_distribution(synthetic_returns)
        assert dist["positive_returns_pct"] + dist["negative_returns_pct"] <= 100.0 + 1e-9

    def test_percentiles_monotonic(self, synthetic_returns):
        dist = RiskAnalytics.compute_return_distribution(synthetic_returns)
        assert dist["percentile_1"] <= dist["percentile_5"]
        assert dist["percentile_5"] <= dist["percentile_25"]
        assert dist["percentile_25"] <= dist["percentile_75"]
        assert dist["percentile_75"] <= dist["percentile_95"]
        assert dist["percentile_95"] <= dist["percentile_99"]


# ---------------------------------------------------------------------------
# RiskAnalytics.compute_drawdowns
# ---------------------------------------------------------------------------

class TestDrawdowns:
    def test_empty_equity(self):
        assert RiskAnalytics.compute_drawdowns(np.array([])) == {}

    def test_monotonically_increasing_equity(self):
        equity = np.array([1.0, 1.1, 1.2, 1.3])
        dd = RiskAnalytics.compute_drawdowns(equity)
        assert dd["max_drawdown"] == pytest.approx(0.0)

    def test_known_drawdown(self):
        equity = np.array([1.0, 1.2, 0.9, 1.0])
        dd = RiskAnalytics.compute_drawdowns(equity)
        expected_dd = (0.9 - 1.2) / 1.2
        assert dd["max_drawdown"] == pytest.approx(expected_dd)
        assert dd["max_drawdown_pct"] == pytest.approx(expected_dd * 100)

    def test_drawdown_series_length(self, synthetic_equity):
        dd = RiskAnalytics.compute_drawdowns(synthetic_equity)
        assert len(dd["drawdown_series"]) == len(synthetic_equity)

    def test_drawdowns_are_nonpositive(self, synthetic_equity):
        dd = RiskAnalytics.compute_drawdowns(synthetic_equity)
        assert np.all(dd["drawdown_series"] <= 1e-15)

    def test_current_drawdown_is_last_value(self, synthetic_equity):
        dd = RiskAnalytics.compute_drawdowns(synthetic_equity)
        assert dd["current_drawdown"] == pytest.approx(dd["drawdown_series"][-1])


# ---------------------------------------------------------------------------
# RiskAnalytics.compute_path_dependent_metrics
# ---------------------------------------------------------------------------

class TestPathDependentMetrics:
    def test_empty_returns(self):
        assert RiskAnalytics.compute_path_dependent_metrics(np.array([]), np.array([])) == {}

    def test_all_positive_returns_win_rate(self):
        rets = np.array([0.01, 0.02, 0.03, 0.01])
        equity = np.cumprod(1 + rets)
        m = RiskAnalytics.compute_path_dependent_metrics(rets, equity)
        assert m["win_rate"] == pytest.approx(1.0)
        assert m["n_losses"] == 0

    def test_all_negative_returns(self):
        rets = np.array([-0.01, -0.02, -0.03])
        equity = np.cumprod(1 + rets)
        m = RiskAnalytics.compute_path_dependent_metrics(rets, equity)
        assert m["win_rate"] == pytest.approx(0.0)
        assert m["n_wins"] == 0

    def test_sharpe_ratio_sign(self, synthetic_returns, synthetic_equity):
        m = RiskAnalytics.compute_path_dependent_metrics(synthetic_returns, synthetic_equity)
        if m["annualized_return"] > RiskAnalytics.RISK_FREE_RATE:
            assert m["sharpe_ratio"] > 0
        elif m["annualized_return"] < RiskAnalytics.RISK_FREE_RATE:
            assert m["sharpe_ratio"] < 0

    def test_var_is_negative_tail(self, synthetic_returns, synthetic_equity):
        m = RiskAnalytics.compute_path_dependent_metrics(synthetic_returns, synthetic_equity)
        assert m["var_95"] <= 0 or np.mean(synthetic_returns) > 0.05
        assert m["var_99"] <= m["var_95"]

    def test_cvar_lte_var(self, synthetic_returns, synthetic_equity):
        m = RiskAnalytics.compute_path_dependent_metrics(synthetic_returns, synthetic_equity)
        assert m["cvar_95"] <= m["var_95"] + 1e-15

    def test_profit_factor_consistent_with_wins_losses(self):
        rets = np.array([0.02, -0.01, 0.03, -0.005])
        equity = np.cumprod(1 + rets)
        m = RiskAnalytics.compute_path_dependent_metrics(rets, equity)
        gross_profit = 0.02 + 0.03
        gross_loss = 0.01 + 0.005
        assert m["profit_factor"] == pytest.approx(gross_profit / gross_loss)

    def test_expectancy_formula(self):
        rets = np.array([0.02, -0.01, 0.03, -0.005])
        equity = np.cumprod(1 + rets)
        m = RiskAnalytics.compute_path_dependent_metrics(rets, equity)
        expected = m["win_rate"] * m["avg_win"] + (1 - m["win_rate"]) * m["avg_loss"]
        assert m["expectancy"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RiskAnalytics.compute_prediction_accuracy
# ---------------------------------------------------------------------------

class TestPredictionAccuracy:
    def test_empty(self):
        assert RiskAnalytics.compute_prediction_accuracy(np.array([]), np.array([])) == {}

    def test_perfect_predictions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = RiskAnalytics.compute_prediction_accuracy(a, a)
        assert m["mae"] == pytest.approx(0.0)
        assert m["rmse"] == pytest.approx(0.0)
        assert m["r_squared"] == pytest.approx(1.0)
        assert m["bias"] == pytest.approx(0.0)

    def test_constant_offset_bias(self):
        actuals = np.array([10.0, 20.0, 30.0, 40.0])
        preds = actuals + 2.0
        m = RiskAnalytics.compute_prediction_accuracy(preds, actuals)
        assert m["bias"] == pytest.approx(2.0)
        assert m["mae"] == pytest.approx(2.0)

    def test_directional_accuracy_perfect(self):
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = RiskAnalytics.compute_prediction_accuracy(preds, actuals)
        assert m["directional_accuracy"] == pytest.approx(1.0)

    def test_correlation_perfect(self):
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = 2.0 * actuals + 1.0
        m = RiskAnalytics.compute_prediction_accuracy(preds, actuals)
        assert m["correlation"] == pytest.approx(1.0, abs=1e-10)

    def test_r_squared_bounds(self, synthetic_predictions):
        preds, actuals = synthetic_predictions
        m = RiskAnalytics.compute_prediction_accuracy(preds, actuals)
        assert m["r_squared"] <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# ResultAggregator
# ---------------------------------------------------------------------------

class TestResultAggregator:
    def test_add_strategy_and_diagnostics(self, synthetic_predictions):
        preds, actuals = synthetic_predictions
        prices = actuals.copy()
        agg = ResultAggregator()
        agg.add_strategy_results("test_strat", preds, actuals, prices)
        diag = agg.compute_full_diagnostics("test_strat")
        assert diag["strategy_name"] == "test_strat"
        assert "return_distribution" in diag
        assert "drawdown_analysis" in diag
        assert "path_dependent_metrics" in diag
        assert "prediction_accuracy" in diag

    def test_unknown_strategy_raises(self):
        agg = ResultAggregator()
        with pytest.raises(ValueError, match="not found"):
            agg.compute_full_diagnostics("nonexistent")

    def test_strategy_returns_shape(self, synthetic_predictions):
        preds, actuals = synthetic_predictions
        prices = actuals.copy()
        agg = ResultAggregator()
        agg.add_strategy_results("s", preds, actuals, prices)
        assert len(agg.strategies["s"].returns) == len(prices) - 1

    def test_comparison_report_columns(self, synthetic_predictions):
        preds, actuals = synthetic_predictions
        prices = actuals.copy()
        agg = ResultAggregator()
        agg.add_strategy_results("a", preds, actuals, prices)
        agg.add_strategy_results("b", preds * 1.01, actuals, prices)
        df = agg.generate_comparison_report()
        assert len(df) == 2
        assert "Sharpe_Ratio" in df.columns
        assert "Max_Drawdown_%" in df.columns

    def test_fallback_returns_without_prices(self):
        preds = np.array([10.0, 11.0, 12.0])
        actuals = np.array([10.5, 11.5, 12.5])
        agg = ResultAggregator()
        agg.add_strategy_results("no_price", preds, actuals, prices=None)
        assert agg.strategies["no_price"].returns is not None
        assert len(agg.strategies["no_price"].returns) == 3

    def test_signal_construction_long_short(self):
        """When pred > price the signal should be +1 (long), when pred < price it should be -1 (short)."""
        prices = np.array([100.0, 102.0, 101.0])
        preds = np.array([105.0, 99.0, 103.0])
        actuals = np.array([102.0, 101.0, 104.0])
        agg = ResultAggregator()
        agg.add_strategy_results("sig", preds, actuals, prices)
        strat = agg.strategies["sig"]
        actual_returns = np.diff(prices) / prices[:-1]
        signals = np.sign(preds[:-1] - prices[:-1])
        expected = signals * actual_returns
        np.testing.assert_allclose(strat.returns, expected)
