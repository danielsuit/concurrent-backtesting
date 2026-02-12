"""Vectorized risk analytics and metrics computation"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StrategyMetrics:
    """Research-grade diagnostics for strategy variant comparison"""
    name: str
    returns: np.ndarray = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    actuals: np.ndarray = field(repr=False)
    
    # Core statistics (computed lazily)
    _stats_cache: Dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        self.returns = np.asarray(self.returns, dtype=np.float64)
        self.predictions = np.asarray(self.predictions, dtype=np.float64)
        self.actuals = np.asarray(self.actuals, dtype=np.float64)
    
    @property
    def n_samples(self) -> int:
        return len(self.returns)
    
    @property  
    def cumulative_returns(self) -> np.ndarray:
        """Vectorized cumulative return series"""
        return np.cumprod(1 + self.returns) - 1
    
    @property
    def equity_curve(self) -> np.ndarray:
        """Equity curve starting at 1.0"""
        return np.cumprod(1 + self.returns)


class RiskAnalytics:
    """Vectorized risk analytics engine for research-grade diagnostics"""
    
    ANNUALIZATION_FACTOR = 252  # trading days per year
    RISK_FREE_RATE = 0.04  # 4% annual risk-free rate
    
    @staticmethod
    def compute_return_distribution(returns: np.ndarray) -> Dict:
        """Vectorized return distribution statistics"""
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)
        
        if n == 0:
            return {}
        
        # Core moments (vectorized)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1) if n > 1 else 0.0
        
        # Higher-order moments
        if std > 0 and n > 2:
            skewness = np.mean(((returns - mean) / std) ** 3) * (n / ((n-1) * (n-2))) * n if n > 2 else 0.0
            excess_kurtosis = (np.mean(((returns - mean) / std) ** 4) - 3) if n > 3 else 0.0
        else:
            skewness = 0.0
            excess_kurtosis = 0.0
        
        # Percentiles (vectorized)
        percentiles = np.percentile(returns, [1, 5, 10, 25, 50, 75, 90, 95, 99]) if n > 0 else np.zeros(9)
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'excess_kurtosis': excess_kurtosis,
            'min': np.min(returns),
            'max': np.max(returns),
            'median': percentiles[4],
            'percentile_1': percentiles[0],
            'percentile_5': percentiles[1],
            'percentile_10': percentiles[2],
            'percentile_25': percentiles[3],
            'percentile_75': percentiles[5],
            'percentile_90': percentiles[6],
            'percentile_95': percentiles[7],
            'percentile_99': percentiles[8],
            'interquartile_range': percentiles[5] - percentiles[3],
            'positive_returns_pct': np.mean(returns > 0) * 100,
            'negative_returns_pct': np.mean(returns < 0) * 100,
        }
    
    @staticmethod
    def compute_drawdowns(equity_curve: np.ndarray) -> Dict:
        """Vectorized drawdown analysis"""
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        
        if len(equity_curve) == 0:
            return {}
        
        # Running maximum (vectorized via cummax)
        running_max = np.maximum.accumulate(equity_curve)
        
        # Drawdown series
        drawdowns = (equity_curve - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        
        # Peak index (before max drawdown)
        peak_idx = np.argmax(equity_curve[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        # Recovery analysis (find first point where equity >= peak after trough)
        if max_dd_idx < len(equity_curve) - 1:
            peak_value = equity_curve[peak_idx]
            recovery_mask = equity_curve[max_dd_idx:] >= peak_value
            if np.any(recovery_mask):
                recovery_idx = max_dd_idx + np.argmax(recovery_mask)
                recovery_duration = recovery_idx - max_dd_idx
            else:
                recovery_idx = None
                recovery_duration = len(equity_curve) - max_dd_idx  # still in drawdown
        else:
            recovery_idx = None
            recovery_duration = 0
        
        # Drawdown duration (time from peak to trough)
        drawdown_duration = max_dd_idx - peak_idx
        
        # Average drawdown
        avg_dd = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0.0
        
        # Calmar ratio (annualized return / max drawdown)
        total_return = equity_curve[-1] / equity_curve[0] - 1 if equity_curve[0] != 0 else 0
        n_periods = len(equity_curve)
        annualized_return = (1 + total_return) ** (RiskAnalytics.ANNUALIZATION_FACTOR / max(n_periods, 1)) - 1
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else np.inf
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_idx': max_dd_idx,
            'peak_idx': peak_idx,
            'drawdown_duration': drawdown_duration,
            'recovery_duration': recovery_duration,
            'avg_drawdown': avg_dd,
            'avg_drawdown_pct': avg_dd * 100,
            'calmar_ratio': calmar_ratio,
            'current_drawdown': drawdowns[-1],
            'current_drawdown_pct': drawdowns[-1] * 100,
            'drawdown_series': drawdowns,
        }
    
    @staticmethod
    def compute_path_dependent_metrics(returns: np.ndarray, equity_curve: np.ndarray) -> Dict:
        """Path-dependent performance metrics for rigorous strategy comparison"""
        returns = np.asarray(returns, dtype=np.float64)
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        n = len(returns)
        
        if n == 0:
            return {}
        
        # Annualized metrics
        total_return = equity_curve[-1] / equity_curve[0] - 1 if equity_curve[0] != 0 else 0
        annualized_return = (1 + total_return) ** (RiskAnalytics.ANNUALIZATION_FACTOR / max(n, 1)) - 1
        annualized_vol = np.std(returns, ddof=1) * np.sqrt(RiskAnalytics.ANNUALIZATION_FACTOR) if n > 1 else 0
        
        # Sharpe ratio (excess return / volatility)
        excess_return = annualized_return - RiskAnalytics.RISK_FREE_RATE
        sharpe_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0
        
        # Sortino ratio (excess return / downside deviation)
        negative_returns = returns[returns < 0]
        downside_dev = np.std(negative_returns, ddof=1) * np.sqrt(RiskAnalytics.ANNUALIZATION_FACTOR) if len(negative_returns) > 1 else 0
        sortino_ratio = excess_return / downside_dev if downside_dev > 0 else 0
        
        # Information ratio components
        tracking_error = annualized_vol  # simplified: tracking vs zero benchmark
        information_ratio = annualized_return / tracking_error if tracking_error > 0 else 0
        
        # Win/loss analysis (vectorized)
        wins = returns > 0
        losses = returns < 0
        n_wins = np.sum(wins)
        n_losses = np.sum(losses)
        
        win_rate = n_wins / n if n > 0 else 0
        avg_win = np.mean(returns[wins]) if n_wins > 0 else 0
        avg_loss = np.mean(returns[losses]) if n_losses > 0 else 0
        
        # Profit factor
        gross_profit = np.sum(returns[wins]) if n_wins > 0 else 0
        gross_loss = abs(np.sum(returns[losses])) if n_losses > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Expectancy (expected value per trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Consecutive wins/losses (vectorized)
        if n > 0:
            win_streaks = np.diff(np.where(np.concatenate(([wins[0]], wins[:-1] != wins[1:], [True])))[0])[::2]
            loss_streaks = np.diff(np.where(np.concatenate(([losses[0]], losses[:-1] != losses[1:], [True])))[0])[::2]
            max_consecutive_wins = np.max(win_streaks) if len(win_streaks) > 0 else 0
            max_consecutive_losses = np.max(loss_streaks) if len(loss_streaks) > 0 else 0
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        
        # Tail risk metrics
        var_95 = np.percentile(returns, 5) if n > 0 else 0  # 5% VaR
        var_99 = np.percentile(returns, 1) if n > 0 else 0  # 1% VaR
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95  # Expected shortfall
        
        # Omega ratio (probability-weighted gains vs losses at threshold 0)
        gains_above_threshold = returns[returns > 0]
        losses_below_threshold = returns[returns < 0]
        omega_ratio = np.sum(gains_above_threshold) / abs(np.sum(losses_below_threshold)) if np.any(losses_below_threshold) else np.inf
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'annualized_volatility': annualized_vol,
            'annualized_volatility_pct': annualized_vol * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'omega_ratio': omega_ratio,
        }
    
    @staticmethod
    def compute_prediction_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> Dict:
        """Prediction accuracy metrics for model comparison"""
        predictions = np.asarray(predictions, dtype=np.float64)
        actuals = np.asarray(actuals, dtype=np.float64)
        n = len(predictions)
        
        if n == 0:
            return {}
        
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / actuals) * 100 if np.all(actuals != 0) else np.zeros(n)
        
        # Directional accuracy (did we predict direction correctly?)
        pred_direction = np.sign(np.diff(predictions)) if n > 1 else np.array([])
        actual_direction = np.sign(np.diff(actuals)) if n > 1 else np.array([])
        directional_accuracy = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0
        
        return {
            'mae': np.mean(abs_errors),
            'mse': np.mean(errors ** 2),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mape': np.mean(pct_errors),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'error_std': np.std(errors, ddof=1) if n > 1 else 0,
            'bias': np.mean(errors),  # systematic over/under prediction
            'directional_accuracy': directional_accuracy,
            'directional_accuracy_pct': directional_accuracy * 100,
            'correlation': np.corrcoef(predictions, actuals)[0, 1] if n > 1 else 0,
            'r_squared': 1 - (np.sum(errors**2) / np.sum((actuals - np.mean(actuals))**2)) if np.var(actuals) > 0 else 0,
        }


class ResultAggregator:
    """Vectorized result aggregation for multi-strategy comparison"""
    def __init__(self):
        self.strategies: Dict[str, StrategyMetrics] = {}
        self.analytics = RiskAnalytics()
    
    def add_strategy_results(self, name: str, predictions: np.ndarray, actuals: np.ndarray, prices: Optional[np.ndarray] = None):
        """Add strategy results for aggregation"""
        predictions = np.asarray(predictions, dtype=np.float64)
        actuals = np.asarray(actuals, dtype=np.float64)
        # Compute simulated returns based on prediction accuracy
        # Strategy: go long if predicted price > current, else flat
        if prices is not None and len(prices) > 1:
            prices = np.asarray(prices, dtype=np.float64)
            actual_returns = np.diff(prices) / prices[:-1]
            # Align predictions with returns (prediction at t predicts price at t+1)
            pred_signals = np.sign(predictions[:-1] - prices[:-1])  # +1 long, -1 short, 0 flat
            strategy_returns = pred_signals * actual_returns
        else:
            # Fallback: use prediction error as proxy for performance
            strategy_returns = -np.abs(predictions - actuals) / np.maximum(actuals, 1e-8)
        
        self.strategies[name] = StrategyMetrics(
            name=name,
            returns=strategy_returns,
            predictions=predictions,
            actuals=actuals
        )
    
    def compute_full_diagnostics(self, strategy_name: str) -> Dict:
        """Compute comprehensive diagnostics for a single strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        metrics = self.strategies[strategy_name]
        
        return {
            'strategy_name': strategy_name,
            'n_samples': metrics.n_samples,
            'return_distribution': self.analytics.compute_return_distribution(metrics.returns),
            'drawdown_analysis': self.analytics.compute_drawdowns(metrics.equity_curve),
            'path_dependent_metrics': self.analytics.compute_path_dependent_metrics(
                metrics.returns, metrics.equity_curve
            ),
            'prediction_accuracy': self.analytics.compute_prediction_accuracy(
                metrics.predictions, metrics.actuals
            ),
        }
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate research-grade comparison report across all strategies"""
        report_data = []
        
        for name, metrics in self.strategies.items():
            diagnostics = self.compute_full_diagnostics(name)
            
            row = {
                'Strategy': name,
                'N_Samples': diagnostics['n_samples'],
                # Return distribution
                'Mean_Return': diagnostics['return_distribution'].get('mean', 0),
                'Std_Return': diagnostics['return_distribution'].get('std', 0),
                'Skewness': diagnostics['return_distribution'].get('skewness', 0),
                'Kurtosis': diagnostics['return_distribution'].get('excess_kurtosis', 0),
                # Risk metrics
                'Max_Drawdown_%': diagnostics['drawdown_analysis'].get('max_drawdown_pct', 0),
                'Avg_Drawdown_%': diagnostics['drawdown_analysis'].get('avg_drawdown_pct', 0),
                'Calmar_Ratio': diagnostics['drawdown_analysis'].get('calmar_ratio', 0),
                # Performance metrics
                'Total_Return_%': diagnostics['path_dependent_metrics'].get('total_return_pct', 0),
                'Sharpe_Ratio': diagnostics['path_dependent_metrics'].get('sharpe_ratio', 0),
                'Sortino_Ratio': diagnostics['path_dependent_metrics'].get('sortino_ratio', 0),
                'Win_Rate_%': diagnostics['path_dependent_metrics'].get('win_rate_pct', 0),
                'Profit_Factor': diagnostics['path_dependent_metrics'].get('profit_factor', 0),
                'Omega_Ratio': diagnostics['path_dependent_metrics'].get('omega_ratio', 0),
                # Tail risk
                'VaR_95': diagnostics['path_dependent_metrics'].get('var_95', 0),
                'CVaR_95': diagnostics['path_dependent_metrics'].get('cvar_95', 0),
                # Prediction accuracy
                'MAE': diagnostics['prediction_accuracy'].get('mae', 0),
                'RMSE': diagnostics['prediction_accuracy'].get('rmse', 0),
                'R_Squared': diagnostics['prediction_accuracy'].get('r_squared', 0),
                'Directional_Acc_%': diagnostics['prediction_accuracy'].get('directional_accuracy_pct', 0),
            }
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def print_diagnostics_summary(self):
        """Print formatted diagnostics summary for all strategies"""
        print("\n" + "=" * 140)
        print("RESEARCH-GRADE STRATEGY DIAGNOSTICS")
        print("=" * 140)
        
        for name in self.strategies:
            diag = self.compute_full_diagnostics(name)
            ret_dist = diag['return_distribution']
            dd = diag['drawdown_analysis']
            path = diag['path_dependent_metrics']
            pred = diag['prediction_accuracy']
            
            print(f"\n{'─' * 140}")
            print(f"STRATEGY: {name.upper()}")
            print(f"{'─' * 140}")
            
            print(f"\n  📊 RETURN DISTRIBUTION:")
            print(f"     Mean: {ret_dist.get('mean', 0):.6f}  |  Std: {ret_dist.get('std', 0):.6f}  |  "
                  f"Skew: {ret_dist.get('skewness', 0):.4f}  |  Kurt: {ret_dist.get('excess_kurtosis', 0):.4f}")
            print(f"     Min: {ret_dist.get('min', 0):.6f}  |  Max: {ret_dist.get('max', 0):.6f}  |  "
                  f"Positive%: {ret_dist.get('positive_returns_pct', 0):.1f}%")
            
            print(f"\n  📉 DRAWDOWN ANALYSIS:")
            print(f"     Max DD: {dd.get('max_drawdown_pct', 0):.2f}%  |  Avg DD: {dd.get('avg_drawdown_pct', 0):.2f}%  |  "
                  f"Current DD: {dd.get('current_drawdown_pct', 0):.2f}%")
            print(f"     Calmar Ratio: {dd.get('calmar_ratio', 0):.4f}  |  DD Duration: {dd.get('drawdown_duration', 0)} periods")
            
            print(f"\n  📈 PATH-DEPENDENT METRICS:")
            print(f"     Total Return: {path.get('total_return_pct', 0):.2f}%  |  "
                  f"Ann. Return: {path.get('annualized_return_pct', 0):.2f}%  |  Ann. Vol: {path.get('annualized_volatility_pct', 0):.2f}%")
            print(f"     Sharpe: {path.get('sharpe_ratio', 0):.4f}  |  Sortino: {path.get('sortino_ratio', 0):.4f}  |  "
                  f"Omega: {path.get('omega_ratio', 0):.4f}")
            print(f"     Win Rate: {path.get('win_rate_pct', 0):.1f}%  |  Profit Factor: {path.get('profit_factor', 0):.4f}  |  "
                  f"Expectancy: {path.get('expectancy', 0):.6f}")
            print(f"     VaR(95%): {path.get('var_95', 0):.6f}  |  CVaR(95%): {path.get('cvar_95', 0):.6f}")
            
            print(f"\n  🎯 PREDICTION ACCURACY:")
            print(f"     MAE: ${pred.get('mae', 0):.4f}  |  RMSE: ${pred.get('rmse', 0):.4f}  |  "
                  f"R²: {pred.get('r_squared', 0):.4f}")
            print(f"     Bias: {pred.get('bias', 0):.4f}  |  Directional Acc: {pred.get('directional_accuracy_pct', 0):.1f}%  |  "
                  f"Correlation: {pred.get('correlation', 0):.4f}")
        
        # Comparison table
        print(f"\n{'=' * 140}")
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 140)
        comparison_df = self.generate_comparison_report()
        print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
