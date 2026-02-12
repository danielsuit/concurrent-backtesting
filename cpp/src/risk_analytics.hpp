#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>
#include <iomanip>
#include <iostream>

namespace analytics {
struct ReturnDistribution {
    double mean = 0;
    double std = 0;
    double skewness = 0;
    double excessKurtosis = 0;
    double min = 0;
    double max = 0;
    double median = 0;
    double positiveReturnsPct = 0;
    double negativeReturnsPct = 0;
};

struct DrawdownAnalysis {
    double maxDrawdown = 0;
    double maxDrawdownPct = 0;
    size_t maxDrawdownIdx = 0;
    size_t peakIdx = 0;
    size_t drawdownDuration = 0;
    size_t recoveryDuration = 0;
    double avgDrawdown = 0;
    double avgDrawdownPct = 0;
    double calmarRatio = 0;
    double currentDrawdown = 0;
    double currentDrawdownPct = 0;
};

struct PathDependentMetrics {
    double totalReturn = 0;
    double totalReturnPct = 0;
    double annualizedReturn = 0;
    double annualizedReturnPct = 0;
    double annualizedVolatility = 0;
    double annualizedVolatilityPct = 0;
    double sharpeRatio = 0;
    double sortinoRatio = 0;
    double winRate = 0;
    double winRatePct = 0;
    double avgWin = 0;
    double avgLoss = 0;
    double profitFactor = 0;
    double expectancy = 0;
    int nWins = 0;
    int nLosses = 0;
    double var95 = 0;
    double var99 = 0;
    double cvar95 = 0;
    double omegaRatio = 0;
};

struct PredictionAccuracy {
    double mae = 0;
    double mse = 0;
    double rmse = 0;
    double mape = 0;
    double maxError = 0;
    double minError = 0;
    double errorStd = 0;
    double bias = 0;
    double directionalAccuracy = 0;
    double directionalAccuracyPct = 0;
    double correlation = 0;
    double rSquared = 0;
};

struct StrategyDiagnostics {
    std::string name;
    size_t nSamples = 0;
    ReturnDistribution returnDist;
    DrawdownAnalysis drawdown;
    PathDependentMetrics pathMetrics;
    PredictionAccuracy predAccuracy;
};

class RiskAnalytics {
public:
    static constexpr double ANNUALIZATION_FACTOR = 252.0;
    static constexpr double RISK_FREE_RATE = 0.04;
    
    static double mean(const std::vector<double>& data) {
        if (data.empty()) return 0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }
    
    static double stddev(const std::vector<double>& data, int ddof = 1) {
        if (data.size() <= static_cast<size_t>(ddof)) return 0;
        double m = mean(data);
        double sum = 0;
        for (double v : data) sum += (v - m) * (v - m);
        return std::sqrt(sum / (data.size() - ddof));
    }
    
    static double percentile(std::vector<double> data, double p) {
        if (data.empty()) return 0;
        std::sort(data.begin(), data.end());
        double idx = (p / 100.0) * (data.size() - 1);
        size_t lower = static_cast<size_t>(idx);
        size_t upper = std::min(lower + 1, data.size() - 1);
        double frac = idx - lower;
        return data[lower] * (1 - frac) + data[upper] * frac;
    }
    
    static ReturnDistribution computeReturnDistribution(const std::vector<double>& returns) {
        ReturnDistribution dist;
        if (returns.empty()) return dist;
        
        size_t n = returns.size();
        dist.mean = mean(returns);
        dist.std = stddev(returns);
        
        // Higher-order moments
        if (dist.std > 0 && n > 2) {
            double sum3 = 0, sum4 = 0;
            for (double r : returns) {
                double z = (r - dist.mean) / dist.std;
                sum3 += z * z * z;
                sum4 += z * z * z * z;
            }
            dist.skewness = sum3 / n;
            dist.excessKurtosis = sum4 / n - 3.0;
        }
        dist.min = *std::min_element(returns.begin(), returns.end());
        dist.max = *std::max_element(returns.begin(), returns.end());
        dist.median = percentile(returns, 50);
        int positive = 0, negative = 0;
        for (double r : returns) {
            if (r > 0) positive++;
            else if (r < 0) negative++;
        }
        dist.positiveReturnsPct = 100.0 * positive / n;
        dist.negativeReturnsPct = 100.0 * negative / n;
        return dist;
    }
    static std::vector<double> computeEquityCurve(const std::vector<double>& returns) {
        std::vector<double> equity(returns.size() + 1, 1.0);
        for (size_t i = 0; i < returns.size(); i++) {
            equity[i + 1] = equity[i] * (1 + returns[i]);
        }
        return equity;
    }
    
    static DrawdownAnalysis computeDrawdowns(const std::vector<double>& equityCurve) {
        DrawdownAnalysis dd;
        if (equityCurve.empty()) return dd;
        size_t n = equityCurve.size();
        std::vector<double> runningMax(n);
        std::vector<double> drawdowns(n);
        runningMax[0] = equityCurve[0];
        for (size_t i = 1; i < n; i++) {
            runningMax[i] = std::max(runningMax[i-1], equityCurve[i]);
        }
        
        for (size_t i = 0; i < n; i++) {
            drawdowns[i] = (equityCurve[i] - runningMax[i]) / runningMax[i];
        }
        
        // Find max drawdown
        dd.maxDrawdownIdx = std::distance(drawdowns.begin(), 
            std::min_element(drawdowns.begin(), drawdowns.end()));
        dd.maxDrawdown = drawdowns[dd.maxDrawdownIdx];
        dd.maxDrawdownPct = dd.maxDrawdown * 100;
        
        // Find peak before max drawdown
        auto peakIt = std::max_element(equityCurve.begin(), 
            equityCurve.begin() + dd.maxDrawdownIdx + 1);
        dd.peakIdx = std::distance(equityCurve.begin(), peakIt);
        dd.drawdownDuration = dd.maxDrawdownIdx - dd.peakIdx;
        
        // Current drawdown
        dd.currentDrawdown = drawdowns.back();
        dd.currentDrawdownPct = dd.currentDrawdown * 100;
        
        // Average drawdown
        double sumNeg = 0;
        int countNeg = 0;
        for (double d : drawdowns) {
            if (d < 0) { sumNeg += d; countNeg++; }
        }
        dd.avgDrawdown = countNeg > 0 ? sumNeg / countNeg : 0;
        dd.avgDrawdownPct = dd.avgDrawdown * 100;
        
        // Calmar ratio
        double totalReturn = equityCurve.back() / equityCurve.front() - 1;
        double annReturn = std::pow(1 + totalReturn, ANNUALIZATION_FACTOR / n) - 1;
        dd.calmarRatio = dd.maxDrawdown != 0 ? annReturn / std::abs(dd.maxDrawdown) : 0;
        
        return dd;
    }
    
    static PathDependentMetrics computePathMetrics(const std::vector<double>& returns, const std::vector<double>& equityCurve) {
        PathDependentMetrics pm;
        if (returns.empty()) return pm;
        size_t n = returns.size();
        
        // Total and annualized return
        pm.totalReturn = equityCurve.back() / equityCurve.front() - 1;
        pm.totalReturnPct = pm.totalReturn * 100;
        pm.annualizedReturn = std::pow(1 + pm.totalReturn, ANNUALIZATION_FACTOR / n) - 1;
        pm.annualizedReturnPct = pm.annualizedReturn * 100;
        
        // Volatility
        pm.annualizedVolatility = stddev(returns) * std::sqrt(ANNUALIZATION_FACTOR);
        pm.annualizedVolatilityPct = pm.annualizedVolatility * 100;
        
        // Sharpe ratio
        double excessReturn = pm.annualizedReturn - RISK_FREE_RATE;
        pm.sharpeRatio = pm.annualizedVolatility > 0 ? excessReturn / pm.annualizedVolatility : 0;
        
        // Sortino ratio
        std::vector<double> negReturns;
        for (double r : returns) if (r < 0) negReturns.push_back(r);
        double downsideDev = stddev(negReturns) * std::sqrt(ANNUALIZATION_FACTOR);
        pm.sortinoRatio = downsideDev > 0 ? excessReturn / downsideDev : 0;
        
        // Win/loss analysis
        double sumWins = 0, sumLosses = 0;
        for (double r : returns) {
            if (r > 0) { pm.nWins++; sumWins += r; }
            else if (r < 0) { pm.nLosses++; sumLosses += r; }
        }
        
        pm.winRate = static_cast<double>(pm.nWins) / n;
        pm.winRatePct = pm.winRate * 100;
        pm.avgWin = pm.nWins > 0 ? sumWins / pm.nWins : 0;
        pm.avgLoss = pm.nLosses > 0 ? sumLosses / pm.nLosses : 0;
        pm.profitFactor = sumLosses != 0 ? sumWins / std::abs(sumLosses) : (sumWins > 0 ? std::numeric_limits<double>::infinity() : 0);
        pm.expectancy = pm.winRate * pm.avgWin + (1 - pm.winRate) * pm.avgLoss;
        
        // VaR and CVaR
        pm.var95 = percentile(returns, 5);
        pm.var99 = percentile(returns, 1);
        
        double sumCvar = 0;
        int countCvar = 0;
        for (double r : returns) {
            if (r <= pm.var95) { sumCvar += r; countCvar++; }
        }
        pm.cvar95 = countCvar > 0 ? sumCvar / countCvar : pm.var95;
        
        // Omega ratio
        pm.omegaRatio = std::abs(sumLosses) > 0 ? sumWins / std::abs(sumLosses) : 
                        std::numeric_limits<double>::infinity();
        
        return pm;
    }
    
    static PredictionAccuracy computePredictionAccuracy(const std::vector<double>& predictions, const std::vector<double>& actuals) {
        PredictionAccuracy pa;
        if (predictions.empty() || predictions.size() != actuals.size()) return pa;
        
        size_t n = predictions.size();
        std::vector<double> errors(n);
        std::vector<double> absErrors(n);
        
        double sumErrors = 0, sumAbsErrors = 0, sumSqErrors = 0, sumPctErrors = 0;
        
        for (size_t i = 0; i < n; i++) {
            errors[i] = predictions[i] - actuals[i];
            absErrors[i] = std::abs(errors[i]);
            sumErrors += errors[i];
            sumAbsErrors += absErrors[i];
            sumSqErrors += errors[i] * errors[i];
            if (actuals[i] != 0) sumPctErrors += std::abs(errors[i] / actuals[i]) * 100;
        }
        
        pa.mae = sumAbsErrors / n;
        pa.mse = sumSqErrors / n;
        pa.rmse = std::sqrt(pa.mse);
        pa.mape = sumPctErrors / n;
        pa.bias = sumErrors / n;
        
        pa.maxError = *std::max_element(absErrors.begin(), absErrors.end());
        pa.minError = *std::min_element(absErrors.begin(), absErrors.end());
        pa.errorStd = stddev(errors);
        
        // Directional accuracy
        int correctDir = 0;
        for (size_t i = 1; i < n; i++) {
            double predDir = predictions[i] - predictions[i-1];
            double actDir = actuals[i] - actuals[i-1];
            if ((predDir > 0 && actDir > 0) || (predDir < 0 && actDir < 0) || 
                (predDir == 0 && actDir == 0)) {
                correctDir++;
            }
        }
        pa.directionalAccuracy = n > 1 ? static_cast<double>(correctDir) / (n - 1) : 0;
        pa.directionalAccuracyPct = pa.directionalAccuracy * 100;
        
        // Correlation and R-squared
        double meanPred = mean(predictions);
        double meanAct = mean(actuals);
        double sumCov = 0, sumVarPred = 0, sumVarAct = 0, sumSqRes = 0, sumSqTot = 0;
        
        for (size_t i = 0; i < n; i++) {
            double dp = predictions[i] - meanPred;
            double da = actuals[i] - meanAct;
            sumCov += dp * da;
            sumVarPred += dp * dp;
            sumVarAct += da * da;
            sumSqRes += errors[i] * errors[i];
            sumSqTot += da * da;
        }
        
        double denom = std::sqrt(sumVarPred * sumVarAct);
        pa.correlation = denom > 0 ? sumCov / denom : 0;
        pa.rSquared = sumSqTot > 0 ? 1 - sumSqRes / sumSqTot : 0;
        
        return pa;
    }
    
    static StrategyDiagnostics computeFullDiagnostics(
        const std::string& name,
        const std::vector<double>& predictions,
        const std::vector<double>& actuals,
        const std::vector<double>& prices) {
        
        StrategyDiagnostics diag;
        diag.name = name;
        diag.nSamples = predictions.size();
        
        // Compute strategy returns based on predictions
        std::vector<double> returns;
        if (prices.size() > 1) {
            for (size_t i = 0; i < prices.size() - 1 && i < predictions.size() - 1; i++) {
                double actualReturn = (prices[i+1] - prices[i]) / prices[i];
                double signal = predictions[i] > prices[i] ? 1.0 : -1.0;
                returns.push_back(signal * actualReturn);
            }
        }
        
        auto equityCurve = computeEquityCurve(returns);
        
        diag.returnDist = computeReturnDistribution(returns);
        diag.drawdown = computeDrawdowns(equityCurve);
        diag.pathMetrics = computePathMetrics(returns, equityCurve);
        diag.predAccuracy = computePredictionAccuracy(predictions, actuals);
        
        return diag;
    }
    
    static void printDiagnostics(const StrategyDiagnostics& diag) {
        std::cout << std::fixed << std::setprecision(4);
        
        std::cout << "\n" << std::string(100, '-') << "\n";
        std::cout << "STRATEGY: " << diag.name << "\n";
        std::cout << std::string(100, '-') << "\n";
        
        std::cout << "\n  RETURN DISTRIBUTION:\n";
        std::cout << "     Mean: " << diag.returnDist.mean 
                  << "  |  Std: " << diag.returnDist.std
                  << "  |  Skew: " << diag.returnDist.skewness
                  << "  |  Kurt: " << diag.returnDist.excessKurtosis << "\n";
        
        std::cout << "\n  DRAWDOWN ANALYSIS:\n";
        std::cout << "     Max DD: " << diag.drawdown.maxDrawdownPct << "%"
                  << "  |  Avg DD: " << diag.drawdown.avgDrawdownPct << "%"
                  << "  |  Calmar: " << diag.drawdown.calmarRatio << "\n";
        
        std::cout << "\n  PATH-DEPENDENT METRICS:\n";
        std::cout << "     Total Return: " << diag.pathMetrics.totalReturnPct << "%"
                  << "  |  Sharpe: " << diag.pathMetrics.sharpeRatio
                  << "  |  Sortino: " << diag.pathMetrics.sortinoRatio << "\n";
        std::cout << "     Win Rate: " << diag.pathMetrics.winRatePct << "%"
                  << "  |  Profit Factor: " << diag.pathMetrics.profitFactor
                  << "  |  Omega: " << diag.pathMetrics.omegaRatio << "\n";
        
        std::cout << "\n  PREDICTION ACCURACY:\n";
        std::cout << "     MAE: $" << diag.predAccuracy.mae
                  << "  |  RMSE: $" << diag.predAccuracy.rmse
                  << "  |  R²: " << diag.predAccuracy.rSquared << "\n";
        std::cout << "     Directional Acc: " << diag.predAccuracy.directionalAccuracyPct << "%"
                  << "  |  Correlation: " << diag.predAccuracy.correlation << "\n";
    }
};

} // namespace analytics
