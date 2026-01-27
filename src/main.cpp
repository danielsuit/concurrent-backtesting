#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "json_parser.hpp"
#include "models.hpp"
#include "data_loader.hpp"
#include "risk_analytics.hpp"

using namespace models;
using namespace data;
using namespace analytics;

// =============================================================================
// CONCURRENT MODEL EXECUTION
// =============================================================================

class ConcurrentMLStrategy {
public:
    LinearModel linearModel;
    LSTMModel lstmModel;
    bool hasLSTM = false;
    
    ConcurrentMLStrategy(const std::string& linearModelPath, const std::string& lstmModelPath = "") {
        std::cout << "Loading linear model from: " << linearModelPath << "\n";
        linearModel = LinearModel::loadFromJson(linearModelPath);
        std::cout << "  Coefficients: " << linearModel.coefficients.size() << "\n";
        std::cout << "  Has scaler: " << (linearModel.hasScaler ? "yes" : "no") << "\n";
        std::cout << "  Has y_scaler: " << (linearModel.hasYScaler ? "yes" : "no") << "\n";
        
        if (!lstmModelPath.empty()) {
            try {
                std::cout << "Loading LSTM model from: " << lstmModelPath << "\n";
                std::cout << "  Parsing JSON file...\n";
                lstmModel = LSTMModel::loadFromJson(lstmModelPath);
                std::cout << "  LSTM Layers: " << lstmModel.lstmLayers.size() << "\n";
                std::cout << "  Dense Layers: " << lstmModel.denseLayers.size() << "\n";
                std::cout << "  Sequence length: " << lstmModel.sequenceLength << "\n";
                std::cout << "  Input size: " << lstmModel.inputSize << "\n";
                if (!lstmModel.lstmLayers.empty()) {
                    hasLSTM = true;
                    std::cout << "  LSTM loaded successfully\n";
                } else {
                    std::cerr << "  Warning: LSTM model has no layers, disabling\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not load LSTM model: " << e.what() << "\n";
                hasLSTM = false;
            }
        }
    }
    
    struct PredictionResult {
        double linearPred = 0;
        double lstmPred = 0;
        double latencyMs = 0;
        bool hasLSTM = false;
    };
    
    // Subsample stride: 1=full sequence, 4=25% of timesteps, 10=10% of timesteps
    int lstmSubsampleStride = 4;  // Process every 4th timestep for ~4x speedup
    
    void setSubsampleStride(int stride) { lstmSubsampleStride = stride; }
    
    PredictionResult predictConcurrent(const std::vector<double>& linearFeatures,
                                        const std::vector<std::vector<double>>& lstmSequence = {}) {
        auto start = std::chrono::high_resolution_clock::now();
        
        PredictionResult result;
        result.hasLSTM = hasLSTM && !lstmSequence.empty();
        
        if (result.hasLSTM) {
            // Run both models concurrently
            auto linearFuture = std::async(std::launch::async, [&]() {
                return linearModel.predict(linearFeatures);
            });
            
            auto lstmFuture = std::async(std::launch::async, [this, &lstmSequence]() {
                return lstmModel.predict(lstmSequence, lstmSubsampleStride);
            });
            
            result.linearPred = linearFuture.get();
            result.lstmPred = lstmFuture.get();
        } else {
            // Only linear model
            result.linearPred = linearModel.predict(linearFeatures);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
        
        return result;
    }
};

// =============================================================================
// MAIN PROGRAM
// =============================================================================

int main(int argc, char* argv[]) {
    std::cout << "=============================================================================\n";
    std::cout << "CONCURRENT ML BACKTESTING ENGINE (C++)\n";
    std::cout << "=============================================================================\n\n";
    
    // Configuration
    std::string dataPath = "../../../desktop/quant/hist/aaplIntra.csv";
    std::string linearModelPath = "models/elasticNet.json";
    std::string lstmModelPath = "models/lstm.json";
    int maxSamples = 100;
    int subsampleStride = 4;  // Default: process every 4th LSTM timestep
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            dataPath = argv[++i];
        } else if (arg == "--linear-model" && i + 1 < argc) {
            linearModelPath = argv[++i];
        } else if (arg == "--lstm-model" && i + 1 < argc) {
            lstmModelPath = argv[++i];
        } else if (arg == "--samples" && i + 1 < argc) {
            maxSamples = std::stoi(argv[++i]);
        } else if (arg == "--subsample" && i + 1 < argc) {
            subsampleStride = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "  --data PATH         Path to CSV data file\n";
            std::cout << "  --linear-model PATH Path to linear model JSON\n";
            std::cout << "  --lstm-model PATH   Path to LSTM model JSON\n";
            std::cout << "  --samples N         Number of samples to evaluate\n";
            std::cout << "  --subsample N       LSTM timestep stride (1=full, 4=fast, 10=fastest)\n";
            return 0;
        }
    }
    
    try {
        // Load data
        std::cout << "Loading data from: " << dataPath << "\n";
        auto df = DataFrame::fromCSV(dataPath);
        std::cout << "  Rows: " << df.nRows() << ", Columns: " << df.nCols() << "\n\n";
        
        // Prepare features
        std::cout << "Preparing features...\n";
        auto linearFeatures = LinearFeatures::fromDataFrame(df);
        std::cout << "  Linear features: " << linearFeatures.X.size() << " samples, " 
                  << (linearFeatures.X.empty() ? 0 : linearFeatures.X[0].size()) << " features\n";
        
        // Prepare LSTM features
        LSTMFeatures lstmFeatures;
        try {
            lstmFeatures = LSTMFeatures::fromDataFrame(df, 600);  // Match LSTM input shape
            std::cout << "  LSTM features: " << lstmFeatures.X.size() << " samples\n";
        } catch (const std::exception& e) {
            std::cerr << "  Warning: Could not prepare LSTM features: " << e.what() << "\n";
        }
        
        // Load models
        std::cout << "\nLoading models...\n";
        ConcurrentMLStrategy strategy(linearModelPath, lstmModelPath);
        strategy.setSubsampleStride(subsampleStride);
        std::cout << "  LSTM subsample stride: " << subsampleStride 
                  << " (processing 1/" << subsampleStride << " timesteps)\n";
        
        // Sample indices
        int maxIdx = static_cast<int>(std::min(linearFeatures.X.size(), 
                                                lstmFeatures.X.empty() ? linearFeatures.X.size() : lstmFeatures.X.size()));
        int numSamples = std::min(maxSamples, maxIdx);
        
        std::vector<int> sampleIndices(maxIdx);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        
        std::mt19937 rng(42);
        std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);
        sampleIndices.resize(numSamples);
        std::sort(sampleIndices.begin(), sampleIndices.end());
        
        // Run predictions
        std::cout << "\nRunning predictions on " << numSamples << " samples...\n\n";
        
        std::vector<double> linearPredictions;
        std::vector<double> lstmPredictions;
        std::vector<double> actuals;
        std::vector<double> prices;
        std::vector<double> latencies;
        
        for (int idx : sampleIndices) {
            auto& linearX = linearFeatures.X[idx];
            std::vector<std::vector<double>> lstmX;
            if (!lstmFeatures.X.empty() && idx < static_cast<int>(lstmFeatures.X.size())) {
                lstmX = lstmFeatures.X[idx];
            }
            
            auto result = strategy.predictConcurrent(linearX, lstmX);
            
            linearPredictions.push_back(result.linearPred);
            actuals.push_back(linearFeatures.y[idx]);
            prices.push_back(linearFeatures.prices[idx]);
            latencies.push_back(result.latencyMs);
            
            if (result.hasLSTM) {
                lstmPredictions.push_back(result.lstmPred);
            }
        }
        
        // Print results header
        std::cout << std::string(110, '-') << "\n";
        std::cout << "MODEL PREDICTIONS COMPARISON\n";
        std::cout << std::string(110, '-') << "\n";
        std::cout << std::left << std::setw(8) << "Sample"
                  << std::setw(14) << "Actual Price"
                  << std::setw(14) << "Linear Pred"
                  << std::setw(14) << "Linear Error";
        
        if (!lstmPredictions.empty()) {
            std::cout << std::setw(14) << "LSTM Pred"
                      << std::setw(14) << "LSTM Error";
        }
        std::cout << std::setw(12) << "Latency" << "\n";
        std::cout << std::string(110, '-') << "\n";
        
        // Print first 10 samples
        int displayCount = std::min(10, numSamples);
        for (int i = 0; i < displayCount; i++) {
            double linearError = linearPredictions[i] - actuals[i];
            
            std::cout << std::fixed << std::setprecision(4);
            std::cout << std::left << std::setw(8) << (i + 1)
                      << "$" << std::setw(13) << actuals[i]
                      << "$" << std::setw(13) << linearPredictions[i]
                      << std::setw(14) << linearError;
            
            if (!lstmPredictions.empty()) {
                double lstmError = lstmPredictions[i] - actuals[i];
                std::cout << "$" << std::setw(13) << lstmPredictions[i]
                          << std::setw(14) << lstmError;
            }
            std::cout << std::setw(10) << latencies[i] << "ms\n";
        }
        
        if (numSamples > displayCount) {
            std::cout << "... (" << (numSamples - displayCount) << " more samples)\n";
        }
        std::cout << std::string(110, '-') << "\n";
        
        // Compute and print diagnostics
        std::cout << "\n";
        
        auto linearDiag = RiskAnalytics::computeFullDiagnostics(
            "ElasticNet_Linear", linearPredictions, actuals, prices);
        
        std::cout << std::string(100, '=') << "\n";
        std::cout << "RESEARCH-GRADE STRATEGY DIAGNOSTICS\n";
        std::cout << std::string(100, '=') << "\n";
        
        RiskAnalytics::printDiagnostics(linearDiag);
        
        if (!lstmPredictions.empty()) {
            auto lstmDiag = RiskAnalytics::computeFullDiagnostics(
                "LSTM_Model", lstmPredictions, actuals, prices);
            RiskAnalytics::printDiagnostics(lstmDiag);
        }
        
        // Summary statistics
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "PERFORMANCE SUMMARY\n";
        std::cout << std::string(100, '=') << "\n";
        
        double avgLatency = RiskAnalytics::mean(latencies);
        double maxLatency = *std::max_element(latencies.begin(), latencies.end());
        double minLatency = *std::min_element(latencies.begin(), latencies.end());
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Linear Model MAE: $" << linearDiag.predAccuracy.mae << "\n";
        std::cout << "Linear Model RMSE: $" << linearDiag.predAccuracy.rmse << "\n";
        std::cout << "Linear Model R²: " << linearDiag.predAccuracy.rSquared << "\n";
        std::cout << "\nLatency Statistics:\n";
        std::cout << "  Average: " << avgLatency << " ms\n";
        std::cout << "  Min: " << minLatency << " ms\n";
        std::cout << "  Max: " << maxLatency << " ms\n";
        
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "Backtesting complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
