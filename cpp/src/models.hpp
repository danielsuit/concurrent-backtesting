#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "json_parser.hpp"

namespace models {
// Standard scaler for feature normalization
struct StandardScaler {
    std::vector<double> mean;
    std::vector<double> scale;
    std::vector<double> transform(const std::vector<double>& features) const {
        if (features.size() != mean.size()) {
            throw std::runtime_error("Feature size mismatch in scaler");
        }
        std::vector<double> result(features.size());
        for (size_t i = 0; i < features.size(); i++) {
            result[i] = (features[i] - mean[i]) / scale[i];
        }
        return result;
    }
};
// Simple scaler for target variable
struct TargetScaler {
    double mean = 0;
    double scale = 1;
    bool enabled = false;
    
    double transform(double value) const {
        return enabled ? (value - mean) / scale : value;
    }
    double inverseTransform(double value) const {
        return enabled ? value * scale + mean : value;
    }
};
// Linear model (Ridge, Lasso, ElasticNet)
class LinearModel {
public:
    std::vector<double> coefficients;
    double intercept = 0;
    StandardScaler scaler;
    TargetScaler yScaler;
    bool hasScaler = false;
    bool hasYScaler = false;
    static LinearModel loadFromJson(const std::string& filepath) {
        LinearModel model;
        auto json = JsonParser::parseFile(filepath);
        // Load coefficients
        model.coefficients = json["coefficients"].toDoubleVector();
        model.intercept = json["intercept"].asNumber();
        // Load feature scaler if present
        if (json.hasKey("scaler")) {
            model.hasScaler = true;
            model.scaler.mean = json["scaler"]["mean"].toDoubleVector();
            model.scaler.scale = json["scaler"]["scale"].toDoubleVector();
        }
        // Load target scaler if present
        if (json.hasKey("y_scaler")) {
            model.hasYScaler = true;
            model.yScaler.enabled = true;
            model.yScaler.mean = json["y_scaler"]["mean"].asNumber();
            model.yScaler.scale = json["y_scaler"]["scale"].asNumber();
        }
        return model;
    }
    
    double predict(const std::vector<double>& features) const {
        std::vector<double> scaledFeatures = hasScaler ? scaler.transform(features) : features;
        if (scaledFeatures.size() != coefficients.size()) {
            throw std::runtime_error("Feature size mismatch: expected " + 
                std::to_string(coefficients.size()) + ", got " + 
                std::to_string(scaledFeatures.size()));
        }
        double result = intercept;
        for (size_t i = 0; i < coefficients.size(); i++) {
            result += coefficients[i] * scaledFeatures[i];
        }
        // Inverse transform if y was scaled during training
        return yScaler.inverseTransform(result);
    }
    
    // Batch prediction
    std::vector<double> predict(const std::vector<std::vector<double>>& features) const {
        std::vector<double> predictions;
        predictions.reserve(features.size());
        for (const auto& f : features) {
            predictions.push_back(predict(f));
        }
        return predictions;
    }
};

// Stacked LSTM model (handles multiple LSTM layers)
class LSTMModel {
public:
    struct LSTMLayer {
        std::vector<std::vector<double>> Wi, Wf, Wc, Wo;  // input weights
        std::vector<std::vector<double>> Ui, Uf, Uc, Uo;  // recurrent weights
        std::vector<double> bi, bf, bc, bo;                // biases
        int inputSize = 0;
        int hiddenSize = 0;
    };
    
    struct DenseLayer {
        std::vector<std::vector<double>> W;
        std::vector<double> b;
        int inputSize = 0;
        int outputSize = 0;
    };
    std::vector<LSTMLayer> lstmLayers;
    std::vector<DenseLayer> denseLayers;
    int sequenceLength = 0;
    int inputSize = 0;
    static LSTMModel loadFromJson(const std::string& filepath) {
        LSTMModel model;
        try {
            auto json = JsonParser::parseFile(filepath);
            auto& inputShape = json["input_shape"].asArray();
            model.sequenceLength = static_cast<int>(inputShape[0].asNumber());
            model.inputSize = static_cast<int>(inputShape[1].asNumber());
            // Collect layer names and sort them to process in order
            std::vector<std::pair<std::string, const JsonValue*>> sortedLayers;
            for (const auto& [name, val] : json["layers"].objectVal) {
                sortedLayers.emplace_back(name, &val);
            }
            std::sort(sortedLayers.begin(), sortedLayers.end());
            
            for (const auto& [layerName, layerPtr] : sortedLayers) {
                const auto& layerData = *layerPtr;
                if (!layerData.hasKey("type") || !layerData.hasKey("weights")) continue;
                std::string type = layerData["type"].asString();
                auto& weights = layerData["weights"].asArray();
                if (type == "LSTM" && weights.size() >= 3) {
                    LSTMLayer layer;
                    auto& kernel = weights[0].asArray();
                    auto& recurrent = weights[1].asArray();
                    auto& bias = weights[2].asArray();
                    layer.hiddenSize = static_cast<int>(bias.size()) / 4;
                    layer.inputSize = static_cast<int>(kernel.size());
                    if (layer.hiddenSize <= 0 || layer.inputSize <= 0) continue;
                    // Initialize and load weights
                    layer.Wi.resize(layer.inputSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Wf.resize(layer.inputSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Wc.resize(layer.inputSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Wo.resize(layer.inputSize, std::vector<double>(layer.hiddenSize, 0));
                    for (int i = 0; i < layer.inputSize && i < static_cast<int>(kernel.size()); i++) {
                        auto& row = kernel[i].asArray();
                        for (int h = 0; h < layer.hiddenSize; h++) {
                            if (static_cast<size_t>(h) < row.size()) layer.Wi[i][h] = row[h].asNumber();
                            if (static_cast<size_t>(layer.hiddenSize + h) < row.size()) layer.Wf[i][h] = row[layer.hiddenSize + h].asNumber();
                            if (static_cast<size_t>(2*layer.hiddenSize + h) < row.size()) layer.Wc[i][h] = row[2*layer.hiddenSize + h].asNumber();
                            if (static_cast<size_t>(3*layer.hiddenSize + h) < row.size()) layer.Wo[i][h] = row[3*layer.hiddenSize + h].asNumber();
                        }
                    }
                    layer.Ui.resize(layer.hiddenSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Uf.resize(layer.hiddenSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Uc.resize(layer.hiddenSize, std::vector<double>(layer.hiddenSize, 0));
                    layer.Uo.resize(layer.hiddenSize, std::vector<double>(layer.hiddenSize, 0));
                    for (int i = 0; i < layer.hiddenSize && i < static_cast<int>(recurrent.size()); i++) {
                        auto& row = recurrent[i].asArray();
                        for (int h = 0; h < layer.hiddenSize; h++) {
                            if (static_cast<size_t>(h) < row.size()) layer.Ui[i][h] = row[h].asNumber();
                            if (static_cast<size_t>(layer.hiddenSize + h) < row.size()) layer.Uf[i][h] = row[layer.hiddenSize + h].asNumber();
                            if (static_cast<size_t>(2*layer.hiddenSize + h) < row.size()) layer.Uc[i][h] = row[2*layer.hiddenSize + h].asNumber();
                            if (static_cast<size_t>(3*layer.hiddenSize + h) < row.size()) layer.Uo[i][h] = row[3*layer.hiddenSize + h].asNumber();
                        }
                    }
                    layer.bi.resize(layer.hiddenSize, 0);
                    layer.bf.resize(layer.hiddenSize, 0);
                    layer.bc.resize(layer.hiddenSize, 0);
                    layer.bo.resize(layer.hiddenSize, 0);
                    for (int h = 0; h < layer.hiddenSize; h++) {
                        if (static_cast<size_t>(h) < bias.size()) layer.bi[h] = bias[h].asNumber();
                        if (static_cast<size_t>(layer.hiddenSize + h) < bias.size()) layer.bf[h] = bias[layer.hiddenSize + h].asNumber();
                        if (static_cast<size_t>(2*layer.hiddenSize + h) < bias.size()) layer.bc[h] = bias[2*layer.hiddenSize + h].asNumber();
                        if (static_cast<size_t>(3*layer.hiddenSize + h) < bias.size()) layer.bo[h] = bias[3*layer.hiddenSize + h].asNumber();
                    }
                    model.lstmLayers.push_back(std::move(layer));
                } else if (type == "Dense" && weights.size() >= 2) {
                    DenseLayer layer;
                    auto& kernel = weights[0].asArray();
                    auto& bias = weights[1].asArray();
                    
                    layer.inputSize = static_cast<int>(kernel.size());
                    layer.outputSize = static_cast<int>(bias.size());
                    
                    layer.W.resize(layer.inputSize);
                    for (int i = 0; i < layer.inputSize; i++) {
                        auto& row = kernel[i].asArray();
                        layer.W[i].resize(layer.outputSize, 0);
                        for (int j = 0; j < layer.outputSize && j < static_cast<int>(row.size()); j++) {
                            layer.W[i][j] = row[j].asNumber();
                        }
                    }
                    
                    layer.b.resize(layer.outputSize);
                    for (int j = 0; j < layer.outputSize; j++) {
                        layer.b[j] = bias[j].asNumber();
                    }
                    
                    model.denseLayers.push_back(std::move(layer));
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load LSTM model: " + std::string(e.what()));
        }
        
        return model;
    }
    
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
    }
    
    static std::vector<double> processLSTMLayer(const LSTMLayer& layer, const std::vector<std::vector<double>>& sequence) {
        const int hiddenSize = layer.hiddenSize;
        const int inputSize = layer.inputSize;
        // Pre-allocate all vectors once
        std::vector<double> h(hiddenSize, 0.0);
        std::vector<double> c(hiddenSize, 0.0);
        std::vector<double> h_new(hiddenSize);
        std::vector<double> c_new(hiddenSize);
        std::vector<double> gates(4 * hiddenSize);  // i, f, c_tilde, o combined
        for (const auto& x : sequence) {
            const int inSize = std::min(inputSize, static_cast<int>(x.size()));
            // Initialize gates with biases
            for (int j = 0; j < hiddenSize; j++) {
                gates[j] = layer.bi[j];                      // i gate
                gates[hiddenSize + j] = layer.bf[j];         // f gate
                gates[2*hiddenSize + j] = layer.bc[j];       // c gate
                gates[3*hiddenSize + j] = layer.bo[j];       // o gate
            }
            // Input contributions (vectorized loop)
            for (int k = 0; k < inSize; k++) {
                const double xk = x[k];
                const double* wi = layer.Wi[k].data();
                const double* wf = layer.Wf[k].data();
                const double* wc = layer.Wc[k].data();
                const double* wo = layer.Wo[k].data();
                for (int j = 0; j < hiddenSize; j++) {
                    gates[j] += wi[j] * xk;
                    gates[hiddenSize + j] += wf[j] * xk;
                    gates[2*hiddenSize + j] += wc[j] * xk;
                    gates[3*hiddenSize + j] += wo[j] * xk;
                }
            }
            
            // Recurrent contributions
            for (int k = 0; k < hiddenSize; k++) {
                const double hk = h[k];
                const double* ui = layer.Ui[k].data();
                const double* uf = layer.Uf[k].data();
                const double* uc = layer.Uc[k].data();
                const double* uo = layer.Uo[k].data();
                for (int j = 0; j < hiddenSize; j++) {
                    gates[j] += ui[j] * hk;
                    gates[hiddenSize + j] += uf[j] * hk;
                    gates[2*hiddenSize + j] += uc[j] * hk;
                    gates[3*hiddenSize + j] += uo[j] * hk;
                }
            }
            // Apply activations and compute new states
            for (int j = 0; j < hiddenSize; j++) {
                const double i = sigmoid(gates[j]);
                const double f = sigmoid(gates[hiddenSize + j]);
                const double c_tilde = std::tanh(std::clamp(gates[2*hiddenSize + j], -10.0, 10.0));
                const double o = sigmoid(gates[3*hiddenSize + j]);
                c_new[j] = f * c[j] + i * c_tilde;
                h_new[j] = o * std::tanh(c_new[j]);
            }
            
            std::swap(h, h_new);
            std::swap(c, c_new);
        }
        
        return h;
    }
    
    static std::vector<double> processDenseLayer(const DenseLayer& layer,
                                                  const std::vector<double>& input) {
        std::vector<double> output(layer.outputSize);
        const int inSize = std::min(layer.inputSize, static_cast<int>(input.size()));
        
        for (int j = 0; j < layer.outputSize; j++) {
            double sum = layer.b[j];
            const double* w = layer.W[0].data() + j;  // First row, j-th element
            for (int i = 0; i < inSize; i++) {
                sum += layer.W[i][j] * input[i];
            }
            output[j] = sum;
        }
        
        return output;
    }
    
    // Subsample sequence to reduce computation (keep every Nth timestep)
    static std::vector<std::vector<double>> subsampleSequence(
            const std::vector<std::vector<double>>& sequence, int stride) {
        if (stride <= 1) return sequence;
        
        std::vector<std::vector<double>> subsampled;
        subsampled.reserve((sequence.size() + stride - 1) / stride);
        for (size_t i = 0; i < sequence.size(); i += stride) {
            subsampled.push_back(sequence[i]);
        }
        // Always include the last timestep
        if (!sequence.empty() && (sequence.size() - 1) % stride != 0) {
            subsampled.push_back(sequence.back());
        }
        return subsampled;
    }
    
    double predict(const std::vector<std::vector<double>>& sequence, int subsampleStride = 1) const {
        if (lstmLayers.empty() || sequence.empty()) {
            return 0.0;
        }
        
        // Subsample for faster inference (trade accuracy for speed)
        const auto& processSeq = (subsampleStride > 1) 
            ? subsampleSequence(sequence, subsampleStride) 
            : sequence;
        
        // Process first LSTM layer
        std::vector<double> hidden = processLSTMLayer(lstmLayers[0], processSeq);
        
        // Process remaining LSTM layers (stacked) - they take previous hidden as a sequence of 1
        for (size_t i = 1; i < lstmLayers.size(); i++) {
            std::vector<std::vector<double>> hiddenSeq = {hidden};
            hidden = processLSTMLayer(lstmLayers[i], hiddenSeq);
        }
        
        // Process Dense layers
        std::vector<double> output = hidden;
        for (const auto& dense : denseLayers) {
            output = processDenseLayer(dense, output);
        }
        
        return output.empty() ? 0.0 : output[0];
    }
    
    int getHiddenSize() const {
        return lstmLayers.empty() ? 0 : lstmLayers[0].hiddenSize;
    }
};
}