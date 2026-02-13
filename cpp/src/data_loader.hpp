#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace data {

struct DataFrame {
    std::vector<std::string> columns;
    std::unordered_map<std::string, size_t> columnIndex;
    std::vector<std::vector<double>> data;  // column-major storage
    
    size_t nRows() const { return data.empty() ? 0 : data[0].size(); }
    size_t nCols() const { return data.size(); }
    
    double get(size_t row, const std::string& col) const {
        return data[columnIndex.at(col)][row];
    }
    
    const std::vector<double>& getColumn(const std::string& col) const {
        return data[columnIndex.at(col)];
    }
    
    bool hasColumn(const std::string& col) const {
        return columnIndex.find(col) != columnIndex.end();
    }
    
    void addColumn(const std::string& name, const std::vector<double>& values) {
        columnIndex[name] = data.size();
        columns.push_back(name);
        data.push_back(values);
    }
    
    static DataFrame fromCSV(const std::string& filepath) {
        DataFrame df;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + filepath);
        }
        
        std::string line;
        
        // Parse header
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string col;
            while (std::getline(ss, col, ',')) {
                // Trim whitespace
                col.erase(0, col.find_first_not_of(" \t\r\n"));
                col.erase(col.find_last_not_of(" \t\r\n") + 1);
                df.columnIndex[col] = df.columns.size();
                df.columns.push_back(col);
            }
            df.data.resize(df.columns.size());
        }
        
        // Parse data rows
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string cell;
            size_t colIdx = 0;
            
            while (std::getline(ss, cell, ',') && colIdx < df.nCols()) {
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                
                try {
                    df.data[colIdx].push_back(std::stod(cell));
                } catch (...) {
                    // Handle non-numeric values (like dates) by using 0
                    df.data[colIdx].push_back(0);
                }
                colIdx++;
            }
            
            // Fill remaining columns with 0 if row is incomplete
            while (colIdx < df.nCols()) {
                df.data[colIdx].push_back(0);
                colIdx++;
            }
        }
        
        return df;
    }
};

// Feature engineering for linear models (matches ElasticNet training features)
struct LinearFeatures {
    std::vector<std::vector<double>> X;  // [n_samples, n_features]
    std::vector<double> y;                // target
    std::vector<double> prices;           // close prices for returns calculation
    
    static LinearFeatures fromDataFrame(const DataFrame& df) {
        LinearFeatures features;
        
        if (!df.hasColumn("Close")) {
            throw std::runtime_error("DataFrame must have 'Close' column");
        }
        
        const auto& close = df.getColumn("Close");
        const auto& high = df.hasColumn("High") ? df.getColumn("High") : close;
        const auto& low = df.hasColumn("Low") ? df.getColumn("Low") : close;
        const auto& volume = df.hasColumn("Volume") ? df.getColumn("Volume") : std::vector<double>(close.size(), 1.0);
        
        size_t n = close.size();
        
        // Lag returns: 1, 2, 3, 5, 10, 20
        std::vector<int> lags = {1, 2, 3, 5, 10, 20};
        std::vector<std::vector<double>> retLags(lags.size());
        for (size_t i = 0; i < lags.size(); i++) {
            retLags[i].resize(n, 0);
            int lag = lags[i];
            for (size_t j = lag; j < n; j++) {
                if (close[j - lag] != 0) {
                    retLags[i][j] = (close[j] - close[j - lag]) / close[j - lag];
                }
            }
        }
        
        // Volatility features: rolling std of returns at windows 5, 10, 20
        std::vector<int> volWindows = {5, 10, 20};
        std::vector<std::vector<double>> volatility(volWindows.size());
        std::vector<double> returns(n, 0);
        for (size_t i = 1; i < n; i++) {
            if (close[i - 1] != 0) {
                returns[i] = (close[i] - close[i - 1]) / close[i - 1];
            }
        }
        for (size_t w = 0; w < volWindows.size(); w++) {
            int win = volWindows[w];
            volatility[w].resize(n, 0);
            for (size_t i = win; i < n; i++) {
                double sum = 0, sum2 = 0;
                for (int j = 0; j < win; j++) {
                    double r = returns[i - j];
                    sum += r;
                    sum2 += r * r;
                }
                double mean = sum / win;
                double var = sum2 / win - mean * mean;
                volatility[w][i] = std::sqrt(std::max(0.0, var));
            }
        }
        
        // Volume features: volume_norm and volume_change
        std::vector<double> volumeNorm(n, 0);
        std::vector<double> volumeChange(n, 0);
        double volSum = 0;
        for (size_t i = 0; i < n; i++) {
            volSum += volume[i];
            if (i >= 20) volSum -= volume[i - 20];
            if (i >= 19) {
                double volMa = volSum / 20.0;
                if (volMa != 0) volumeNorm[i] = volume[i] / volMa;
            }
            if (i > 0 && volume[i - 1] != 0) {
                volumeChange[i] = (volume[i] - volume[i - 1]) / volume[i - 1];
            }
        }
        
        // Close relative to MAs: 5, 10, 20
        std::vector<int> maWindows = {5, 10, 20};
        std::vector<std::vector<double>> closeMa(maWindows.size());
        for (size_t w = 0; w < maWindows.size(); w++) {
            int win = maWindows[w];
            closeMa[w].resize(n, 0);
            double maSum = 0;
            for (size_t i = 0; i < n; i++) {
                maSum += close[i];
                if (i >= static_cast<size_t>(win)) maSum -= close[i - win];
                if (i >= static_cast<size_t>(win - 1)) {
                    double ma = maSum / win;
                    if (ma != 0) closeMa[w][i] = close[i] / ma;
                }
            }
        }
        
        // High-Low range
        std::vector<double> hlRange(n, 0);
        for (size_t i = 0; i < n; i++) {
            if (close[i] != 0) {
                hlRange[i] = (high[i] - low[i]) / close[i];
            }
        }
        
        // Create target (next close price)
        std::vector<double> target(n, 0);
        for (size_t i = 0; i < n - 1; i++) {
            target[i] = close[i + 1];
        }
        
        // Find first valid index (need enough history)
        size_t startIdx = 20;
        
        // Build feature matrix: 6 lag returns + 3 volatility + 2 volume + 3 close_ma + 1 hl_range = 15 features
        for (size_t i = startIdx; i < n - 1; i++) {
            std::vector<double> row;
            
            // Lag returns
            for (const auto& retLag : retLags) {
                row.push_back(retLag[i]);
            }
            
            // Volatility
            for (const auto& vol : volatility) {
                row.push_back(vol[i]);
            }
            
            // Volume features
            row.push_back(volumeNorm[i]);
            row.push_back(volumeChange[i]);
            
            // Close relative to MAs
            for (const auto& cma : closeMa) {
                row.push_back(cma[i]);
            }
            
            // High-Low range
            row.push_back(hlRange[i]);
            
            // Check for inf/nan
            bool valid = true;
            for (double v : row) {
                if (!std::isfinite(v)) { valid = false; break; }
            }
            
            if (valid) {
                features.X.push_back(row);
                features.y.push_back(target[i]);
                features.prices.push_back(close[i]);
            }
        }
        
        return features;
    }
};

// Feature engineering for LSTM models
struct LSTMFeatures {
    std::vector<std::vector<std::vector<double>>> X;  // [n_samples, seq_len, n_features]
    std::vector<double> y;
    std::vector<double> prices;
    
    static LSTMFeatures fromDataFrame(const DataFrame& df, int sequenceLength = 600, const std::vector<std::string>& featureCols = {"Open", "High", "Low", "Close", "Volume", "Number Ticks"}) {
        LSTMFeatures features;
        
        size_t n = df.nRows();
        
        // Get feature columns
        std::vector<const std::vector<double>*> cols;
        for (const auto& col : featureCols) {
            if (df.hasColumn(col)) {
                cols.push_back(&df.getColumn(col));
            }
        }
        
        if (cols.empty()) {
            throw std::runtime_error("No valid feature columns found");
        }
        
        const auto& close = df.getColumn("Close");
        
        // Create sequences
        for (size_t i = sequenceLength; i < n - 1; i++) {
            std::vector<std::vector<double>> sequence;
            
            for (size_t t = i - sequenceLength; t < i; t++) {
                std::vector<double> timestep;
                for (const auto* col : cols) {
                    timestep.push_back((*col)[t]);
                }
                sequence.push_back(timestep);
            }
            
            features.X.push_back(sequence);
            features.y.push_back(close[i + 1]);  // next close as target
            features.prices.push_back(close[i]);
        }
        
        return features;
    }
};

}
