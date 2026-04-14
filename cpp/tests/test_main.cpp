#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../src/json_parser.hpp"
#include "../src/models.hpp"
#include "../src/data_loader.hpp"
#include "../src/risk_analytics.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>

using namespace models;
using namespace data;
using namespace analytics;

// =========================================================================
// Helpers
// =========================================================================

static void write_tmp(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

// =========================================================================
// JsonParser
// =========================================================================

TEST_SUITE("JsonParser") {
    TEST_CASE("parse number") {
        JsonParser p;
        auto v = p.parse("42.5");
        CHECK(v.asNumber() == doctest::Approx(42.5));
    }

    TEST_CASE("parse negative number") {
        JsonParser p;
        auto v = p.parse("-3.14");
        CHECK(v.asNumber() == doctest::Approx(-3.14));
    }

    TEST_CASE("parse scientific notation") {
        JsonParser p;
        auto v = p.parse("1.5e-3");
        CHECK(v.asNumber() == doctest::Approx(0.0015));
    }

    TEST_CASE("parse string") {
        JsonParser p;
        auto v = p.parse("\"hello world\"");
        CHECK(v.asString() == "hello world");
    }

    TEST_CASE("parse array") {
        JsonParser p;
        auto v = p.parse("[1, 2, 3]");
        CHECK(v.size() == 3);
        CHECK(v[0].asNumber() == doctest::Approx(1));
        CHECK(v[2].asNumber() == doctest::Approx(3));
    }

    TEST_CASE("parse object") {
        JsonParser p;
        auto v = p.parse("{\"a\": 1, \"b\": \"two\"}");
        CHECK(v["a"].asNumber() == doctest::Approx(1));
        CHECK(v["b"].asString() == "two");
    }

    TEST_CASE("parse nested structure") {
        JsonParser p;
        auto v = p.parse("{\"arr\": [1, [2, 3]], \"obj\": {\"x\": 10}}");
        CHECK(v["arr"][0].asNumber() == doctest::Approx(1));
        CHECK(v["arr"][1][1].asNumber() == doctest::Approx(3));
        CHECK(v["obj"]["x"].asNumber() == doctest::Approx(10));
    }

    TEST_CASE("toDoubleVector") {
        JsonParser p;
        auto v = p.parse("[1.1, 2.2, 3.3]");
        auto vec = v.toDoubleVector();
        CHECK(vec.size() == 3);
        CHECK(vec[0] == doctest::Approx(1.1));
        CHECK(vec[2] == doctest::Approx(3.3));
    }

    TEST_CASE("hasKey") {
        JsonParser p;
        auto v = p.parse("{\"present\": 1}");
        CHECK(v.hasKey("present"));
        CHECK_FALSE(v.hasKey("missing"));
    }

    TEST_CASE("parse true/false/null") {
        JsonParser p;
        CHECK(p.parse("true").asNumber() == doctest::Approx(1));
        CHECK(p.parse("false").asNumber() == doctest::Approx(0));
        CHECK(p.parse("null").type == JsonValue::Null);
    }
}

// =========================================================================
// StandardScaler / TargetScaler
// =========================================================================

TEST_SUITE("Scalers") {
    TEST_CASE("StandardScaler transform") {
        StandardScaler s;
        s.mean = {10.0, 20.0};
        s.scale = {2.0, 5.0};
        auto result = s.transform({14.0, 30.0});
        CHECK(result[0] == doctest::Approx(2.0));
        CHECK(result[1] == doctest::Approx(2.0));
    }

    TEST_CASE("StandardScaler size mismatch throws") {
        StandardScaler s;
        s.mean = {1.0};
        s.scale = {1.0};
        CHECK_THROWS(s.transform({1.0, 2.0}));
    }

    TEST_CASE("TargetScaler round-trip") {
        TargetScaler ts;
        ts.mean = 50.0;
        ts.scale = 10.0;
        ts.enabled = true;
        double original = 42.0;
        double scaled = ts.transform(original);
        double recovered = ts.inverseTransform(scaled);
        CHECK(recovered == doctest::Approx(original));
    }

    TEST_CASE("TargetScaler disabled is identity") {
        TargetScaler ts;
        ts.enabled = false;
        CHECK(ts.transform(7.0) == doctest::Approx(7.0));
        CHECK(ts.inverseTransform(7.0) == doctest::Approx(7.0));
    }
}

// =========================================================================
// LinearModel
// =========================================================================

TEST_SUITE("LinearModel") {
    TEST_CASE("predict without scaler") {
        LinearModel m;
        m.coefficients = {2.0, -1.0, 0.5};
        m.intercept = 3.0;
        m.hasScaler = false;
        double pred = m.predict({1.0, 2.0, 4.0});
        // 3 + 2*1 + (-1)*2 + 0.5*4 = 3 + 2 - 2 + 2 = 5
        CHECK(pred == doctest::Approx(5.0));
    }

    TEST_CASE("predict with scaler") {
        LinearModel m;
        m.coefficients = {1.0};
        m.intercept = 0.0;
        m.hasScaler = true;
        m.scaler.mean = {10.0};
        m.scaler.scale = {2.0};
        double pred = m.predict({14.0});
        // scaled = (14-10)/2 = 2.0,  result = 0 + 1*2 = 2.0
        CHECK(pred == doctest::Approx(2.0));
    }

    TEST_CASE("predict with y_scaler inverse transform") {
        LinearModel m;
        m.coefficients = {1.0};
        m.intercept = 0.0;
        m.hasScaler = false;
        m.hasYScaler = true;
        m.yScaler.enabled = true;
        m.yScaler.mean = 100.0;
        m.yScaler.scale = 10.0;
        double pred = m.predict({0.5});
        // raw = 0 + 1*0.5 = 0.5,  inverse = 0.5*10 + 100 = 105
        CHECK(pred == doctest::Approx(105.0));
    }

    TEST_CASE("batch predict") {
        LinearModel m;
        m.coefficients = {1.0};
        m.intercept = 0.0;
        std::vector<std::vector<double>> batch = {{2.0}, {3.0}, {4.0}};
        auto preds = m.predict(batch);
        CHECK(preds.size() == 3);
        CHECK(preds[0] == doctest::Approx(2.0));
        CHECK(preds[2] == doctest::Approx(4.0));
    }

    TEST_CASE("size mismatch throws") {
        LinearModel m;
        m.coefficients = {1.0, 2.0};
        m.intercept = 0.0;
        CHECK_THROWS(m.predict({1.0}));
    }

    TEST_CASE("loadFromJson round-trip") {
        std::string path = "/tmp/test_linear_model.json";
        write_tmp(path, R"({
            "coefficients": [0.5, -0.3],
            "intercept": 1.0,
            "scaler": {"mean": [10, 20], "scale": [2, 5]},
            "y_scaler": {"mean": 100, "scale": 10}
        })");
        auto m = LinearModel::loadFromJson(path);
        CHECK(m.coefficients.size() == 2);
        CHECK(m.intercept == doctest::Approx(1.0));
        CHECK(m.hasScaler);
        CHECK(m.hasYScaler);
        CHECK(m.scaler.mean[0] == doctest::Approx(10));
        CHECK(m.yScaler.scale == doctest::Approx(10));
    }
}

// =========================================================================
// LSTMModel (unit-level)
// =========================================================================

TEST_SUITE("LSTMModel") {
    TEST_CASE("sigmoid") {
        CHECK(LSTMModel::sigmoid(0.0) == doctest::Approx(0.5));
        CHECK(LSTMModel::sigmoid(500.0) == doctest::Approx(1.0));
        CHECK(LSTMModel::sigmoid(-500.0) == doctest::Approx(0.0));
    }

    TEST_CASE("processLSTMStep updates hidden state") {
        LSTMModel::LSTMLayer layer;
        layer.inputSize = 2;
        layer.hiddenSize = 1;
        layer.Wi = {{0.1}, {0.2}};
        layer.Wf = {{0.1}, {0.2}};
        layer.Wc = {{0.1}, {0.2}};
        layer.Wo = {{0.1}, {0.2}};
        layer.Ui = {{0.1}};
        layer.Uf = {{0.1}};
        layer.Uc = {{0.1}};
        layer.Uo = {{0.1}};
        layer.bi = {0.0};
        layer.bf = {0.0};
        layer.bc = {0.0};
        layer.bo = {0.0};

        std::vector<double> h = {0.0};
        std::vector<double> c = {0.0};
        std::vector<double> gates(4);
        std::vector<double> x = {1.0, 1.0};

        LSTMModel::processLSTMStep(layer, x, h, c, gates);
        CHECK(h[0] != 0.0);
        CHECK(c[0] != 0.0);
    }

    TEST_CASE("subsampleSequence stride 1 is identity") {
        std::vector<std::vector<double>> seq = {{1}, {2}, {3}, {4}, {5}};
        auto sub = LSTMModel::subsampleSequence(seq, 1);
        CHECK(sub.size() == 5);
    }

    TEST_CASE("subsampleSequence stride 2 keeps every other + last") {
        std::vector<std::vector<double>> seq = {{1}, {2}, {3}, {4}, {5}};
        auto sub = LSTMModel::subsampleSequence(seq, 2);
        // indices 0, 2, 4 → size 3; last (4) is already included
        CHECK(sub.size() == 3);
        CHECK(sub[0][0] == doctest::Approx(1));
        CHECK(sub[1][0] == doctest::Approx(3));
        CHECK(sub.back()[0] == doctest::Approx(5));
    }

    TEST_CASE("subsampleSequence always includes last timestep") {
        std::vector<std::vector<double>> seq = {{1}, {2}, {3}, {4}, {5}, {6}};
        auto sub = LSTMModel::subsampleSequence(seq, 4);
        CHECK(sub.back()[0] == doctest::Approx(6));
    }

    TEST_CASE("processDenseLayer linear activation") {
        LSTMModel::DenseLayer d;
        d.inputSize = 2;
        d.outputSize = 1;
        d.W = {{3.0}, {-1.0}};
        d.b = {0.5};
        d.activation = "linear";
        auto out = LSTMModel::processDenseLayer(d, {2.0, 1.0});
        // 3*2 + (-1)*1 + 0.5 = 5.5
        CHECK(out[0] == doctest::Approx(5.5));
    }

    TEST_CASE("processDenseLayer relu activation") {
        LSTMModel::DenseLayer d;
        d.inputSize = 1;
        d.outputSize = 2;
        d.W = {{1.0, -1.0}};
        d.b = {0.0, 0.0};
        d.activation = "relu";
        auto out = LSTMModel::processDenseLayer(d, {5.0});
        CHECK(out[0] == doctest::Approx(5.0));
        CHECK(out[1] == doctest::Approx(0.0));
    }

    TEST_CASE("predict returns 0 for empty model") {
        LSTMModel m;
        CHECK(m.predict({}) == doctest::Approx(0.0));
    }
}

// =========================================================================
// DataFrame / CSV loading
// =========================================================================

TEST_SUITE("DataFrame") {
    TEST_CASE("fromCSV basic") {
        std::string path = "/tmp/test_data.csv";
        write_tmp(path, "A,B,C\n1.0,2.0,3.0\n4.0,5.0,6.0\n");
        auto df = DataFrame::fromCSV(path);
        CHECK(df.nRows() == 2);
        CHECK(df.nCols() == 3);
        CHECK(df.get(0, "A") == doctest::Approx(1.0));
        CHECK(df.get(1, "C") == doctest::Approx(6.0));
    }

    TEST_CASE("fromCSV handles non-numeric values gracefully") {
        std::string path = "/tmp/test_data_dates.csv";
        write_tmp(path, "Label,Value\nabc,100\ndef,200\n");
        auto df = DataFrame::fromCSV(path);
        CHECK(df.nRows() == 2);
        CHECK(df.get(0, "Label") == doctest::Approx(0.0));
        CHECK(df.get(0, "Value") == doctest::Approx(100.0));
    }

    TEST_CASE("fromCSV handles short rows by padding with zeros") {
        std::string path = "/tmp/test_data_short.csv";
        write_tmp(path, "A,B,C\n1.0,2.0\n4.0,5.0,6.0\n");
        auto df = DataFrame::fromCSV(path);
        CHECK(df.get(0, "C") == doctest::Approx(0.0));
    }

    TEST_CASE("fromCSV missing file throws") {
        CHECK_THROWS(DataFrame::fromCSV("/tmp/nonexistent_xyz.csv"));
    }

    TEST_CASE("hasColumn and getColumn") {
        std::string path = "/tmp/test_hascol.csv";
        write_tmp(path, "X,Y\n1,2\n3,4\n");
        auto df = DataFrame::fromCSV(path);
        CHECK(df.hasColumn("X"));
        CHECK_FALSE(df.hasColumn("Z"));
        CHECK(df.getColumn("Y")[0] == doctest::Approx(2.0));
    }

    TEST_CASE("addColumn") {
        DataFrame df;
        df.addColumn("test", {1.0, 2.0, 3.0});
        CHECK(df.hasColumn("test"));
        CHECK(df.nRows() == 3);
        CHECK(df.get(2, "test") == doctest::Approx(3.0));
    }
}

// =========================================================================
// LinearFeatures
// =========================================================================

TEST_SUITE("LinearFeatures") {
    TEST_CASE("fromDataFrame produces 15 features") {
        std::string path = "/tmp/test_features.csv";
        std::ofstream f(path);
        f << "Close,High,Low,Volume\n";
        for (int i = 0; i < 100; i++) {
            double price = 100.0 + i * 0.5;
            f << price << "," << price + 0.5 << "," << price - 0.5 << "," << (1000 + i) << "\n";
        }
        f.close();
        auto df = DataFrame::fromCSV(path);
        auto feats = LinearFeatures::fromDataFrame(df);
        REQUIRE(!feats.X.empty());
        CHECK(feats.X[0].size() == 15);
    }

    TEST_CASE("target is next close price") {
        std::string path = "/tmp/test_features_target.csv";
        std::ofstream f(path);
        f << "Close,High,Low,Volume\n";
        for (int i = 0; i < 50; i++) {
            double price = 100.0 + i;
            f << price << "," << price + 1 << "," << price - 1 << "," << 5000 << "\n";
        }
        f.close();
        auto df = DataFrame::fromCSV(path);
        auto feats = LinearFeatures::fromDataFrame(df);
        auto& close = df.getColumn("Close");
        for (size_t i = 0; i < std::min(feats.y.size(), (size_t)5); i++) {
            bool found = false;
            for (size_t j = 1; j < close.size(); j++) {
                if (std::abs(feats.y[i] - close[j]) < 1e-6) { found = true; break; }
            }
            CHECK(found);
        }
    }

    TEST_CASE("no inf/nan in features") {
        std::string path = "/tmp/test_features_clean.csv";
        std::ofstream f(path);
        f << "Close,High,Low,Volume\n";
        for (int i = 0; i < 80; i++) {
            double price = 50.0 + i * 0.3;
            f << price << "," << price + 0.2 << "," << price - 0.2 << "," << (2000 + i * 10) << "\n";
        }
        f.close();
        auto df = DataFrame::fromCSV(path);
        auto feats = LinearFeatures::fromDataFrame(df);
        for (const auto& row : feats.X) {
            for (double v : row) {
                CHECK(std::isfinite(v));
            }
        }
    }
}

// =========================================================================
// RiskAnalytics
// =========================================================================

TEST_SUITE("RiskAnalytics") {
    TEST_CASE("mean and stddev") {
        std::vector<double> data = {2.0, 4.0, 6.0, 8.0, 10.0};
        CHECK(RiskAnalytics::mean(data) == doctest::Approx(6.0));
        double expected_std = std::sqrt(10.0);  // sample std with ddof=1
        CHECK(RiskAnalytics::stddev(data) == doctest::Approx(expected_std));
    }

    TEST_CASE("mean of empty is 0") {
        CHECK(RiskAnalytics::mean({}) == doctest::Approx(0.0));
    }

    TEST_CASE("stddev of single element is 0") {
        CHECK(RiskAnalytics::stddev({5.0}) == doctest::Approx(0.0));
    }

    TEST_CASE("percentile") {
        std::vector<double> data = {10, 20, 30, 40, 50};
        CHECK(RiskAnalytics::percentile(data, 0) == doctest::Approx(10));
        CHECK(RiskAnalytics::percentile(data, 50) == doctest::Approx(30));
        CHECK(RiskAnalytics::percentile(data, 100) == doctest::Approx(50));
    }

    TEST_CASE("computeEquityCurve") {
        std::vector<double> rets = {0.1, -0.05, 0.02};
        auto eq = RiskAnalytics::computeEquityCurve(rets);
        CHECK(eq.size() == 4);
        CHECK(eq[0] == doctest::Approx(1.0));
        CHECK(eq[1] == doctest::Approx(1.1));
        CHECK(eq[2] == doctest::Approx(1.1 * 0.95));
        CHECK(eq[3] == doctest::Approx(1.1 * 0.95 * 1.02));
    }

    TEST_CASE("computeReturnDistribution empty") {
        auto dist = RiskAnalytics::computeReturnDistribution({});
        CHECK(dist.mean == doctest::Approx(0));
    }

    TEST_CASE("computeReturnDistribution known values") {
        std::vector<double> rets = {0.01, -0.01, 0.02, -0.02, 0.0};
        auto dist = RiskAnalytics::computeReturnDistribution(rets);
        CHECK(dist.mean == doctest::Approx(0.0));
        CHECK(dist.min == doctest::Approx(-0.02));
        CHECK(dist.max == doctest::Approx(0.02));
        CHECK(dist.positiveReturnsPct + dist.negativeReturnsPct <= 100.0 + 1e-9);
    }

    TEST_CASE("computeDrawdowns monotonically increasing") {
        auto eq = std::vector<double>{1.0, 1.1, 1.2, 1.3, 1.4};
        auto dd = RiskAnalytics::computeDrawdowns(eq);
        CHECK(dd.maxDrawdown == doctest::Approx(0.0));
    }

    TEST_CASE("computeDrawdowns known drawdown") {
        auto eq = std::vector<double>{1.0, 1.2, 0.9, 1.0};
        auto dd = RiskAnalytics::computeDrawdowns(eq);
        double expected = (0.9 - 1.2) / 1.2;
        CHECK(dd.maxDrawdown == doctest::Approx(expected));
        CHECK(dd.maxDrawdownPct == doctest::Approx(expected * 100));
    }

    TEST_CASE("Sharpe ratio sign consistency") {
        std::vector<double> rets(100);
        for (int i = 0; i < 100; i++) rets[i] = 0.01;
        auto eq = RiskAnalytics::computeEquityCurve(rets);
        auto pm = RiskAnalytics::computePathMetrics(rets, eq);
        CHECK(pm.sharpeRatio > 0);
    }

    TEST_CASE("VaR and CVaR ordering") {
        std::vector<double> rets = {-0.05, -0.03, -0.01, 0.01, 0.02, 0.04,
                                     -0.04, -0.02, 0.0, 0.03, 0.05, -0.06,
                                     0.01, -0.01, 0.02, -0.02, 0.03, -0.03,
                                     0.04, -0.04};
        auto eq = RiskAnalytics::computeEquityCurve(rets);
        auto pm = RiskAnalytics::computePathMetrics(rets, eq);
        CHECK(pm.var99 <= pm.var95);
        CHECK(pm.cvar95 <= pm.var95 + 1e-12);
    }

    TEST_CASE("PredictionAccuracy perfect predictions") {
        std::vector<double> vals = {1, 2, 3, 4, 5};
        auto pa = RiskAnalytics::computePredictionAccuracy(vals, vals);
        CHECK(pa.mae == doctest::Approx(0.0));
        CHECK(pa.rmse == doctest::Approx(0.0));
        CHECK(pa.rSquared == doctest::Approx(1.0));
        CHECK(pa.bias == doctest::Approx(0.0));
    }

    TEST_CASE("PredictionAccuracy constant offset") {
        std::vector<double> actuals = {10, 20, 30, 40};
        std::vector<double> preds = {12, 22, 32, 42};
        auto pa = RiskAnalytics::computePredictionAccuracy(preds, actuals);
        CHECK(pa.bias == doctest::Approx(2.0));
        CHECK(pa.mae == doctest::Approx(2.0));
    }

    TEST_CASE("PredictionAccuracy mismatched sizes returns default") {
        auto pa = RiskAnalytics::computePredictionAccuracy({1, 2}, {1, 2, 3});
        CHECK(pa.mae == doctest::Approx(0.0));
    }

    TEST_CASE("computeFullDiagnostics smoke test") {
        std::vector<double> preds = {100, 101, 102, 103, 104};
        std::vector<double> actuals = {100, 101, 102, 103, 104};
        std::vector<double> prices = {100, 101, 99, 103, 102};
        auto diag = RiskAnalytics::computeFullDiagnostics("test", preds, actuals, prices);
        CHECK(diag.name == "test");
        CHECK(diag.nSamples == 5);
    }
}
