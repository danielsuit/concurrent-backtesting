#include <iostream>
#include "src/json_parser.hpp"

int main() {
    try {
        std::cout << "Loading JSON...\n";
        auto json = JsonParser::parseFile("models/lstm.json");
        std::cout << "JSON loaded successfully\n";
        
        if (json.hasKey("layers")) {
            auto& layers = json["layers"].objectVal;
            for (const auto& [name, val] : layers) {
                std::cout << "Layer: " << name << "\n";
                if (val.hasKey("type")) {
                    std::cout << "  Type: " << val["type"].asString() << "\n";
                }
                if (val.hasKey("weights")) {
                    auto& weights = val["weights"].asArray();
                    std::cout << "  Weights arrays: " << weights.size() << "\n";
                    for (size_t i = 0; i < weights.size(); i++) {
                        auto& w = weights[i].asArray();
                        std::cout << "    Weight[" << i << "] size: " << w.size();
                        if (!w.empty() && w[0].type == JsonValue::Array) {
                            std::cout << " x " << w[0].asArray().size() << " (2D)";
                        }
                        std::cout << "\n";
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
