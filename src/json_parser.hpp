#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <unordered_map>

// Simple JSON parser for model weights (handles arrays and objects)
class JsonValue {
public:
    enum Type { Null, Number, String, Array, Object };
    
    Type type = Null;
    double numberVal = 0;
    std::string stringVal;
    std::vector<JsonValue> arrayVal;
    std::unordered_map<std::string, JsonValue> objectVal;
    
    double asNumber() const { return numberVal; }
    const std::string& asString() const { return stringVal; }
    const std::vector<JsonValue>& asArray() const { return arrayVal; }
    const JsonValue& operator[](const std::string& key) const { return objectVal.at(key); }
    const JsonValue& operator[](size_t idx) const { return arrayVal.at(idx); }
    bool hasKey(const std::string& key) const { return objectVal.find(key) != objectVal.end(); }
    size_t size() const { return type == Array ? arrayVal.size() : objectVal.size(); }
    
    std::vector<double> toDoubleVector() const {
        std::vector<double> result;
        for (const auto& v : arrayVal) {
            result.push_back(v.asNumber());
        }
        return result;
    }
};

class JsonParser {
    std::string text;
    size_t pos = 0;
    
    void skipWhitespace() {
        while (pos < text.size() && std::isspace(text[pos])) pos++;
    }
    
    JsonValue parseValue() {
        skipWhitespace();
        if (pos >= text.size()) return JsonValue{};
        
        char c = text[pos];
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == '-' || std::isdigit(c)) return parseNumber();
        if (text.substr(pos, 4) == "null") { pos += 4; return JsonValue{}; }
        if (text.substr(pos, 4) == "true") { pos += 4; JsonValue v; v.type = JsonValue::Number; v.numberVal = 1; return v; }
        if (text.substr(pos, 5) == "false") { pos += 5; JsonValue v; v.type = JsonValue::Number; v.numberVal = 0; return v; }
        
        throw std::runtime_error("Invalid JSON at position " + std::to_string(pos));
    }
    
    JsonValue parseObject() {
        JsonValue obj;
        obj.type = JsonValue::Object;
        pos++; // skip '{'
        skipWhitespace();
        
        while (pos < text.size() && text[pos] != '}') {
            skipWhitespace();
            if (text[pos] == '}') break;
            
            auto key = parseString();
            skipWhitespace();
            if (text[pos] != ':') throw std::runtime_error("Expected ':' in object");
            pos++;
            
            obj.objectVal[key.stringVal] = parseValue();
            
            skipWhitespace();
            if (text[pos] == ',') pos++;
        }
        pos++; // skip '}'
        return obj;
    }
    
    JsonValue parseArray() {
        JsonValue arr;
        arr.type = JsonValue::Array;
        pos++; // skip '['
        skipWhitespace();
        
        while (pos < text.size() && text[pos] != ']') {
            arr.arrayVal.push_back(parseValue());
            skipWhitespace();
            if (text[pos] == ',') pos++;
            skipWhitespace();
        }
        pos++; // skip ']'
        return arr;
    }
    
    JsonValue parseString() {
        JsonValue str;
        str.type = JsonValue::String;
        pos++; // skip opening '"'
        
        while (pos < text.size() && text[pos] != '"') {
            if (text[pos] == '\\' && pos + 1 < text.size()) {
                pos++;
                switch (text[pos]) {
                    case 'n': str.stringVal += '\n'; break;
                    case 't': str.stringVal += '\t'; break;
                    case 'r': str.stringVal += '\r'; break;
                    default: str.stringVal += text[pos]; break;
                }
            } else {
                str.stringVal += text[pos];
            }
            pos++;
        }
        pos++; // skip closing '"'
        return str;
    }
    
    JsonValue parseNumber() {
        JsonValue num;
        num.type = JsonValue::Number;
        size_t start = pos;
        
        if (text[pos] == '-') pos++;
        while (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '.' || 
               text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
            if ((text[pos] == '+' || text[pos] == '-') && pos > start && 
                text[pos-1] != 'e' && text[pos-1] != 'E') break;
            pos++;
        }
        
        num.numberVal = std::stod(text.substr(start, pos - start));
        return num;
    }
    
public:
    JsonValue parse(const std::string& jsonText) {
        text = jsonText;
        pos = 0;
        return parseValue();
    }
    
    static JsonValue parseFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        
        JsonParser parser;
        return parser.parse(buffer.str());
    }
};
