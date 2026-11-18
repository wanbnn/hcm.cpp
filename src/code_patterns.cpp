#include "core/code_patterns.hpp"
#include "core/text_parser.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cognicore {
namespace {

std::string extract_string(const std::string& line, const std::string& key) {
    const auto quoted = "\"" + key + "\"";
    const auto at = line.find(quoted);
    if (at == std::string::npos) return {};
    const auto colon = line.find(':', at + quoted.size());
    if (colon == std::string::npos) return {};
    const auto first = line.find('"', colon);
    if (first == std::string::npos) return {};
    std::string value;
    for (size_t i = first + 1; i < line.size(); ++i) {
        if (line[i] == '"') break;
        if (line[i] == '\\' && i + 1 < line.size()) {
            const char escaped = line[++i];
            switch (escaped) {
                case 'n': value.push_back('\n'); break;
                case 't': value.push_back('\t'); break;
                case 'r': value.push_back('\r'); break;
                default: value.push_back(escaped); break;
            }
        } else value.push_back(line[i]);
    }
    return value;
}

std::vector<std::string> extract_array(const std::string& line, const std::string& key) {
    std::vector<std::string> values;
    const auto quoted = "\"" + key + "\"";
    const auto at = line.find(quoted);
    if (at == std::string::npos) return values;
    const auto begin = line.find('[', at);
    const auto end = line.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) return values;
    size_t cursor = begin + 1;
    while (cursor < end) {
        const auto first = line.find('"', cursor);
        if (first == std::string::npos || first >= end) break;
        const auto last = line.find('"', first + 1);
        if (last == std::string::npos || last > end) break;
        values.push_back(TextParser::utf8_tolower(line.substr(first + 1, last - first - 1)));
        cursor = last + 1;
    }
    return values;
}

std::string escape_json(const std::string& value) {
    std::string out;
    for (char ch : value) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            default: out += ch; break;
        }
    }
    return out;
}

std::string array_json(const std::vector<std::string>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ",";
        out += "\"" + escape_json(values[i]) + "\"";
    }
    return out + "]";
}

bool has_tag(const std::string& text, const std::vector<std::string>& tags) {
    for (const auto& tag : tags) if (!tag.empty() && text.find(tag) != std::string::npos) return true;
    return false;
}

} // namespace

bool CodePatternLibrary::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;
    std::vector<CodeConstruction> patterns;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        CodeConstruction pattern;
        pattern.id = extract_string(line, "id");
        pattern.name = extract_string(line, "name");
        pattern.language = extract_string(line, "language");
        pattern.tags = extract_array(line, "tags");
        pattern.required_tags = extract_array(line, "requires");
        pattern.template_code = extract_string(line, "template_code");
        if (pattern.id.empty() || pattern.template_code.empty()) continue;
        patterns.push_back(std::move(pattern));
    }
    patterns_ = std::move(patterns);
    return true;
}

bool CodePatternLibrary::save(const std::string& path) const {
    const auto temporary = path + ".tmp";
    std::ofstream out(temporary);
    if (!out) return false;
    for (const auto& pattern : patterns_) {
        out << "{\"id\":\"" << escape_json(pattern.id) << "\",\"name\":\"" << escape_json(pattern.name)
            << "\",\"language\":\"" << escape_json(pattern.language) << "\",\"tags\":"
            << array_json(pattern.tags) << ",\"requires\":" << array_json(pattern.required_tags)
            << ",\"template_code\":\"" << escape_json(pattern.template_code) << "\"}\n";
    }
    out.close();
    if (!out) return false;
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    return !ec;
}

bool CodePatternLibrary::load_default() {
#ifdef COGNICORE_DATA_DIR
    const std::string compiled = std::string(COGNICORE_DATA_DIR) + "/portuguese_code_patterns.jsonl";
    if (std::filesystem::exists(compiled) && load(compiled)) return true;
#endif
    for (const auto* candidate : {"data/portuguese_code_patterns.jsonl",
                                  "../data/portuguese_code_patterns.jsonl",
                                  "../../data/portuguese_code_patterns.jsonl"}) {
        if (std::filesystem::exists(candidate) && load(candidate)) return true;
    }
    return false;
}

const CodeConstruction* CodePatternLibrary::match(const std::string& lower_task,
                                                  const std::string& language) const {
    const CodeConstruction* best = nullptr;
    size_t best_score = 0;
    for (const auto& pattern : patterns_) {
        if (pattern.language != language) continue;
        if (!pattern.required_tags.empty() && !has_tag(lower_task, pattern.required_tags)) continue;
        size_t score = 0;
        for (const auto& tag : pattern.tags)
            if (lower_task.find(tag) != std::string::npos) ++score;
        if (!score) continue;
        if (score > best_score) { best_score = score; best = &pattern; }
    }
    return best;
}

} // namespace cognicore
