#include "core/discourse_config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cognicore {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

double number_value(const std::string& json, const std::string& key, double fallback) {
    const auto at = json.find("\"" + key + "\"");
    if (at == std::string::npos) return fallback;
    const auto colon = json.find(':', at);
    if (colon == std::string::npos) return fallback;
    return std::strtod(json.c_str() + colon + 1, nullptr);
}

int int_value(const std::string& json, const std::string& key, int fallback) {
    return static_cast<int>(number_value(json, key, fallback));
}

std::string string_value(const std::string& json, const std::string& key) {
    const auto at = json.find("\"" + key + "\"");
    if (at == std::string::npos) return {};
    const auto colon = json.find(':', at);
    if (colon == std::string::npos) return {};
    const auto first = json.find('"', colon + 1);
    if (first == std::string::npos) return {};
    const auto second = json.find('"', first + 1);
    if (second == std::string::npos) return {};
    return json.substr(first + 1, second - first - 1);
}

RelationType relation_from(const std::string& value) {
    if (value == "PART_OF") return RelationType::PART_OF;
    if (value == "FUNCTION") return RelationType::FUNCTION;
    if (value == "ENABLES") return RelationType::ENABLES;
    if (value == "CAUSAL") return RelationType::CAUSAL;
    if (value == "SEQUENCE") return RelationType::SEQUENCE;
    if (value == "TAXONOMIC") return RelationType::TAXONOMIC;
    return RelationType::HAS_PART;
}

} // namespace

bool DiscourseConfig::load(const std::string& model_dir) {
    const auto path = model_dir + "/discourse_pt.json";
    const auto json = read_file(path);
    if (!json.empty()) {
        max_pronoun_distance = int_value(json, "max_pronoun_distance", max_pronoun_distance);
        repetition_penalty = number_value(json, "repetition_penalty", repetition_penalty);
        pronoun_bonus = number_value(json, "pronoun_bonus", pronoun_bonus);
        length_penalty = number_value(json, "length_penalty", length_penalty);
        connector_penalty = number_value(json, "connector_penalty", connector_penalty);
        const auto zero = json.find("\"allow_zero_anaphora\"");
        if (zero != std::string::npos) allow_zero_anaphora = json.find("true", zero) != std::string::npos;
    }
    constructions.clear();
    std::ifstream bank(model_dir + "/constructions_pt.jsonl");
    std::string line;
    while (std::getline(bank, line)) {
        if (line.empty() || line.front() == '#') continue;
        Construction construction;
        construction.id = string_value(line, "id");
        construction.pattern = string_value(line, "pattern");
        construction.relation = relation_from(string_value(line, "relation"));
        construction.cost = number_value(line, "cost", 0.0);
        if (!construction.pattern.empty() && !construction.id.empty())
            constructions.push_back(std::move(construction));
    }
    return true;
}

std::vector<const Construction*> DiscourseConfig::constructions_for(RelationType relation) const {
    std::vector<const Construction*> selected;
    for (const auto& construction : constructions)
        if (construction.relation == relation) selected.push_back(&construction);
    return selected;
}

} // namespace cognicore
