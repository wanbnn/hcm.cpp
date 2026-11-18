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

std::vector<std::string> array_values(const std::string& json, const std::string& key) {
    std::vector<std::string> out;
    const auto at = json.find("\"" + key + "\"");
    if (at == std::string::npos) return out;
    const auto open = json.find('[', at);
    const auto close = json.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return out;
    size_t i = open + 1;
    while (i < close) {
        const auto q1 = json.find('"', i);
        if (q1 == std::string::npos || q1 >= close) break;
        const auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 >= close) break;
        out.push_back(json.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
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
        beam_width = int_value(json, "beam_width", beam_width);
        const auto zero = json.find("\"allow_zero_anaphora\"");
        if (zero != std::string::npos) allow_zero_anaphora = json.find("true", zero) != std::string::npos;
    }
    contractions.clear();
    const auto morphology = read_file(model_dir + "/morphology_pt.json");
    for (const auto& item : array_values(morphology, "contractions")) {
        const auto eq = item.find('=');
        if (eq != std::string::npos) contractions[item.substr(0, eq)] = item.substr(eq + 1);
    }
    proper_nouns = array_values(morphology, "proper_nouns");
    entity_types.clear();
    const auto types = read_file(model_dir + "/entity_types.json");
    for (const auto& item : array_values(types, "types")) {
        const auto eq = item.find('=');
        if (eq != std::string::npos) entity_types[item.substr(0, eq)] = item.substr(eq + 1);
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

std::string DiscourseConfig::entity_type(const std::string& canonical) const {
    const auto it = entity_types.find(canonical);
    return it == entity_types.end() ? std::string{} : it->second;
}

std::string DiscourseConfig::contraction(const std::string& preposition, GramGender gender,
                                         GramNumber number) const {
    const char* article = gender == GramGender::FEM
        ? (number == GramNumber::PLUR ? "as" : "a")
        : (number == GramNumber::PLUR ? "os" : "o");
    const auto key = preposition + "|" + article;
    const auto it = contractions.find(key);
    return it == contractions.end() ? std::string{} : it->second;
}

std::string DiscourseConfig::display_text(std::string text) const {
    for (const auto& proper : proper_nouns) {
        if (proper.empty()) continue;
        std::string lower_text = text;
        for (auto& ch : lower_text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        const auto needle = proper;
        std::string needle_lower = needle;
        for (auto& ch : needle_lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        std::string replacement;
        std::istringstream words(needle);
        std::string word;
        while (words >> word) {
            if (!replacement.empty()) replacement += ' ';
            const auto lower_word = [&]() {
                std::string copy = word;
                for (auto& ch : copy) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                return copy;
            }();
            if (lower_word == "de" || lower_word == "do" || lower_word == "da" ||
                lower_word == "dos" || lower_word == "das" || lower_word == "e") {
                replacement += lower_word;
            } else {
                word[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(word[0])));
                replacement += word;
            }
        }
        size_t at = 0;
        while ((at = lower_text.find(needle_lower, at)) != std::string::npos) {
            text.replace(at, needle.size(), replacement);
            lower_text.replace(at, needle.size(), needle_lower);
            at += replacement.size();
        }
    }
    return text;
}

std::vector<const Construction*> DiscourseConfig::constructions_for(RelationType relation) const {
    std::vector<const Construction*> selected;
    for (const auto& construction : constructions)
        if (construction.relation == relation) selected.push_back(&construction);
    return selected;
}

} // namespace cognicore
