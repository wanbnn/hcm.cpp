#include "core/language_rules.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace cognicore {
namespace {

std::vector<std::string> json_string_array(const std::string& json, const std::string& key) {
    std::vector<std::string> values;
    const std::string quoted = "\"" + key + "\"";
    const size_t key_at = json.find(quoted);
    if (key_at == std::string::npos) return values;
    const size_t begin = json.find('[', key_at);
    const size_t end = json.find(']', begin);
    if (begin == std::string::npos || end == std::string::npos) return values;
    size_t at = begin + 1;
    while (at < end) {
        const size_t quote = json.find('"', at);
        if (quote == std::string::npos || quote >= end) break;
        std::string value;
        size_t cursor = quote + 1;
        while (cursor < end && json[cursor] != '"') {
            if (json[cursor] == '\\' && cursor + 1 < end) {
                const char escaped = json[cursor + 1];
                switch (escaped) {
                    case 'n': value.push_back('\n'); break;
                    case 't': value.push_back('\t'); break;
                    case 'r': value.push_back('\r'); break;
                    default: value.push_back(escaped); break;
                }
                cursor += 2;
            } else {
                value.push_back(json[cursor++]);
            }
        }
        values.push_back(std::move(value));
        at = cursor + 1;
    }
    return values;
}

std::string json_escape(const std::string& value) {
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

void write_array(std::ostringstream& out, const char* key, const std::vector<std::string>& values, bool last = false) {
    out << "  \"" << key << "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ", ";
        out << "\"" << json_escape(values[i]) << "\"";
    }
    out << "]" << (last ? "" : ",") << "\n";
}

} // namespace

bool LanguageRules::contains_any(const std::string& text, const std::vector<std::string>& cues) const {
    for (const auto& cue : cues) if (!cue.empty() && text.find(cue) != std::string::npos) return true;
    return false;
}

bool LanguageRules::starts_with_any(const std::string& text, const std::vector<std::string>& cues) const {
    for (const auto& cue : cues) if (!cue.empty() && text.starts_with(cue)) return true;
    return false;
}

bool LanguageRules::is_instruction_or_correction(const std::string& lower_text) const {
    // Instruction/correction cues are turn-level prefixes. A negation inside a
    // normal assertion ("O gelo não gera fogo") must still be learned as a
    // negated fact, not discarded as a correction turn.
    return starts_with_any(lower_text, instruction_prefixes) ||
           starts_with_any(lower_text, correction_markers);
}

std::string LanguageRules::verb_preposition(const std::string& lemma) const {
    for (const auto& item : verb_prepositions) {
        const auto eq = item.find('=');
        if (eq != std::string::npos && item.substr(0, eq) == lemma) return item.substr(eq + 1);
    }
    return {};
}

bool LanguageRules::is_code_followup(const std::string& lower_text) const {
    return contains_any(lower_text, followup_markers);
}

bool LanguageRules::is_hypothetical(const std::string& lower_text) const {
    return starts_with_any(lower_text, hypothetical_prefixes);
}

bool LanguageRules::is_leading_rule(const std::string& lower_text) const {
    return starts_with_any(lower_text, leading_rule_prefixes);
}

std::string LanguageRules::requested_language(const std::string& lower_text, const std::string& fallback) const {
    if (contains_any(lower_text, python_cues)) return "python";
    if (contains_any(lower_text, javascript_cues)) return "javascript";
    if (contains_any(lower_text, typescript_cues)) return "typescript";
    if (contains_any(lower_text, cpp_cues)) return "cpp";
    return fallback;
}

std::string LanguageRules::render_template(const std::vector<std::string>& templates,
                                           const std::string& lang, const std::string& seed,
                                           const std::vector<std::pair<std::string, std::string>>& slots) const {
    std::vector<std::string> selected;
    const auto prefix = lang + "|";
    for (const auto& tmpl : templates) if (tmpl.starts_with(prefix)) selected.push_back(tmpl.substr(prefix.size()));
    if (selected.empty()) {
        for (const auto& tmpl : templates) {
            const auto bar = tmpl.find('|');
            selected.push_back(bar == std::string::npos ? tmpl : tmpl.substr(bar + 1));
        }
    }
    if (selected.empty()) return {};
    (void)seed;
    std::string value = selected.front();
    for (const auto& [key, replacement] : slots) {
        const auto needle = "{" + key + "}";
        size_t at;
        while ((at = value.find(needle)) != std::string::npos) value.replace(at, needle.size(), replacement);
    }
    return value;
}

bool LanguageRules::load(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const auto json = buffer.str();
    language = json.find("\"language\": \"en\"") != std::string::npos ? "en" : "pt";
    instruction_prefixes = json_string_array(json, "instruction_prefixes");
    correction_markers = json_string_array(json, "correction_markers");
    question_prefixes = json_string_array(json, "question_prefixes");
    causal_cues = json_string_array(json, "causal_cues");
    code_cues = json_string_array(json, "code_cues");
    generation_cues = json_string_array(json, "generation_cues");
    code_generation_markers = json_string_array(json, "code_generation_markers");
    math_cues = json_string_array(json, "math_cues");
    resource_cues = json_string_array(json, "resource_cues");
    followup_markers = json_string_array(json, "followup_markers");
    hypothetical_prefixes = json_string_array(json, "hypothetical_prefixes");
    leading_rule_prefixes = json_string_array(json, "leading_rule_prefixes");
    consequence_markers = json_string_array(json, "consequence_markers");
    quantifier_words = json_string_array(json, "quantifier_words");
    python_cues = json_string_array(json, "python_cues");
    javascript_cues = json_string_array(json, "javascript_cues");
    typescript_cues = json_string_array(json, "typescript_cues");
    cpp_cues = json_string_array(json, "cpp_cues");
    math_add_cues = json_string_array(json, "math_add_cues");
    math_subtract_cues = json_string_array(json, "math_subtract_cues");
    math_multiply_cues = json_string_array(json, "math_multiply_cues");
    math_divide_cues = json_string_array(json, "math_divide_cues");
    section_introduction = json_string_array(json, "section_introduction");
    section_causes = json_string_array(json, "section_causes");
    section_consequences = json_string_array(json, "section_consequences");
    section_alternatives = json_string_array(json, "section_alternatives");
    section_conclusion = json_string_array(json, "section_conclusion");
    document_connectors = json_string_array(json, "document_connectors");
    part_of_templates = json_string_array(json, "part_of_templates");
    sequence_ordinals = json_string_array(json, "sequence_ordinals");
    document_title = json_string_array(json, "document_title");
    verb_prepositions = json_string_array(json, "verb_prepositions");
    verb_synonyms = json_string_array(json, "verb_synonyms");
    noun_synonyms = json_string_array(json, "noun_synonyms");
    relative_patterns = json_string_array(json, "relative_patterns");
    subordinate_patterns = json_string_array(json, "subordinate_patterns");
    aggregation_patterns = json_string_array(json, "aggregation_patterns");
    punctuation_sentence = json_string_array(json, "punctuation_sentence");
    punctuation_clause = json_string_array(json, "punctuation_clause");
    centering_pronouns = json_string_array(json, "centering_pronouns");
    list_sum_cues = json_string_array(json, "list_sum_cues");
    document_cues = json_string_array(json, "document_cues");
    document_topic_markers = json_string_array(json, "document_topic_markers");
    social_greet = json_string_array(json, "social_greet");
    social_thank = json_string_array(json, "social_thank");
    social_farewell = json_string_array(json, "social_farewell");
    social_introduce = json_string_array(json, "social_introduce");
    social_help = json_string_array(json, "social_help");
    social_smalltalk = json_string_array(json, "social_smalltalk");
    agree_templates = json_string_array(json, "agree_templates");
    clarify_templates = json_string_array(json, "clarify_templates");
    unknown_topic_templates = json_string_array(json, "unknown_topic_templates");
    empty_episode_templates = json_string_array(json, "empty_episode_templates");
    resource_unknown = json_string_array(json, "resource_unknown");
    resource_exceeds = json_string_array(json, "resource_exceeds");
    resource_within = json_string_array(json, "resource_within");
    conditional_conditional = json_string_array(json, "conditional_conditional");
    conditional_temporal = json_string_array(json, "conditional_temporal");
    conditional_causal = json_string_array(json, "conditional_causal");
    conditional_concessive = json_string_array(json, "conditional_concessive");
    conditional_consecutive = json_string_array(json, "conditional_consecutive");
    causal_explanation_templates = json_string_array(json, "causal_explanation_templates");
    math_value_templates = json_string_array(json, "math_value_templates");
    math_equation_templates = json_string_array(json, "math_equation_templates");
    document_empty_templates = json_string_array(json, "document_empty_templates");
    api_not_found_templates = json_string_array(json, "api_not_found_templates");
    connectors_result = json_string_array(json, "connectors_result");
    connectors_addition = json_string_array(json, "connectors_addition");
    connectors_contrast = json_string_array(json, "connectors_contrast");
    connectors_alternative = json_string_array(json, "connectors_alternative");
    connectors_consequence = json_string_array(json, "connectors_consequence");
    math_word_operators.clear();
    for (const auto& value : json_string_array(json, "math_word_operators")) {
        const auto separator = value.find('|');
        if (separator == std::string::npos) continue;
        math_word_operators.emplace_back(value.substr(0, separator), value.substr(separator + 1));
    }
    source_path = path;
    return true;
}

bool LanguageRules::save(const std::string& path) const {
    std::ostringstream out;
    out << "{\n";
    out << "  \"language\": \"" << json_escape(language) << "\",\n";
    write_array(out, "instruction_prefixes", instruction_prefixes);
    write_array(out, "correction_markers", correction_markers);
    write_array(out, "question_prefixes", question_prefixes);
    write_array(out, "causal_cues", causal_cues);
    write_array(out, "code_cues", code_cues);
    write_array(out, "generation_cues", generation_cues);
    write_array(out, "code_generation_markers", code_generation_markers);
    write_array(out, "math_cues", math_cues);
    write_array(out, "resource_cues", resource_cues);
    write_array(out, "followup_markers", followup_markers);
    write_array(out, "hypothetical_prefixes", hypothetical_prefixes);
    write_array(out, "leading_rule_prefixes", leading_rule_prefixes);
    write_array(out, "consequence_markers", consequence_markers);
    write_array(out, "quantifier_words", quantifier_words);
    write_array(out, "python_cues", python_cues);
    write_array(out, "javascript_cues", javascript_cues);
    write_array(out, "typescript_cues", typescript_cues);
    write_array(out, "cpp_cues", cpp_cues);
    std::vector<std::string> operators;
    for (const auto& [word, op] : math_word_operators) operators.push_back(word + "|" + op);
    write_array(out, "math_word_operators", operators);
    write_array(out, "math_add_cues", math_add_cues);
    write_array(out, "math_subtract_cues", math_subtract_cues);
    write_array(out, "math_multiply_cues", math_multiply_cues);
    write_array(out, "math_divide_cues", math_divide_cues);
    write_array(out, "section_introduction", section_introduction);
    write_array(out, "section_causes", section_causes);
    write_array(out, "section_consequences", section_consequences);
    write_array(out, "section_alternatives", section_alternatives);
    write_array(out, "section_conclusion", section_conclusion);
    write_array(out, "document_connectors", document_connectors);
    write_array(out, "part_of_templates", part_of_templates);
    write_array(out, "sequence_ordinals", sequence_ordinals);
    write_array(out, "document_title", document_title);
    write_array(out, "verb_prepositions", verb_prepositions);
    write_array(out, "verb_synonyms", verb_synonyms);
    write_array(out, "noun_synonyms", noun_synonyms);
    write_array(out, "relative_patterns", relative_patterns);
    write_array(out, "subordinate_patterns", subordinate_patterns);
    write_array(out, "aggregation_patterns", aggregation_patterns);
    write_array(out, "punctuation_sentence", punctuation_sentence);
    write_array(out, "punctuation_clause", punctuation_clause);
    write_array(out, "centering_pronouns", centering_pronouns);
    write_array(out, "list_sum_cues", list_sum_cues);
    write_array(out, "document_cues", document_cues);
    write_array(out, "document_topic_markers", document_topic_markers);
    write_array(out, "social_greet", social_greet);
    write_array(out, "social_thank", social_thank);
    write_array(out, "social_farewell", social_farewell);
    write_array(out, "social_introduce", social_introduce);
    write_array(out, "social_help", social_help);
    write_array(out, "social_smalltalk", social_smalltalk);
    write_array(out, "agree_templates", agree_templates);
    write_array(out, "clarify_templates", clarify_templates);
    write_array(out, "unknown_topic_templates", unknown_topic_templates);
    write_array(out, "empty_episode_templates", empty_episode_templates);
    write_array(out, "resource_unknown", resource_unknown);
    write_array(out, "resource_exceeds", resource_exceeds);
    write_array(out, "resource_within", resource_within);
    write_array(out, "conditional_conditional", conditional_conditional);
    write_array(out, "conditional_temporal", conditional_temporal);
    write_array(out, "conditional_causal", conditional_causal);
    write_array(out, "conditional_concessive", conditional_concessive);
    write_array(out, "conditional_consecutive", conditional_consecutive);
    write_array(out, "causal_explanation_templates", causal_explanation_templates);
    write_array(out, "math_value_templates", math_value_templates);
    write_array(out, "math_equation_templates", math_equation_templates);
    write_array(out, "document_empty_templates", document_empty_templates);
    write_array(out, "api_not_found_templates", api_not_found_templates);
    write_array(out, "connectors_result", connectors_result);
    write_array(out, "connectors_addition", connectors_addition);
    write_array(out, "connectors_contrast", connectors_contrast);
    write_array(out, "connectors_alternative", connectors_alternative);
    write_array(out, "connectors_consequence", connectors_consequence, true);
    out << "}\n";
    const auto temporary = path + ".tmp";
    std::ofstream file(temporary);
    if (!file) return false;
    file << out.str();
    file.close();
    if (!file) return false;
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    return !ec;
}

bool LanguageRules::load_default() {
#ifdef COGNICORE_DATA_DIR
    const std::string compiled = std::string(COGNICORE_DATA_DIR) + "/portuguese_model_rules.json";
    if (std::filesystem::exists(compiled) && load(compiled)) return true;
#endif
    for (const auto* candidate : {"data/portuguese_model_rules.json",
                                  "../data/portuguese_model_rules.json",
                                  "../../data/portuguese_model_rules.json"}) {
        if (std::filesystem::exists(candidate) && load(candidate)) return true;
    }
    return false;
}

} // namespace cognicore
