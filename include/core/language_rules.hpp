#pragma once

#include <string>
#include <utility>
#include <vector>

namespace cognicore {

/// Language-specific heuristics are model data, not inference code. The model
/// directory carries `language_rules.json`; training copies the selected rules
/// into the artifact. The code below only knows how to consume cue lists.
struct LanguageRules {
    std::string language = "pt";
    std::string source_path;

    std::vector<std::string> instruction_prefixes;
    std::vector<std::string> correction_markers;
    std::vector<std::string> question_prefixes;
    std::vector<std::string> causal_cues;
    std::vector<std::string> code_cues;
    std::vector<std::string> generation_cues;
    std::vector<std::string> code_generation_markers;
    std::vector<std::string> math_cues;
    std::vector<std::string> resource_cues;
    std::vector<std::string> followup_markers;
    std::vector<std::string> hypothetical_prefixes;
    std::vector<std::string> leading_rule_prefixes;
    std::vector<std::string> consequence_markers;
    std::vector<std::string> quantifier_words;
    std::vector<std::string> python_cues;
    std::vector<std::string> javascript_cues;
    std::vector<std::string> typescript_cues;
    std::vector<std::string> cpp_cues;
    std::vector<std::pair<std::string, std::string>> math_word_operators;
    std::vector<std::string> math_add_cues;
    std::vector<std::string> math_subtract_cues;
    std::vector<std::string> math_multiply_cues;
    std::vector<std::string> math_divide_cues;
    std::vector<std::string> section_introduction;
    std::vector<std::string> section_causes;
    std::vector<std::string> section_consequences;
    std::vector<std::string> section_alternatives;
    std::vector<std::string> section_conclusion;
    std::vector<std::string> document_connectors;
    std::vector<std::string> document_title;
    std::vector<std::string> verb_synonyms;
    std::vector<std::string> noun_synonyms;
    std::vector<std::string> relative_patterns;
    std::vector<std::string> subordinate_patterns;
    std::vector<std::string> aggregation_patterns;
    std::vector<std::string> punctuation_sentence;
    std::vector<std::string> punctuation_clause;
    std::vector<std::string> centering_pronouns;

    bool load(const std::string& path);
    bool save(const std::string& path) const;
    /// Searches data/portuguese_model_rules.json, ../data, ../../data and an
    /// optional compile-time COGNICORE_DATA_DIR. Never embeds word lists.
    bool load_default();
    bool empty() const { return instruction_prefixes.empty() && question_prefixes.empty(); }

    bool contains_any(const std::string& text, const std::vector<std::string>& cues) const;
    bool starts_with_any(const std::string& text, const std::vector<std::string>& cues) const;
    bool is_instruction_or_correction(const std::string& lower_text) const;
    bool is_code_followup(const std::string& lower_text) const;
    bool is_hypothetical(const std::string& lower_text) const;
    bool is_leading_rule(const std::string& lower_text) const;
    /// Returns "python", "javascript", "typescript", "cpp" or fallback.
    std::string requested_language(const std::string& lower_text, const std::string& fallback) const;
};

} // namespace cognicore
