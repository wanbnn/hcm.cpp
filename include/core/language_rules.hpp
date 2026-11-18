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
    std::vector<std::string> list_sum_cues;
    std::vector<std::string> document_cues;
    std::vector<std::string> document_topic_markers;
    std::vector<std::string> social_greet;
    std::vector<std::string> social_thank;
    std::vector<std::string> social_farewell;
    std::vector<std::string> social_introduce;
    std::vector<std::string> social_help;
    std::vector<std::string> social_smalltalk;
    std::vector<std::string> agree_templates;
    std::vector<std::string> clarify_templates;
    std::vector<std::string> unknown_topic_templates;
    std::vector<std::string> empty_episode_templates;
    std::vector<std::string> resource_unknown;
    std::vector<std::string> resource_exceeds;
    std::vector<std::string> resource_within;
    std::vector<std::string> conditional_conditional;
    std::vector<std::string> conditional_temporal;
    std::vector<std::string> conditional_causal;
    std::vector<std::string> conditional_concessive;
    std::vector<std::string> conditional_consecutive;
    std::vector<std::string> causal_explanation_templates;
    std::vector<std::string> math_value_templates;
    std::vector<std::string> math_equation_templates;
    std::vector<std::string> document_empty_templates;
    std::vector<std::string> api_not_found_templates;
    std::vector<std::string> connectors_result;
    std::vector<std::string> connectors_addition;
    std::vector<std::string> connectors_contrast;
    std::vector<std::string> connectors_alternative;
    std::vector<std::string> connectors_consequence;

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
    /// Selects a "<lang>|text" template and fills {slots}. Templates are model
    /// data; no response prose is hardcoded in the inference path.
    std::string render_template(const std::vector<std::string>& templates, const std::string& lang,
                                const std::string& seed,
                                const std::vector<std::pair<std::string, std::string>>& slots = {}) const;
};

} // namespace cognicore
