#pragma once

#include "core/language_rules.hpp"
#include "core/realizer.hpp"

#include <string>

namespace cognicore {

/// Data-oriented lexicalized grammar realizer. It assembles clause trees with
/// relative, subordinate and aggregating constructions. Patterns and punctuation
/// come from the model's language_rules.json, never from hardcoded prose.
class LexicalizedGrammarRealizer : public ISurfaceRealizer {
public:
    explicit LexicalizedGrammarRealizer(const LanguageRules& rules = LanguageRules{});
    void set_rules(const LanguageRules& rules) { rules_ = rules; }
    std::string realize(const SemanticPlan& plan) const override;

private:
    std::string clause(const CausalStep& step, const std::string& subject, const std::string& lang) const;
    std::string verb_phrase(const CausalStep& step, const std::string& lang) const;
    std::string subject_phrase(const CausalStep& step, const std::string& lang) const;
    std::string object_phrase(const CausalStep& step, const std::string& lang) const;
    std::string fill(const std::string& pattern, const SemanticPlan& plan,
                     const CausalStep& main_step, const CausalStep& dependent) const;
    LanguageRules rules_;
};

} // namespace cognicore
