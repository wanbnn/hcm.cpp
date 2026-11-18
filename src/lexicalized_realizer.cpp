#include "core/lexicalized_realizer.hpp"
#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <cctype>

namespace cognicore {
namespace {

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
    size_t at;
    while ((at = text.find(from)) != std::string::npos) text.replace(at, from.size(), to);
    return text;
}

std::string capitalize(std::string text) {
    if (!text.empty() && static_cast<unsigned char>(text[0]) < 128)
        text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
    return text;
}

std::string first_or(const std::vector<std::string>& values, const std::string& fallback) {
    return values.empty() ? fallback : values.front();
}

} // namespace

LexicalizedGrammarRealizer::LexicalizedGrammarRealizer(const LanguageRules& rules) : rules_(rules) {}

std::string LexicalizedGrammarRealizer::verb_phrase(const CausalStep& step, const std::string& lang) const {
    return MorphologicalRealizer{}.realize_clause(step, "", lang, "[PRED:VERB]");
}

std::string LexicalizedGrammarRealizer::subject_phrase(const CausalStep& step, const std::string& lang) const {
    const auto& name = step.surface_src.empty() ? step.src : step.surface_src;
    const auto lower = TextParser::utf8_tolower(name);
    for (const auto* article : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
        if (lower.starts_with(article)) return name;
    return MorphologicalRealizer::definite_article(step.src_gender, step.src_number, lang) + " " + name;
}

std::string LexicalizedGrammarRealizer::object_phrase(const CausalStep& step, const std::string& lang) const {
    const auto& name = step.surface_dst.empty() ? step.dst : step.surface_dst;
    if (step.type == RelationType::PROPERTY) return name;
    if (step.type == RelationType::TAXONOMIC && lang == "pt")
        return (step.dst_number == GramNumber::PLUR
            ? (step.dst_gender == GramGender::FEM ? "umas " : "uns ")
            : (step.dst_gender == GramGender::FEM ? "uma " : "um ")) + name;
    const auto lower = TextParser::utf8_tolower(name);
    for (const auto* article : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
        if (lower.starts_with(article)) return name;
    return MorphologicalRealizer::definite_article(step.dst_gender, step.dst_number, lang) + " " + name;
}

std::string LexicalizedGrammarRealizer::clause(const CausalStep& step, const std::string& subject,
                                                const std::string& lang) const {
    return MorphologicalRealizer{}.realize_clause(step, subject, lang);
}

std::string LexicalizedGrammarRealizer::fill(const std::string& pattern, const SemanticPlan& plan,
                                             const CausalStep& main_step, const CausalStep& dependent) const {
    const auto lang = plan.language.empty() ? std::string("pt") : plan.language;
    std::string output = pattern;
    output = replace_all(output, "{MAIN_SUBJ}", subject_phrase(main_step, lang));
    output = replace_all(output, "{MAIN_VERB}", verb_phrase(main_step, lang));
    output = replace_all(output, "{MAIN_OBJ}", object_phrase(main_step, lang));
    output = replace_all(output, "{SUBJ}", subject_phrase(main_step, lang));
    output = replace_all(output, "{VERB1}", verb_phrase(main_step, lang));
    output = replace_all(output, "{OBJ1}", object_phrase(main_step, lang));
    output = replace_all(output, "{VERB2}", verb_phrase(dependent, lang));
    output = replace_all(output, "{OBJ2}", object_phrase(dependent, lang));
    output = replace_all(output, "{REL_VERB}", verb_phrase(dependent, lang));
    output = replace_all(output, "{REL_OBJ}", object_phrase(dependent, lang));
    output = replace_all(output, "{SUB_SUBJ}", subject_phrase(dependent, lang));
    output = replace_all(output, "{SUB_VERB}", verb_phrase(dependent, lang));
    output = replace_all(output, "{SUB_OBJ}", object_phrase(dependent, lang));
    output = replace_all(output, "{ANTE}", subject_phrase(main_step, lang));
    const auto end = first_or(rules_.punctuation_sentence, ".");
    while (!output.empty() && output.back() == ' ') output.pop_back();
    if (output.empty() || (output.back() != '.' && output.back() != '!' && output.back() != '?'))
        output += end;
    return capitalize(output);
}

std::string LexicalizedGrammarRealizer::realize(const SemanticPlan& plan) const {
    MorphologicalRealizer morphology;
    if (plan.resource || plan.quoted_statement || !plan.conditional_conclusions.empty() ||
        plan.paths.empty() || plan.paths.front().steps.empty() ||
        plan.intent == DialogIntent::GREET || plan.intent == DialogIntent::AGREE ||
        plan.intent == DialogIntent::ASK_CLARIFY || plan.paths.front().steps.size() < 2)
        return morphology.realize(plan);

    const auto& steps = plan.paths.front().steps;
    const auto& first = steps[0];
    const auto& second = steps[1];
    const auto lang = plan.language.empty() ? std::string("pt") : plan.language;

    if (first.src == second.src && !rules_.aggregation_patterns.empty()) {
        auto text = fill(rules_.aggregation_patterns.front(), plan, first, second);
        // The shared subject is already present in [SUBJ]; do not repeat it.
        return text;
    }
    if (first.dst == second.src) {
        if (plan.tone == "warning" && !rules_.subordinate_patterns.empty())
            return fill(rules_.subordinate_patterns.front(), plan, first, second);
        if (!rules_.relative_patterns.empty())
            return fill(rules_.relative_patterns.front(), plan, first, second);
    }
    // Generic fallback: two explicit sentences with a clause join.
    const auto join = first_or(rules_.punctuation_clause, ", ");
    auto text = clause(first, subject_phrase(first, lang), lang) + join +
                clause(second, subject_phrase(second, lang), lang) + first_or(rules_.punctuation_sentence, ".");
    return capitalize(text);
}

} // namespace cognicore
