#include "core/language_rules.hpp"
#include "core/lexicalized_realizer.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

using namespace cognicore;

namespace {
CausalStep make_step(const std::string& src, const std::string& rel, const std::string& dst,
                     RelationType type = RelationType::CAUSAL) {
    CausalStep step;
    step.src = src;
    step.rel = rel;
    step.dst = dst;
    step.type = type;
    step.confidence = 1.0f;
    step.tense = GramTense::PRES;
    step.src_gender = GramGender::MASC;
    step.src_number = GramNumber::SING;
    step.dst_gender = GramGender::MASC;
    step.dst_number = GramNumber::SING;
    return step;
}
}

int main() {
    LanguageRules rules;
    assert(rules.load_default());
    LexicalizedGrammarRealizer realizer(rules);

    SemanticPlan relative;
    relative.intent = DialogIntent::EXPLAIN_CAUSE;
    relative.language = "pt";
    relative.tone = "didactic";
    CausalPath relative_path;
    relative_path.steps = {make_step("atrito", "gerar", "calor"),
                           make_step("calor", "derreter", "gelo")};
    relative.paths.push_back(relative_path);
    const auto relative_text = realizer.realize(relative);
    assert(relative_text.find(", que ") != std::string::npos);
    assert(relative_text.find("calor") != std::string::npos);
    assert(relative_text.back() == '.');

    SemanticPlan aggregation;
    aggregation.intent = DialogIntent::EXPLAIN_CAUSE;
    aggregation.language = "pt";
    aggregation.tone = "casual";
    CausalPath aggregation_path;
    aggregation_path.steps = {make_step("atrito", "gerar", "calor"),
                              make_step("atrito", "aquecer", "gelo")};
    aggregation.paths.push_back(aggregation_path);
    const auto aggregation_text = realizer.realize(aggregation);
    assert(aggregation_text.find(" e ") != std::string::npos);
    assert(std::count(aggregation_text.begin(), aggregation_text.end(), 'a') > 0);
    const auto first_atrito = aggregation_text.find("atrito");
    assert(first_atrito != std::string::npos);
    assert(aggregation_text.find("atrito", first_atrito + 1) == std::string::npos);
    assert(aggregation_text.back() == '.');

    SemanticPlan subordinate;
    subordinate.intent = DialogIntent::EXPLAIN_CAUSE;
    subordinate.language = "pt";
    subordinate.tone = "warning";
    CausalPath subordinate_path;
    subordinate_path.steps = {make_step("atrito", "gerar", "calor"),
                              make_step("calor", "derreter", "gelo")};
    subordinate.paths.push_back(subordinate_path);
    const auto subordinate_text = realizer.realize(subordinate);
    assert(subordinate_text.find(", porque ") != std::string::npos);
    assert(subordinate_text.back() == '.');

    std::cout << relative_text << "\n" << aggregation_text << "\n" << subordinate_text
              << "\n[PASS] marco14 lexicalized grammar\n";
    return 0;
}
