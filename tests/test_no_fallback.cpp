#include "core/morphological_realizer.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

int main() {
    LanguageRules rules;
    assert(rules.load_default());

    rules.social_greet = {"pt|SINAL_GREET_MODELO"};
    rules.agree_templates = {"pt|SINAL_AGREE_MODELO"};
    rules.unknown_topic_templates = {"pt|SEM_CONEXAO:{concept}"};
    rules.resource_unknown = {"pt|RECURSO:{entity}:{capacity}"};
    rules.conditional_causal = {"pt|REGRA:{antecedent}->{consequent}"};
    rules.connectors_result = {"pt|CONECTOR_RESULTADO, "};
    MorphologicalRealizer realizer(rules);

    SemanticPlan greet;
    greet.language = "pt";
    greet.intent = DialogIntent::GREET;
    assert(realizer.realize(greet) == "SINAL_GREET_MODELO");

    SemanticPlan agree;
    agree.language = "pt";
    agree.intent = DialogIntent::AGREE;
    assert(realizer.realize(agree) == "SINAL_AGREE_MODELO");

    SemanticPlan unknown;
    unknown.language = "pt";
    unknown.intent = DialogIntent::ASK_CLARIFY;
    unknown.query_concepts = {"xyz"};
    assert(realizer.realize(unknown) == "SEM_CONEXAO:xyz");

    SemanticPlan resource;
    resource.language = "pt";
    ResourceAssessment assessment;
    assessment.entity = "servidor";
    assessment.capacity_surface = "2 GB";
    resource.resource = assessment;
    assert(realizer.realize(resource) == "RECURSO:servidor:2 GB");

    SemanticPlan conditional;
    conditional.language = "pt";
    conditional.conditional_conclusions = {{"motor aquece", "bomba falha"}};
    conditional.conditional_kinds = {static_cast<uint8_t>(ConditionKind::CAUSAL)};
    assert(realizer.realize(conditional) == "REGRA:motor aquece->bomba falha");

    std::cout << "[PASS] no hardcoded answer fallback\n";
    return 0;
}
