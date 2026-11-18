#include "core/assistant.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

int main() {
    ModelBundle bundle;
    bundle.graph.add_fact("atrito", "gerar", "calor");
    bundle.graph.add_fact("calor", "derreter", "gelo");
    bundle.graph.add_fact("gelo", "ser", "sólido", RelationType::TAXONOMIC);
    AssistantSession session(bundle);
    const size_t edges_before = bundle.graph.num_edges();

    const auto document = session.respond("Escreva um texto sobre calor");
    assert(document.route == AssistantRoute::DOCUMENT_GENERATION);
    assert(document.text.find("# ") != std::string::npos);
    assert(document.text.find("## ") != std::string::npos);
    assert(document.text.find("calor") != std::string::npos);

    const auto equation = session.respond("Resolva 2x + 3 = 11");
    assert(equation.route == AssistantRoute::MATH_REASONING);
    assert(equation.text.find("x = 4") != std::string::npos);

    const auto fraction = session.respond("Quanto é 1/2 + 1/3?");
    assert(fraction.route == AssistantRoute::MATH_REASONING);
    assert(fraction.text.find("5/6") != std::string::npos);

    const auto list_code = session.respond("Escreva um script python que soma uma lista de valores");
    assert(list_code.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(list_code.text.find("for value in values") != std::string::npos);

    const auto arithmetic_code = session.respond("script python que soma 2 + 2");
    assert(arithmetic_code.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(arithmetic_code.text.find("2 + 2") != std::string::npos);

    assert(bundle.graph.num_edges() == edges_before);
    std::cout << "[PASS] marco18 unified chat\n";
    return 0;
}
