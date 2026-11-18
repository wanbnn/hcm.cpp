#include "core/assistant.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace cognicore;
int main() {
    ModelBundle bundle;
    TreeSitterCodeParser::ingest_source(
        "int double_value(int x) { return x * 2; }\nint increment(int x) { return x + 1; }\n",
        "workspace.cpp", CodeLanguage::CPP, bundle.code_graph, &bundle.sdm);
    AssistantRouter router;
    assert(router.classify("Olá", bundle) == AssistantRoute::CONVERSATIONAL_GENERAL);
    assert(router.classify("Por que o fogo causa calor?", bundle) == AssistantRoute::CAUSAL_REASONING);
    assert(router.classify("Explique a função double_value", bundle) == AssistantRoute::CODE_QUERY_OR_EXPLAIN);
    assert(router.classify("Gere uma função para combinar double_value e increment", bundle) == AssistantRoute::CODE_GENERATE_OR_PATCH);
    AssistantSession session(bundle);
    auto fact = session.respond("O atrito gera calor.");
    assert(fact.route == AssistantRoute::FACT_UPDATE);
    session.respond("O calor derrete gelo.");
    auto causal = session.respond("Por que o atrito causa gelo?");
    assert(causal.route == AssistantRoute::CAUSAL_REASONING && !causal.trace.paths.empty());
    assert(causal.trace.paths[0].steps.size() == 2);
    auto code = session.respond("Gere uma função para combinar double_value e increment");
    std::cout << code.text;
    assert(code.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(TreeSitterCodeParser::validate_syntax(code.text, CodeLanguage::CPP));
    assert(code.text.find("double_value(input)") != std::string::npos);
    assert(code.text.find("increment(step1)") != std::string::npos);
    const auto dir = std::filesystem::temp_directory_path() / ("cognicore_session_" + std::to_string(getpid()));
    assert(bundle.save(dir.string()));
    ModelBundle reloaded;
    assert(reloaded.load(dir.string()));
    AssistantSession resumed(reloaded);
    const auto explanation = resumed.respond("Explique a função double_value");
    assert(explanation.text.find("Retorno: int") != std::string::npos);
    assert(!reloaded.graph.query_direct("atrito").empty());
    assert(reloaded.code_graph.get_param_types("double_value") == std::vector<std::string>{"int"});
    std::filesystem::remove_all(dir);
}
