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
    session.set_online_learning(true); // this test exercises the opt-in learning path
    auto fact = session.respond("O atrito gera calor.");
    assert(fact.route == AssistantRoute::FACT_UPDATE);
    session.respond("O calor derrete gelo.");
    auto causal = session.respond("Por que o atrito causa gelo?");
    assert(causal.route == AssistantRoute::CAUSAL_REASONING && !causal.trace.paths.empty());
    assert(causal.trace.paths[0].steps.size() == 2);
    for (const auto& step : causal.trace.paths[0].steps) {
        assert(step.edge_id < bundle.graph.edges().size());
        const auto& edge = bundle.graph.edges()[step.edge_id];
        assert(bundle.graph.nodes()[edge.src].name == step.src);
        assert(bundle.graph.nodes()[edge.dst].name == step.dst);
    }
    const auto unknown = session.respond("O que você sabe sobre teletransporte?");
    assert(unknown.trace.paths.empty());
    assert(unknown.text.find("teletransporte") != std::string::npos);
    auto code = session.respond("Gere uma função para combinar double_value e increment");
    std::cout << code.text;
    assert(code.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(TreeSitterCodeParser::validate_syntax(code.text, CodeLanguage::CPP));
    assert(code.text.find("double_value(input)") != std::string::npos);
    assert(code.text.find("increment(step1)") != std::string::npos);
    const auto graph_before_rule = bundle.graph.edges().size();
    session.respond("Se carregar grafo sem mmap então excesso de memória.");
    assert(bundle.graph.edges().size() == graph_before_rule);
    bundle.graph.add_fact("excesso de memória", "causar", "lentidão");
    const auto conditional = session.respond("Se carregar grafo sem mmap, o que acontece?");
    assert(conditional.trace.plan.conditional_conclusions.size() == 1);
    assert(conditional.text.find("então excesso de memória") != std::string::npos);
    assert(conditional.text.find("lentidão") != std::string::npos);
    const auto not_applicable = session.respond("Se carregar grafo com mmap, o que acontece?");
    assert(not_applicable.trace.plan.conditional_conclusions.empty());
    const auto no_assumption = session.respond("Se não carregar grafo sem mmap, o que acontece?");
    assert(no_assumption.trace.plan.conditional_conclusions.empty());
    // A conditional is not a new asserted fact, and missing quantities remain unknown.
    session.respond("Meu servidor antigo só tem 2 GB de RAM e roda em CPU.");
    const auto before = bundle.graph.edges().size();
    const auto resource = session.respond("Se eu tentar carregar um grafo gigante inteiro na RAM lá sem mmap, o que acontece?");
    assert(resource.route == AssistantRoute::CAUSAL_REASONING);
    assert(bundle.graph.edges().size() == before);
    assert(resource.trace.plan.resource.has_value());
    assert(resource.trace.plan.resource->entity == "servidor antigo");
    assert(resource.trace.plan.resource->result == ResourceAssessment::Result::UNKNOWN);
    assert(!resource.trace.recalled_episodes.empty());
    assert(resource.trace.recalled_episodes.front().similarity > .15);
    assert(resource.trace.plan.assumptions.size() == 1);
    assert(resource.text.find("preciso saber") != std::string::npos);
    const auto exceeds = session.respond("Se eu carregar 3 GB na RAM do servidor antigo, o que acontece?");
    assert(exceeds.trace.plan.resource->result == ResourceAssessment::Result::EXCEEDS);
    const auto within = session.respond("Se eu carregar 1 GiB na RAM do servidor antigo, o que acontece?");
    assert(within.trace.plan.resource->result == ResourceAssessment::Result::WITHIN);
    assert(within.trace.plan.resource->demand_bytes == 1073741824ULL);
    // One-shot correction changes the constraint without fitting a policy.
    session.respond("Meu servidor antigo tem 4 GB de RAM.");
    const auto updated = session.respond("Se eu carregar 3 GB na RAM lá, o que acontece?");
    assert(updated.trace.plan.resource->result == ResourceAssessment::Result::WITHIN);
    assert(updated.trace.plan.resource->capacity_bytes == 4000000000ULL);
    assert(updated.trace.stage_ms.size() == 6);
    assert(updated.trace.debug_report().find("PLANO ANTES DO REALIZADOR") != std::string::npos);
    // Every reported fired clause must reproduce the actual class votes.
    std::vector<int32_t> reconstructed(updated.trace.class_votes.size());
    const auto clauses_per_class = bundle.tsetlin.config().clauses_per_class;
    const auto stride = updated.trace.policy_source == "context_policy" ? bundle.context_policy.clauses_per_class : clauses_per_class;
    for (size_t i = 0; i < updated.trace.fired_clauses.size(); ++i)
        reconstructed[updated.trace.fired_clauses[i] / stride] += updated.trace.fired_literals[i].sign;
    assert(reconstructed == updated.trace.class_votes);
    struct InspectingMouth : ISurfaceRealizer {
        mutable SemanticPlan received;
        std::string realize(const SemanticPlan& p) const override { received = p; return "realização isolada"; }
    } mouth;
    const auto isolated = session.respond("Se eu carregar 8 GB na RAM do servidor antigo, o que acontece?", CodeLanguage::CPP, &mouth);
    assert(isolated.text == "realização isolada");
    assert(mouth.received.resource->result == ResourceAssessment::Result::EXCEEDS);
    assert(serialize_semantic_plan(mouth.received) == serialize_semantic_plan(isolated.trace.plan));
    assert(!parse_memory_quantity("-3 GB"));
    assert(!parse_memory_quantity("3 GB para um servidor de 2 GB"));
    session.respond("Meu servidor antigo não tem 8 GB de RAM.");
    assert(bundle.episodic_memory.recall("servidor antigo", "ram_capacity")->fact.value == "4 gb");
    const auto dir = std::filesystem::temp_directory_path() / ("cognicore_session_" + std::to_string(getpid()));
    assert(bundle.save(dir.string()));
    ModelBundle reloaded;
    assert(reloaded.load(dir.string()));
    AssistantSession resumed(reloaded);
    resumed.set_online_learning(true);
    const auto explanation = resumed.respond("Explique a função double_value");
    assert(explanation.text.find("Retorno: int") != std::string::npos);
    assert(!reloaded.graph.query_direct("atrito").empty());
    assert(reloaded.code_graph.get_param_types("double_value") == std::vector<std::string>{"int"});
    const auto persisted = resumed.respond("Se eu carregar 5 GB na RAM do servidor antigo, o que acontece?");
    assert(persisted.trace.plan.resource && persisted.trace.plan.resource->capacity_bytes == 4000000000ULL);
    assert(persisted.trace.plan.resource->result == ResourceAssessment::Result::EXCEEDS);
    std::filesystem::remove_all(dir);
}
