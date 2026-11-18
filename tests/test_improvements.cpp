#include "core/analogical_realizer.hpp"
#include "core/assistant.hpp"
#include "core/code_synthesizer.hpp"
#include "core/context_memory.hpp"
#include "core/discourse_planner.hpp"
#include "core/graph_mmap.hpp"
#include "core/math_solver.hpp"
#include "core/model_bundle.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace cognicore;

void test_general_parser_and_conditionals() {
    TextParser parser;
    auto doc = parser.parse_text(
        "O pesquisador analisou o experimento. "
        "A equipe registrou o resultado, e o laboratório publicou a conclusão.");
    assert(doc.triples.size() >= 3);
    bool analyzed = false, published = false;
    for (const auto& triple : doc.triples) {
        if (triple.relation == "analisar" && triple.subject.find("pesquisador") != std::string::npos)
            analyzed = true;
        if (triple.relation == "publicar" && triple.object.find("conclusão") != std::string::npos)
            published = true;
    }
    assert(analyzed && published);

    const auto conditional = parser.parse_text("Se o motor aquece então a bomba falha.");
    assert(conditional.conditionals.size() == 1);
    assert(conditional.conditionals[0].antecedent.find("motor") != std::string::npos);
    assert(conditional.conditionals[0].consequent.find("bomba") != std::string::npos);
    // A conditional is not asserted as a fact.
    assert(conditional.triples.empty());
    std::cout << "[PASS] general parser + conditionals\n";
}

void test_context_memory_resolution_and_carryover() {
    TextParser parser;
    CausalGraph graph;
    ContextMemory memory;
    memory.observe_entity("servidor antigo", GramGender::MASC, GramNumber::SING);
    memory.observe_entity("fogo", GramGender::MASC, GramNumber::SING);
    auto resolved = memory.resolve("Se eu carregar tudo lá sem mmap, o que acontece?", parser);
    assert(resolved.has_value());
    assert(resolved->referent == "servidor antigo");
    assert(resolved->reason.find("contextual") != std::string::npos);

    memory.observe_concepts({"grafo", "memória"});
    memory.observe_user_turn("O grafo usa memória.", parser, graph);
    const auto carried = memory.carryover_concepts("E isso?", parser, graph);
    assert(!carried.empty());
    assert(carried.front() == "grafo" || carried.front() == "memória");

    // A follow-up that introduces its own concept must not inherit the topic.
    graph.add_fact("vento", "esfriar", "água");
    const auto explicit_turn = memory.carryover_concepts("O que o vento faz?", parser, graph);
    assert(explicit_turn.empty());
    std::cout << "[PASS] contextual memory\n";
}

void test_discourse_planner_prefers_seed_endpoint() {
    CausalGraph graph;
    graph.add_fact("atrito", "gerar", "calor");
    graph.add_fact("calor", "derreter", "gelo");
    graph.add_fact("gelo", "ser", "sólido", RelationType::TAXONOMIC);
    const std::vector<std::string> seeds{"atrito", "gelo"};
    const auto candidates = graph.find_paths(seeds, 3, 6, nullptr);
    const auto plan = DiscoursePlanner{}.plan(candidates, DialogIntent::EXPLAIN_CAUSE, false, seeds);
    assert(!plan.paths.empty());
    assert(plan.paths.front().steps.size() >= 2);
    assert(plan.paths.front().steps.back().dst == "gelo");
    assert(!plan.alternatives.empty());
    assert(plan.alternatives.front().steps.back().dst == "sólido");
    std::cout << "[PASS] discourse planner\n";
}

void test_dop_typed_clauses_and_contrast() {
    TextParser parser;
    AnalogicalFluencyRealizer realizer;
    const std::vector<std::string> corpus = {
        "O gelo é um sólido.",
        "O fogo gera calor; por isso, o calor provoca expansão; assim, a expansão move objetos.",
        "A chuva causa umidade; portanto, a umidade causa corrosão; consequentemente, a corrosão gera desgaste.",
        "O atrito gera calor, mas o vento esfria a água.",
        "O estudo gera compreensão. Além disso, a prática gera habilidade."
    };
    for (int i = 0; i < 6; ++i) for (const auto& sentence : corpus)
        realizer.observe(sentence, parser);
    realizer.train();

    SemanticPlan taxonomic;
    taxonomic.intent = DialogIntent::ANSWER_FACT;
    taxonomic.paths.push_back({{{"gelo", "ser", "sólido", RelationType::TAXONOMIC, 1, false,
                                  GramTense::PRES, "", GramGender::MASC, GramNumber::SING,
                                  GramGender::MASC, GramNumber::SING}}, 1});
    const auto taxonomic_derivation = realizer.derive(taxonomic);
    assert(!taxonomic_derivation.candidates.empty());
    const auto& taxonomic_text = taxonomic_derivation.candidates[taxonomic_derivation.selected].text;
    assert(taxonomic_text.find("é um sólido") != std::string::npos);
    for (const auto& candidate : taxonomic_derivation.candidates)
        assert(AnalogicalFluencyRealizer::preserves_plan(candidate, taxonomic));

    SemanticPlan contrast;
    contrast.intent = DialogIntent::EXPLAIN_CAUSE;
    contrast.paths.push_back({{{"atrito", "gerar", "calor", RelationType::CAUSAL, 1, false,
                                GramTense::PRES, ""},
                               {"vento", "esfriar", "água", RelationType::CAUSAL, 1, false,
                                GramTense::PRES, ""}}, 1});
    const auto contrast_derivation = realizer.derive(contrast);
    bool has_contrast = false;
    for (const auto& candidate : contrast_derivation.candidates) {
        assert(AnalogicalFluencyRealizer::preserves_plan(candidate, contrast));
        if (candidate.text.find(", mas ") != std::string::npos) has_contrast = true;
    }
    assert(has_contrast);
    std::cout << "[PASS] DOP typed clauses + contrast\n";
}

void test_broad_ast_synthesis() {
    Codebook codebook;
    CodeGraph graph;
    const std::string library =
        "int add(int a, int b) { return a + b; }\n"
        "int scale(int value, int factor) { return value * factor; }\n"
        "void log_value(int value) { (void)value; }\n"
        "int answer() { return 42; }\n"
        "struct Engine { static int run(int x) { return x + 1; } };\n";
    assert(TreeSitterCodeParser::ingest_source(library, "improvements.cpp", CodeLanguage::CPP, graph) >= 5);
    CodeSynthesizer synthesizer;

    const auto multi = synthesizer.synthesize("add e scale", CodeLanguage::CPP, graph, codebook);
    assert(multi.syntax_valid);
    assert(multi.code.find("add(input, 0)") != std::string::npos);
    assert(multi.code.find("scale(step1, input)") != std::string::npos);

    const auto void_result = synthesizer.synthesize("scale e log_value", CodeLanguage::CPP, graph, codebook);
    assert(void_result.syntax_valid);
    assert(void_result.code.find("log_value(step1)") != std::string::npos);
    assert(void_result.code.find("void synthesized_pipeline") != std::string::npos);

    const auto zero_arg = synthesizer.synthesize("answer", CodeLanguage::CPP, graph, codebook);
    assert(zero_arg.syntax_valid);
    assert(zero_arg.code.find("synthesized_pipeline()") != std::string::npos);
    assert(zero_arg.code.find("return answer()") != std::string::npos);

    const auto method = synthesizer.synthesize("run", CodeLanguage::CPP, graph, codebook);
    assert(method.syntax_valid);
    assert(method.code.find("Engine::run(input)") != std::string::npos);
    std::cout << "[PASS] broad AST synthesis\n";
}

void test_embedded_dependency_parse() {
    TextParser parser;
    auto doc = parser.parse_text("O calor, que derrete gelo, aquece a água.");
    bool relative = false, main = false;
    for (const auto& triple : doc.triples) {
        if (triple.subject == "calor" && triple.relation == "derreter" && triple.object == "gelo") relative = true;
        if (triple.subject == "calor" && triple.relation == "aquecer" && triple.object == "água") main = true;
    }
    assert(relative && main);
    for (const auto& triple : doc.triples) assert(triple.subject != "que");

    auto quantified = parser.parse_text("Todos os motores geram calor e nenhum grafo sem mmap cabe na RAM.");
    bool all = false, none = false;
    for (const auto& triple : quantified.triples) {
        if (triple.quantifier == "ALL") all = true;
        if (triple.quantifier == "NO") none = true;
    }
    assert(all && none);

    auto temporal = parser.parse_text("Quando o motor aquece, a bomba falha.");
    assert(temporal.conditionals.size() == 1);
    assert(temporal.conditionals[0].kind == ConditionKind::TEMPORAL);
    assert(temporal.triples.empty());

    const auto& morph = temporal.morph_sentences;
    assert(!morph.empty());
    bool has_subject = false, has_advcl = false;
    for (const auto& token : morph.front()) {
        if (token.deprel == "nsubj") has_subject = true;
        if (token.deprel == "advcl") has_advcl = true;
    }
    assert(has_subject && has_advcl);
    std::cout << "[PASS] embedded dependency parser\n";
}

void test_graph_mmap_equivalence() {
    CausalGraph graph;
    graph.add_fact("atrito", "gerar", "calor");
    graph.add_fact("calor", "derreter", "gelo");
    graph.add_fact("gelo", "ser", "sólido", RelationType::TAXONOMIC);
    const auto path = std::filesystem::temp_directory_path() /
        ("cognicore_graph_mmap_" + std::to_string(getpid()) + ".bin");
    assert(MappedCausalGraph::write(path.string(), graph));
    auto mapped = MappedCausalGraph::open(path.string());
    assert(mapped && mapped->valid());
    assert(mapped->num_nodes() == graph.num_nodes());
    assert(mapped->num_edges() == graph.num_edges());
    assert(mapped->find_node("calor") == graph.find_node("calor"));
    const auto direct_memory = graph.query_direct("atrito");
    const auto direct_mapped = mapped->query_direct("atrito");
    assert(direct_memory.size() == direct_mapped.size());
    if (!direct_memory.empty()) {
        assert(direct_memory.front().src == direct_mapped.front().src);
        assert(direct_memory.front().dst == direct_mapped.front().dst);
    }
    const std::vector<std::string> seeds{"atrito", "gelo"};
    const auto paths_memory = graph.find_paths(seeds, 3, 6, nullptr);
    const auto paths_mapped = mapped->find_paths(seeds, 3, 6);
    assert(!paths_mapped.empty());
    assert(paths_memory.front().steps.size() == paths_mapped.front().steps.size());
    assert(paths_memory.front().steps.back().dst == paths_mapped.front().steps.back().dst);
    std::filesystem::remove(path);
    std::cout << "[PASS] graph mmap equivalence\n";
}

void test_ast_overloads_methods_patches() {
    Codebook codebook;
    CodeGraph graph;
    const std::string library =
        "int convert(int x) { return x + 1; }\n"
        "double convert(double x) { return x + 1.0; }\n"
        "struct Engine {\n"
        "  int apply(int x) { return x + 1; }\n"
        "  static int run(int x) { return x + 2; }\n"
        "};\n"
        "int process(const std::string& path) { return convert(1); }\n";
    assert(TreeSitterCodeParser::ingest_source(library, "ast_v3.cpp", CodeLanguage::CPP, graph) >= 5);
    assert(graph.find_symbols_by_suffix("convert").size() == 2);
    CodeSynthesizer synthesizer;
    const auto instance = synthesizer.synthesize("apply", CodeLanguage::CPP, graph, codebook);
    assert(instance.syntax_valid);
    assert(instance.code.find("Engine receiver_0") != std::string::npos);
    assert(instance.code.find("receiver_0.apply(input)") != std::string::npos);
    const auto static_method = synthesizer.synthesize("run", CodeLanguage::CPP, graph, codebook);
    assert(static_method.syntax_valid);
    assert(static_method.code.find("Engine::run(input)") != std::string::npos);
    const auto patch = synthesizer.synthesize("patch process", CodeLanguage::CPP, graph, codebook);
    assert(patch.syntax_valid && patch.is_patch);
    assert(patch.patch_target == "process");
    assert(patch.code.find("patched_process") != std::string::npos);
    std::cout << "[PASS] AST overloads, methods and patches\n";
}

void test_math_solver_and_ppm_calibration() {
    MathSolver solver;
    const auto arithmetic = solver.solve("Quanto é 2 + 3 * 4?");
    assert(arithmetic && arithmetic->value == 14);
    const auto word_problem = solver.solve("João tem 5 maçãs e compra mais 3. Quantas maçãs ele tem?");
    assert(word_problem && word_problem->value == 8);
    const auto subtraction = solver.solve("Maria tinha 10 balas e perdeu 4.");
    assert(subtraction && subtraction->value == 6);

    AnalogicalFluencyRealizer realizer;
    TextParser parser;
    realizer.observe("O fogo gera calor e o calor aquece a água.", parser);
    realizer.train();
    const auto evaluation = realizer.evaluate_text("o fogo gera calor");
    assert(evaluation.bytes == 17);
    assert(evaluation.bits_per_byte > 0.0);
    assert(evaluation.perplexity > 1.0);
    std::cout << "[PASS] math solver + PPM calibration\n";
}

void test_code_followups_and_no_fact_pollution() {
    ModelBundle bundle;
    AssistantRouter router;
    assert(router.classify("script python que soma 2 + 2", bundle) == AssistantRoute::CODE_GENERATE_OR_PATCH);
    AssistantSession session(bundle);
    const size_t edges_before = bundle.graph.num_edges();
    const auto first = session.respond("script python que soma 2 + 2");
    assert(first.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(first.text.find("def synthesized_script") != std::string::npos);
    assert(first.text.find("2 + 2") != std::string::npos);
    const auto followup = session.respond("crie ele");
    assert(followup.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(followup.text.find("2 + 2") != std::string::npos);
    const auto correction = session.respond("não, você que deve criar o script.");
    assert(correction.route == AssistantRoute::CODE_GENERATE_OR_PATCH);
    assert(correction.text.find("2 + 2") != std::string::npos);
    assert(bundle.graph.num_edges() == edges_before);
    std::cout << "[PASS] code follow-ups do not pollute facts\n";
}

void test_model_bundle_mapped_graph_persistence() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("cognicore_mmap_model_" + std::to_string(getpid()));
    ModelBundle bundle;
    bundle.graph.add_fact("atrito", "gerar", "calor");
    bundle.graph.add_fact("calor", "derreter", "gelo");
    assert(bundle.save(dir.string()));
    assert(std::filesystem::exists(dir / "language_rules.json"));
    ModelBundle loaded;
    assert(loaded.load(dir.string(), true));
    assert(loaded.mapped_graph && loaded.mapped_graph->valid());
    assert(loaded.mapped_graph->find_node("atrito") != UINT32_MAX);
    assert(loaded.language_rules.contains_any("script python", loaded.language_rules.code_cues));
    assert(loaded.refresh_mapped_graph(dir.string()));
    std::filesystem::remove_all(dir);
    std::cout << "[PASS] model bundle mapped graph\n";
}

int main() {
    test_general_parser_and_conditionals();
    test_context_memory_resolution_and_carryover();
    test_discourse_planner_prefers_seed_endpoint();
    test_dop_typed_clauses_and_contrast();
    test_broad_ast_synthesis();
    test_embedded_dependency_parse();
    test_graph_mmap_equivalence();
    test_ast_overloads_methods_patches();
    test_math_solver_and_ppm_calibration();
    test_code_followups_and_no_fact_pollution();
    test_model_bundle_mapped_graph_persistence();
    return 0;
}
