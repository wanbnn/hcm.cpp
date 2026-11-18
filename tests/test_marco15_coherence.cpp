#include "core/coherence.hpp"
#include "core/language_rules.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>

using namespace cognicore;

int main() {
    LanguageRules rules;
    assert(rules.load_default());
    TextParser parser;
    CoherenceModel coherence;
    for (int i = 0; i < 20; ++i) {
        coherence.observe("O atrito gera calor. O calor derrete o gelo.", parser);
    }
    coherence.observe("O atrito gera gelo.", parser);
    assert(coherence.unigrams() > 0 && coherence.bigrams() > 0 && coherence.trigrams() > 0);

    CausalGraph graph;
    graph.add_fact("atrito", "gerar", "calor");
    graph.add_fact("calor", "derreter", "gelo");
    SemanticPlan plan;
    plan.language = "pt";
    plan.intent = DialogIntent::EXPLAIN_CAUSE;
    plan.paths.push_back({{{"atrito", "gerar", "calor", RelationType::CAUSAL, 1, false,
                            GramTense::PRES, "", GramGender::MASC, GramNumber::SING,
                            GramGender::MASC, GramNumber::SING}}, 1});
    plan.query_concepts = {"atrito", "calor"};

    const auto normal = coherence.score("O atrito gera o calor.", plan, "", graph, rules);
    const auto strange = coherence.score("O zzz gera o qqq.", plan, "", graph, rules);
    assert(normal.total > strange.total);
    assert(normal.language < strange.language);
    assert(normal.grounding > strange.grounding);

    const auto pronoun = coherence.score("Ele aquece a água.", plan,
                                         "O calor derrete o gelo.", graph, rules);
    const auto repeat = coherence.score("O calor derrete o gelo.", plan,
                                        "O calor derrete o gelo.", graph, rules);
    assert(pronoun.centering > repeat.centering);

    const auto chosen = coherence.choose({"O zzz gera o qqq.", "O atrito gera o calor."},
                                         plan, "", graph, rules);
    assert(chosen.find("atrito") != std::string::npos);

    const auto path = std::filesystem::temp_directory_path() /
        ("cognicore_coherence_" + std::to_string(getpid()) + ".bin");
    assert(coherence.save(path.string()));
    CoherenceModel loaded;
    assert(loaded.load(path.string()));
    assert(loaded.choose({"O zzz gera o qqq.", "O atrito gera o calor."},
                         plan, "", graph, rules).find("atrito") != std::string::npos);
    std::filesystem::remove(path);
    std::cout << "[PASS] marco15 coherence ranker\n";
    return 0;
}
