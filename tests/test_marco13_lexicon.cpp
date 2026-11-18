#include "core/lexicon.hpp"
#include "core/morphological_realizer.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>

using namespace cognicore;

int main() {
    LanguageRules rules;
    assert(rules.load_default());
    TextParser parser;
    Lexicon lexicon;
    lexicon.load_synonyms(rules);
    for (int i = 0; i < 6; ++i) {
        lexicon.observe("O host produz calor.", parser, "technical");
        lexicon.observe("O motor produz calor.", parser, "technical");
    }
    lexicon.observe("O servidor gera calor.", parser, "technical");
    assert(lexicon.entries() > 0 && lexicon.collocations() > 0);
    assert(lexicon.choose_verb("gerar", "calor", "technical") == "produzir");
    assert(lexicon.choose_noun("servidor", "produzir", "technical") == "host");

    SemanticPlan plan;
    plan.intent = DialogIntent::EXPLAIN_CAUSE;
    plan.tone = "technical";
    plan.paths.push_back({{{"servidor", "gerar", "calor", RelationType::CAUSAL, 1, false,
                            GramTense::PRES, "", GramGender::MASC, GramNumber::SING,
                            GramGender::MASC, GramNumber::SING}}, 1});
    const auto lexical = LexicalChooser{}.apply(plan, lexicon);
    assert(lexical.paths[0].steps[0].surface_rel == "produzir");
    assert(lexical.paths[0].steps[0].rel == "gerar");
    const auto text = MorphologicalRealizer{}.realize(lexical);
    assert(text.find("produz") != std::string::npos);
    assert(text.find("host") != std::string::npos);

    const auto path = std::filesystem::temp_directory_path() /
        ("cognicore_lexicon_" + std::to_string(getpid()) + ".bin");
    assert(lexicon.save(path.string()));
    Lexicon loaded;
    loaded.load_synonyms(rules);
    assert(loaded.load(path.string()));
    assert(loaded.choose_verb("gerar", "calor", "technical") == "produzir");
    std::filesystem::remove(path);
    std::cout << "[PASS] marco13 lexicon and lexical choice\n";
    return 0;
}
