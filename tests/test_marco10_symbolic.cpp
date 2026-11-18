#include "core/model_bundle.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace cognicore;
int main() {
    ModelBundle model;
    TextParser parser;
    const std::string corpus = "O atrito gera calor; por isso, o calor derrete gelo.";
    for (int i = 0; i < 40; ++i) model.realizer.observe_sentence_tokens(parser.tokenize(corpus));
    SemanticPlan plan;
    plan.intent = DialogIntent::EXPLAIN_CAUSE;
    plan.paths.push_back({{{"atrito", "gerar", "calor", RelationType::CAUSAL, 1, false, GramTense::PRES, ""}, {"calor", "derreter", "gelo", RelationType::CAUSAL, 1, false, GramTense::PRES, ""}}, 1});
    const auto output = model.realizer.realize(plan);
    assert(output.find("por isso") != std::string::npos);
    assert(output.find("gera") != std::string::npos && output.find("derrete") != std::string::npos);
    plan.paths[0].steps[0].src = "material inédito";
    const auto novel = model.realizer.realize(plan);
    assert(novel.find("material inédito") != std::string::npos);
    assert(corpus.find(novel) == std::string::npos);
    auto denied = plan;
    denied.paths[0].steps[0].negated = true;
    assert(model.realizer.realize(denied).find("não gera") != std::string::npos);
    assert(serialize_semantic_plan(denied).find("<NEG: true>") != std::string::npos);
    // A same-type frame must not substitute an unrelated predicate.
    model.realizer.add_frame({"[SUBJ:ENT] explode [OBJ:ENT]", RelationType::CAUSAL, 10000});
    assert(model.realizer.realize(plan).find("explode") == std::string::npos);
    const auto dir = std::filesystem::temp_directory_path() / ("cognicore_symbolic_" + std::to_string(getpid()));
    assert(model.save(dir.string()));
    ModelBundle loaded;
    assert(loaded.load(dir.string()));
    assert(loaded.realizer.realize(plan) == model.realizer.realize(plan));
    assert(loaded.realizer.num_ngrams() > 0);
    std::filesystem::remove_all(dir);
    std::cout << novel << '\n';
}
