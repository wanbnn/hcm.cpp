#include "core/model_bundle.hpp"
#include "core/morphological_realizer.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <unistd.h>
using namespace cognicore;
int main() {
    ModelBundle model;
    TextParser parser;
    const std::vector<std::string> corpus = {
        "O fogo gera calor; por isso, o calor provoca expansão; assim, a expansão move objetos.",
        "A chuva causa umidade; portanto, a umidade causa corrosão; consequentemente, a corrosão gera desgaste.",
        "O estudo gera compreensão. Além disso, a prática gera habilidade. Também, a leitura gera conhecimento.",
        "O vento gera movimento, e o movimento gera energia."
    };
    for (int i = 0; i < 8; ++i) for (const auto& sentence : corpus) model.fluent_realizer.observe(sentence, parser);
    model.fluent_realizer.train();
    SemanticPlan plan; plan.intent = DialogIntent::EXPLAIN_CAUSE;
    plan.paths.push_back({{{"atrito inédito", "gerar", "calor", RelationType::CAUSAL, 1, false, GramTense::PRES, ""},
        {"calor", "causar", "dilatação", RelationType::CAUSAL, 1, false, GramTense::PRES, ""},
        {"dilatação", "gerar", "movimento", RelationType::CAUSAL, 1, false, GramTense::PRES, ""}}, 1});
    plan.paths[0].steps[1].dst_gender = GramGender::FEM;
    plan.paths[0].steps[2].src_gender = GramGender::FEM;
    const auto before = serialize_semantic_plan(plan);
    const auto derivation = model.fluent_realizer.derive(plan);
    assert(derivation.candidates.size() >= 8 && derivation.candidates.size() <= 16);
    std::set<std::string> unique;
    for (const auto& candidate : derivation.candidates) {
        assert(AnalogicalFluencyRealizer::preserves_plan(candidate, plan));
        assert(candidate.text.find("atrito inédito") != std::string::npos);
        assert(candidate.text.find("a dilatação") != std::string::npos);
        assert(candidate.text.find("o dilatação") == std::string::npos);
        assert(candidate.ppm_cost_q16 > 0);
        assert(candidate.fragment_ids.size() >= 3);
        for (const auto& sentence : corpus) assert(sentence.find(candidate.text) == std::string::npos);
        unique.insert(candidate.text);
    }
    assert(unique.size() == derivation.candidates.size());
    assert(!derivation.active_slots.empty() && !derivation.retrieved_fragments.empty());
    for (const auto& c : derivation.candidates) assert(derivation.candidates[derivation.selected].rank_q16 <= c.rank_q16);
    assert(serialize_semantic_plan(plan) == before);
    auto corrupt = derivation.candidates.front(); corrupt.realized_steps[0].negated = true;
    assert(!AnalogicalFluencyRealizer::preserves_plan(corrupt, plan));
    auto missing = derivation.candidates.front(); missing.text = "O atrito gera calor.";
    assert(!AnalogicalFluencyRealizer::preserves_plan(missing, plan));
    auto inserted = derivation.candidates.front(); inserted.text += " O gelo explode.";
    assert(!AnalogicalFluencyRealizer::preserves_plan(inserted, plan));
    auto negative = plan; negative.paths[0].steps[0].negated = true;
    negative.paths[0].steps[0].modality = "poder";
    const auto denied = model.fluent_realizer.derive(negative);
    for (const auto& c : denied.candidates) {
        assert(c.text.find("não pode gerar") != std::string::npos);
        assert(AnalogicalFluencyRealizer::preserves_plan(c, negative));
    }
    PPMModel ppm;
    for (int i = 0; i < 50; ++i) ppm.observe("o fogo gera calor e o calor aquece a água.");
    assert(ppm.cost_q16("o fogo gera calor") < ppm.cost_q16("zxq bkj yvp zzz"));
    const auto dir = std::filesystem::temp_directory_path() / ("cognicore_dop_" + std::to_string(getpid()));
    assert(model.save(dir.string()));
    ModelBundle loaded; assert(loaded.load(dir.string()));
    assert(loaded.fluent_realizer.mapped() && loaded.fluent_realizer.ready());
    assert(loaded.fluent_realizer.realize(plan) == model.fluent_realizer.realize(plan));
    // Atomic replacement must preserve live mapped readers, including saving to the same path.
    assert(loaded.save(dir.string()));
    assert(loaded.fluent_realizer.realize(plan) == model.fluent_realizer.realize(plan));
    loaded.fluent_realizer.observe("O frio gera gelo; por isso, o gelo ocupa espaço.", parser);
    loaded.fluent_realizer.train(); assert(loaded.save(dir.string()));
    std::ofstream broken(dir / "broken.bin", std::ios::binary); broken << "DOP1"; broken.close();
    assert(!loaded.fluent_realizer.load((dir / "broken.bin").string()));
    assert(!loaded.fluent_realizer.realize(plan).empty());
    std::filesystem::remove_all(dir);
    std::cout << derivation.candidates[derivation.selected].text << '\n';
}
