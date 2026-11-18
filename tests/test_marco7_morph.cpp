#include "core/morphological_realizer.hpp"
#include "core/text_parser.hpp"
#include "core/model_bundle.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unistd.h>

#include <cassert>
#include <iostream>

using namespace cognicore;

void test_conllu_and_negation_parsing() {
    TextParser parser;

    const std::string conllu =
        "# text = A água não gera fogo.\n"
        "1\tA\to\tDET\t_\tDefinite=Def|Gender=Fem|Number=Sing\t2\tdet\t_\t_\n"
        "2\tágua\tágua\tNOUN\t_\tGender=Fem|Number=Sing\t4\tnsubj\t_\t_\n"
        "3\tnão\tnão\tADV\t_\tPolarity=Neg\t4\tadvmod\t_\t_\n"
        "4\tgera\tgerar\tVERB\t_\tMood=Ind|Number=Sing|Person=3|Tense=Pres\t0\troot\t_\t_\n"
        "5\tfogo\tfogo\tNOUN\t_\tGender=Masc|Number=Sing\t4\tobj\t_\t_\n"
        "6\t.\t.\tPUNCT\t_\t_\t4\tpunct\t_\t_\n\n";

    auto doc = parser.parse_conllu(conllu);
    assert(doc.triples.size() == 1);
    assert(doc.triples[0].subject == "água");
    assert(doc.triples[0].relation == "gerar");
    assert(doc.triples[0].object == "fogo");
    assert(doc.triples[0].negated == true);
    assert(doc.triples[0].subj_gender == GramGender::FEM);
    assert(doc.triples[0].subj_number == GramNumber::SING);

    // Rule-based fallback with negation & modality
    auto doc2 = parser.parse_text("O gelo não pode gerar fogo.");
    assert(doc2.triples.size() == 1);
    assert(doc2.triples[0].negated == true);
    assert(doc2.triples[0].modality == "pode");

    std::cout << "[PASS] test_conllu_and_negation_parsing\n";
}

void test_morphological_realizer_multihop_and_anaphora() {
    MorphologicalRealizer realizer;

    // Test verb conjugation & preposition contraction
    assert(MorphologicalRealizer::conjugate_verb("gerar", GramTense::PRES, GramNumber::SING, "pt") == "gera");
    assert(MorphologicalRealizer::conjugate_verb("gerar", GramTense::PAST, GramNumber::PLUR, "pt") == "geraram");
    assert(MorphologicalRealizer::conjugate_verb("derreter", GramTense::FUT, GramNumber::SING, "pt") == "derreterá");
    assert(MorphologicalRealizer::contract_prepositions("efeito de o calor em a água") == "efeito do calor na água");

    // 3-hop path where subject repeats in step 0 and step 1 (triggers pronominal anaphora)
    // and step 2 is inhibitory (triggers "Entretanto,")
    CausalPath path;
    path.score = 3.0f;
    path.steps.push_back(CausalStep{
        "energia", "gerar", "calor", RelationType::CAUSAL, 0.9f,
        false, GramTense::PRES, "", GramGender::FEM, GramNumber::SING, GramGender::MASC, GramNumber::SING
    });
    path.steps.push_back(CausalStep{
        "energia", "aquecer", "água", RelationType::CAUSAL, 0.9f,
        false, GramTense::PRES, "pode", GramGender::FEM, GramNumber::SING, GramGender::FEM, GramNumber::SING
    });
    path.steps.push_back(CausalStep{
        "água", "apagar", "fogo", RelationType::INHIBITS, 0.9f,
        false, GramTense::PRES, "", GramGender::FEM, GramNumber::SING, GramGender::MASC, GramNumber::SING
    });

    SemanticPlan plan;
    plan.intent = DialogIntent::EXPLAIN_CAUSE;
    plan.language = "pt";
    plan.paths.push_back(path);

    std::string out = realizer.realize(plan);
    std::cout << "MorphologicalRealizer output:\n  " << out << "\n";

    // Must start with "A energia gera o calor."
    assert(out.find("A energia gera o calor.") != std::string::npos);
    // Second sentence must use anaphoric pronoun "ela" instead of repeating "energia", with modality "pode aquecer"
    assert(out.find("Além disso, ela pode aquecer a água.") != std::string::npos);
    // Third sentence must use contrastive connector "Entretanto,"
    assert(out.find("Entretanto, a água apaga o fogo.") != std::string::npos);

    std::cout << "[PASS] test_morphological_realizer_multihop_and_anaphora\n";
}

void test_dependency_features_and_english() {
    TextParser parser;
    const auto doc = parser.parse_conllu(
        "# text = As chamas nunca poderão gerar gases quentes.\r\n"
        "1-2\tNas\t_\t_\t_\t_\t_\t_\t_\t_\r\n"
        "1\tAs\to\tDET\t_\tGender=Fem|Number=Plur\t2\tdet\t_\t_\r\n"
        "2\tchamas\tchama\tNOUN\t_\tGender=Fem|Number=Plur\t5\tnsubj\t_\t_\r\n"
        "3\tnunca\tnunca\tADV\t_\tPolarity=Neg\t5\tadvmod\t_\t_\r\n"
        "4\tpoderão\tpoder\tAUX\t_\tTense=Fut|Number=Plur\t5\taux\t_\t_\r\n"
        "5\tgerar\tgerar\tVERB\t_\tVerbForm=Inf\t0\troot\t_\t_\r\n"
        "6\tgases\tgás\tNOUN\t_\tGender=Masc|Number=Plur\t5\tobj\t_\t_\r\n"
        "7\tquentes\tquente\tADJ\t_\tNumber=Plur\t6\tamod\t_\t_");
    assert(doc.morph_sentences.size() == 1 && doc.morph_sentences[0].size() == 7);
    assert(doc.morph_sentences[0][6].upos == "ADJ");
    assert(doc.triples.size() == 1);
    const auto& t = doc.triples[0];
    assert(t.subject == "chamas" && t.object == "gases quentes");
    assert(t.negated && t.modality == "poder" && t.tense == GramTense::FUT);
    CausalStep step{t.subject, t.relation, t.object, t.type, t.confidence,
                   t.negated, t.tense, t.modality, t.subj_gender, t.subj_number,
                   t.obj_gender, t.obj_number};
    SemanticPlan plan;
    plan.paths.push_back({{step}, 1});
    assert(MorphologicalRealizer{}.realize(plan) == "As chamas não poderão gerar os gases quentes.");
    bool rejected = false;
    try { parser.parse_conllu("x\tbroken"); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    assert(MorphologicalRealizer::conjugate_verb("study", GramTense::PRES, GramNumber::SING, "en") == "studies");
    assert(MorphologicalRealizer::conjugate_verb("write", GramTense::PAST, GramNumber::SING, "en") == "wrote");
    assert(MorphologicalRealizer::conjugate_verb("be", GramTense::SUBJ, GramNumber::SING, "en") == "be");
    assert(MorphologicalRealizer::conjugate_verb("apagar", GramTense::SUBJ, GramNumber::PLUR) == "apaguem");
    assert(MorphologicalRealizer::contract_prepositions("de o fogo") == "do fogo");
    plan.language = "en";
    step = {"fire", "be", "hot", RelationType::PROPERTY, 1, true, GramTense::PAST, ""};
    plan.paths[0].steps = {step};
    assert(MorphologicalRealizer{}.realize(plan) == "The fire was not hot.");
    step.rel = "cause"; step.dst = "heat"; step.type = RelationType::CAUSAL; step.tense = GramTense::FUT;
    plan.paths[0].steps = {step};
    assert(MorphologicalRealizer{}.realize(plan) == "The fire will not cause the heat.");
}

void test_graph_persistence_and_online_metadata() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("cognicore_marco7_" + std::to_string(getpid()));
    ModelBundle bundle;
    bundle.infer_and_respond("As chamas não podem gerar gases.", true);
    auto steps = bundle.graph.query_direct("chamas");
    assert(steps.size() == 1 && steps[0].negated);
    assert(steps[0].src_number == GramNumber::PLUR && steps[0].src_gender == GramGender::FEM);
    bundle.graph.add_fact("chamas", "gerar", "gases", RelationType::CAUSAL, 1);
    bundle.graph.add_fact("chamas", "gerar", "gases", RelationType::CAUSAL, 1, false, true, GramTense::FUT, "poder");
    bundle.graph.consolidate();
    assert(bundle.graph.num_edges() == 3);
    assert(bundle.save(dir.string()));
    ModelBundle loaded;
    assert(loaded.load(dir.string(), true));
    const auto restored = loaded.graph.query_direct("chamas");
    assert(restored.size() == 3);
    bool found = false;
    for (const auto& st : restored) if (st.modality == "podem") {
        assert(st.negated && st.src_number == GramNumber::PLUR && st.src_gender == GramGender::FEM);
        found = true;
    }
    assert(found);
    // Replace only graph.bin with the pre-M7 layout to exercise migration.
    {
        std::ofstream out(dir / "graph.bin", std::ios::binary);
        auto pod = [&](const auto& value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); };
        auto str = [&](const std::string& value) { pod(uint32_t(value.size())); out.write(value.data(), value.size()); };
        pod(uint32_t(0x434F4747)); pod(uint32_t(2)); pod(uint32_t(1)); pod(uint32_t(1));
        pod(uint32_t(0)); pod(uint32_t(1)); str("água");
        pod(uint32_t(1)); pod(uint32_t(1)); str("fogo"); str("apaga");
        struct LegacyEdge { uint32_t s, r, d; RelationType type; uint32_t freq; float conf; bool taught; };
        const LegacyEdge edge{0, 0, 1, RelationType::INHIBITS, 1, 0.9f, false};
        pod(edge);
    }
    ModelBundle legacy;
    assert(legacy.load(dir.string(), true));
    const auto old = legacy.graph.query_direct("água");
    assert(old.size() == 1 && old[0].src_gender == GramGender::FEM && !old[0].negated);
    assert(legacy.save(dir.string()));
    ModelBundle migrated;
    assert(migrated.load(dir.string(), false));
    assert(migrated.graph.num_edges() == 1);
    std::filesystem::remove_all(dir);
}

int main() {
    test_conllu_and_negation_parsing();
    test_morphological_realizer_multihop_and_anaphora();
    test_dependency_features_and_english();
    test_graph_persistence_and_online_metadata();
    return 0;
}
