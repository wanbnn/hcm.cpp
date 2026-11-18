#include "core/model_bundle.hpp"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cognicore;

void test_end_to_end_cognitive_pipeline() {
    const std::string model_dir = "/tmp/cognicore_e2e_test_model";
    fs::remove_all(model_dir);

    // 1. Mini synthetic physics dataset (multi-hop facts in separate sentences)
    const std::string book_text =
        "O atrito gera calor. "
        "O calor derrete gelo. "
        "O gelo é um sólido. "
        "A evaporação produz vapor. "
        "O vapor forma nuvens. "
        "A água apaga fogo. "
        "O fogo aquece água. "
        "A gravidade provoca queda.";

    // 2. 20 synthetic dialogue examples for Tsetlin Machine
    struct DialEx {
        std::string text;
        DialogIntent intent;
    };
    std::vector<DialEx> dialogues = {
        {"Por que o gelo derrete?", DialogIntent::EXPLAIN_CAUSE},
        {"Por que o atrito aquece?", DialogIntent::EXPLAIN_CAUSE},
        {"Qual a causa do calor?", DialogIntent::EXPLAIN_CAUSE},
        {"Por que a nuvem se forma?", DialogIntent::EXPLAIN_CAUSE},
        {"O que causa vapor?", DialogIntent::EXPLAIN_CAUSE},
        {"O que o atrito faz com o gelo?", DialogIntent::EXPLAIN_CAUSE},
        {"O que é gelo?", DialogIntent::ANSWER_FACT},
        {"O que a água faz?", DialogIntent::ANSWER_FACT},
        {"Qual o efeito da gravidade?", DialogIntent::ANSWER_FACT},
        {"O que o fogo aquece?", DialogIntent::ANSWER_FACT},
        {"Compare gelo e vapor.", DialogIntent::COMPARE},
        {"Qual a diferença entre calor e fogo?", DialogIntent::COMPARE},
        {"Se houver atrito, o que acontece?", DialogIntent::HYPOTHESIZE},
        {"Se o calor aumentar, o gelo derrete?", DialogIntent::HYPOTHESIZE},
        {"Olá, tudo bem?", DialogIntent::GREET},
        {"Oi, como funciona?", DialogIntent::GREET},
        {"Sim, concordo.", DialogIntent::AGREE},
        {"Isso faz sentido.", DialogIntent::AGREE},
        {"Não entendi bem.", DialogIntent::ASK_CLARIFY},
        {"Pode explicar melhor?", DialogIntent::ASK_CLARIFY}
    };

    {
        ModelBundle bundle;
        bundle.parser.load_rules_json(std::string(COGNICORE_DATA_DIR) + "/linguistic_rules.json");

        // Ingest book text
        auto doc = bundle.parser.parse_text(book_text);
        assert(doc.triples.size() >= 6);

        for (const auto& s_toks : doc.sentences_tokens) {
            bundle.realizer.observe_sentence_tokens(s_toks);
            for (size_t i = 0; i < s_toks.size(); ++i) {
                std::vector<std::string> ctx;
                for (size_t j = (i >= 2 ? i - 2 : 0); j <= std::min(s_toks.size() - 1, i + 2); ++j) {
                    if (i != j) ctx.push_back(s_toks[j]);
                }
                bundle.codebook.observe_context(s_toks[i], ctx);
            }
        }
        bundle.codebook.finalize_semantics();

        for (const auto& tr : doc.triples) {
            bundle.graph.add_fact(tr.subject, tr.relation, tr.object, tr.type, tr.confidence);
            HyperVector addr = HyperVector::from_token(tr.subject);
            HyperVector data = HyperVector::from_token(tr.object);
            bundle.sdm.write(addr, data);
        }
        for (const auto& cf : doc.frames) {
            bundle.realizer.add_frame(cf);
        }

        // Train dialogue Tsetlin Machine
        std::vector<std::vector<uint64_t>> X;
        std::vector<size_t> y;
        for (const auto& ex : dialogues) {
            X.push_back(bundle.encode_dialogue_features(ex.text));
            y.push_back(static_cast<size_t>(ex.intent));
        }
        bundle.tsetlin.fit(X, y, 25);

        // Consolidate ("Sleep")
        bundle.graph.consolidate();
        bundle.realizer.consolidate_frames();
        bundle.sdm.binarize_for_inference();
        bundle.sparse_tsetlin = bundle.tsetlin.export_sparse();
        bundle.has_sparse_tsetlin = true;

        assert(bundle.save(model_dir));
        std::cout << "[PASS] Ingest, Train, Consolidate & Save (disk="
                  << ModelBundle::disk_size_bytes(model_dir) << " bytes)\n";
    }

    // 3. Reload in a brand-new ModelBundle using mmap (CPU-only)
    ModelBundle loaded;
    assert(loaded.load(model_dir, true));
    assert(loaded.has_sparse_tsetlin);
    assert(loaded.sdm.is_binarized());

    // 4. Multi-hop reasoning query combining TWO separate sentences:
    //    "O atrito gera calor." + "O calor derrete gelo."
    auto trace = loaded.infer_and_respond("Por que o atrito afeta o gelo?");
    std::cout << "Query: 'Por que o atrito afeta o gelo?'\n";
    std::cout << "  Intent: " << intent_to_string(trace.intent) << "\n";
    std::cout << "  Paths found: " << trace.paths.size() << "\n";
    std::cout << "  Response: " << trace.response << "\n";

    assert(!trace.paths.empty());
    assert(trace.paths[0].steps.size() >= 2);
    assert(trace.paths[0].steps[0].src == "atrito");
    assert(trace.paths[0].steps.back().dst == "gelo");
    // Verify the generated sentence is NOT a verbatim copy of any original sentence
    assert(book_text.find(trace.response) == std::string::npos);

    // 5. Online one-shot learning during chat (without retraining Tsetlin!)
    auto learn_trace = loaded.infer_and_respond("O vento esfria a água.", true);
    std::cout << "Teach: 'O vento esfria a água.' -> " << learn_trace.response << "\n";
    assert(learn_trace.intent == DialogIntent::AGREE);

    auto query_new = loaded.infer_and_respond("O que o vento faz com o fogo?", false);
    std::cout << "Query after online teach: 'O que o vento faz com o fogo?'\n";
    std::cout << "  Paths found: " << query_new.paths.size() << "\n";
    std::cout << "  Response: " << query_new.response << "\n";
    // Multi-hop over newly learned fact + old fact: vento -> esfria -> água -> apaga -> fogo!
    assert(!query_new.paths.empty());
    assert(query_new.paths[0].steps.size() >= 2);

    fs::remove_all(model_dir);
    std::cout << "[PASS] test_end_to_end_cognitive_pipeline\n";
}

int main() {
    test_end_to_end_cognitive_pipeline();
    return 0;
}
