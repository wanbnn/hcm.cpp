#pragma once

#include "core/causal_graph.hpp"
#include "core/code_ast.hpp"
#include "core/realizer.hpp"
#include "core/sdm.hpp"
#include "core/text_parser.hpp"
#include "core/tsetlin.hpp"
#include "core/vsa.hpp"

#include <string>
#include <vector>

namespace cognicore {

struct ModelBundle {
    Codebook codebook;
    KanervaSDM sdm;
    TsetlinMachine tsetlin;
    SparseTsetlinModel sparse_tsetlin;
    bool has_sparse_tsetlin = false;
    CausalGraph graph;
    CodeGraph code_graph;
    std::vector<std::string> realizer_pairs;
    std::vector<std::pair<std::string, std::string>> episodes;
    ConstructionGrammarRealizer realizer;
    TextParser parser;

    /// Builds the [VSA(4096) + context_bits(64)] packed uint64_t feature vector for Tsetlin.
    std::vector<uint64_t> encode_dialogue_features(const std::string& user_text) const;

    /// Runs the full inference pipeline (CPU-only):
    /// VSA + SDM + Tsetlin (sparse if available) + CausalGraph + Realizer.
    struct InferenceTrace {
        std::vector<std::string> activated_concepts;
        DialogIntent intent = DialogIntent::ANSWER_FACT;
        std::vector<CausalPath> paths;
        std::string response;
    };
    InferenceTrace infer_and_respond(const std::string& user_text, bool online_learn = true,
                                     const ISurfaceRealizer* surface = nullptr);

    /// Saves all model components into `dir_path` in compact binary format.
    bool save(const std::string& dir_path) const;

    /// Loads all model components from `dir_path` (using mmap when `use_mmap` is true).
    bool load(const std::string& dir_path, bool use_mmap = true);

    /// Total size on disk in bytes for `dir_path`.
    static size_t disk_size_bytes(const std::string& dir_path);

    /// Estimated RAM footprint in bytes.
    size_t estimated_ram_bytes() const noexcept;
};

} // namespace cognicore
