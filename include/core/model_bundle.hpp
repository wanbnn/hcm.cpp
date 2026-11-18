#pragma once

#include "core/causal_graph.hpp"
#include "core/code_patterns.hpp"
#include "core/coherence.hpp"
#include "core/context_memory.hpp"
#include "core/episodic_memory.hpp"
#include "core/graph_mmap.hpp"
#include "core/language_rules.hpp"
#include "core/lexicon.hpp"
#include "core/code_ast.hpp"
#include "core/realizer.hpp"
#include "core/analogical_realizer.hpp"
#include "core/sdm.hpp"
#include "core/term_resolver.hpp"
#include "core/discourse_config.hpp"
#include "core/text_parser.hpp"
#include "core/tsetlin.hpp"
#include "core/vsa.hpp"

#include <string>
#include <vector>

namespace cognicore {

struct ModelBundle {
    ModelBundle();
    LanguageRules language_rules;
    Lexicon lexicon;
    CoherenceModel coherence;
    CodePatternLibrary code_patterns;
    Codebook codebook;
    KanervaSDM sdm;
    TsetlinMachine tsetlin;
    SparseTsetlinModel sparse_tsetlin;
    bool has_sparse_tsetlin = false;
    SparseTsetlinModel context_policy;
    bool has_context_policy = false;
    EpisodicMemory episodic_memory;
    ContextMemory context_memory;
    TermResolver terms;
    DiscourseConfig discourse;
    CausalGraph graph;
    std::shared_ptr<MappedCausalGraph> mapped_graph;
    CodeGraph code_graph;
    std::vector<std::string> realizer_pairs;
    std::vector<std::pair<std::string, std::string>> episodes;
    ConstructionGrammarRealizer realizer;
    AnalogicalFluencyRealizer fluent_realizer;
    TextParser parser;

    /// Builds the [VSA(4096) + context_bits(64)] packed uint64_t feature vector for Tsetlin.
    std::vector<uint64_t> encode_dialogue_features(const std::string& user_text) const;

    DialogIntent classify_dialogue(const std::string& user_text) const;

    /// Runs the full inference pipeline (CPU-only):
    /// VSA + SDM + Tsetlin (sparse if available) + CausalGraph + Realizer.
    struct InferenceTrace {
        std::vector<std::string> activated_concepts;
        DialogIntent intent = DialogIntent::ANSWER_FACT;
        std::vector<CausalPath> paths;
        std::vector<CausalPath> alternatives;
        std::vector<uint8_t> rhetorical_roles;
        std::string planning_evidence;
        SemanticPlan plan;
        std::vector<std::pair<std::string, double>> stage_ms;
        std::vector<int32_t> class_votes;
        std::vector<size_t> fired_clauses;
        std::vector<SparseClause> fired_literals;
        std::string policy_source, decision_override, code_evidence, router_evidence, fluency_evidence;
        std::vector<uint32_t> active_slots, slot_distances, edge_ids;
        std::vector<EpisodicRecall> recalled_episodes;
        std::string resolved_reference;
        std::string context_evidence;
        std::vector<std::string> carried_concepts;
        std::string response;
        std::string debug_report() const;
    };
    InferenceTrace infer_and_respond(const std::string& user_text, bool online_learn = false,
                                     const ISurfaceRealizer* surface = nullptr);

    /// Saves all model components into `dir_path` in compact binary format.
    bool save(const std::string& dir_path) const;

    /// Loads all model components from `dir_path` (using mmap when `use_mmap` is true).
    bool load(const std::string& dir_path, bool use_mmap = true);

    /// Reopens graph_mmap.bin after a save so read-only path queries use the
    /// mapped compact index. Call after `save` once the graph has changed.
    bool refresh_mapped_graph(const std::string& dir_path);

    /// Total size on disk in bytes for `dir_path`.
    static size_t disk_size_bytes(const std::string& dir_path);

    /// Estimated RAM footprint in bytes.
    size_t estimated_ram_bytes() const noexcept;
};

} // namespace cognicore
