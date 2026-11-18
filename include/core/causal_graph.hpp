#pragma once

#include "core/vsa.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

enum class RelationType : uint8_t {
    CAUSAL    = 0,
    INHIBITS  = 1,
    TAXONOMIC = 2,
    PROPERTY  = 3,
    ACTION    = 4
};

enum class GramTense : uint8_t {
    PRES = 0,
    PAST = 1,
    FUT  = 2,
    INF  = 3,
    SUBJ = 4
};

enum class GramGender : uint8_t {
    MASC = 0,
    FEM  = 1,
    NEUT = 2
};

enum class GramNumber : uint8_t {
    SING = 0,
    PLUR = 1
};

struct GraphNode {
    uint32_t id = 0;
    std::string name;
    uint32_t frequency = 1;
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::SING;
};

struct GraphEdge {
    uint32_t src = 0;
    uint32_t rel = 0; // index into relation_names_
    uint32_t dst = 0;
    RelationType type = RelationType::CAUSAL;
    uint32_t frequency = 1;
    float confidence = 0.8f;
    bool user_taught = false;
    bool negated = false;
    GramTense tense = GramTense::PRES;
    char modality[16]{}; // e.g. "pode", "nunca", "can"
};

struct CausalStep {
    std::string src;
    std::string rel;
    std::string dst;
    RelationType type = RelationType::CAUSAL;
    float confidence = 0.8f;
    bool negated = false;
    GramTense tense = GramTense::PRES;
    std::string modality;
    GramGender src_gender = GramGender::MASC;
    GramNumber src_number = GramNumber::SING;
    GramGender dst_gender = GramGender::MASC;
    GramNumber dst_number = GramNumber::SING;
    uint32_t edge_id = UINT32_MAX; // Snapshot-local provenance; never serialized as GraphEdge.
};

struct CausalPath {
    std::vector<CausalStep> steps;
    float score = 0.0f;
};

class CausalGraph {
public:
    uint32_t get_or_add_node(const std::string& name,
                             GramGender gender = GramGender::MASC,
                             GramNumber number = GramNumber::SING);
    uint32_t find_node(const std::string& name) const; // UINT32_MAX if not found
    uint32_t get_or_add_relation(const std::string& rel);

    void add_fact(const std::string& subj,
                  const std::string& rel,
                  const std::string& obj,
                  RelationType type = RelationType::CAUSAL,
                  float confidence = 0.8f,
                  bool user_taught = false,
                  bool negated = false,
                  GramTense tense = GramTense::PRES,
                  const std::string& modality = "",
                  GramGender subj_gender = GramGender::MASC,
                  GramNumber subj_number = GramNumber::SING,
                  GramGender obj_gender = GramGender::MASC,
                  GramNumber obj_number = GramNumber::SING);

    /// Multi-hop beam search / spreading activation from seed concepts.
    std::vector<CausalPath> find_paths(const std::vector<std::string>& seed_concepts,
                                       size_t max_hops = 3,
                                       size_t max_results = 8,
                                       const Codebook* codebook = nullptr) const;

    /// Finds direct facts about a concept (outgoing and incoming).
    std::vector<CausalStep> query_direct(const std::string& concept_name, size_t limit = 8) const;

    /// Sleep consolidation: merge duplicate edges, prune low-confidence noise,
    /// and promote shared properties into taxonomic hierarchies.
    size_t consolidate(uint32_t min_freq = 1, float min_conf = 0.2f);

    size_t num_nodes() const noexcept { return nodes_.size(); }
    size_t num_edges() const noexcept { return edges_.size(); }
    size_t memory_bytes() const noexcept;

    const std::vector<GraphNode>& nodes() const { return nodes_; }
    const std::vector<std::string>& relations() const { return relation_names_; }
    const std::vector<GraphEdge>& edges() const { return edges_; }

    void load_raw(std::vector<GraphNode> nodes,
                  std::vector<std::string> relation_names,
                  std::vector<GraphEdge> edges);

private:
    void rebuild_adjacency();

    std::unordered_map<std::string, uint32_t> node_map_;
    std::vector<GraphNode> nodes_;

    std::unordered_map<std::string, uint32_t> rel_map_;
    std::vector<std::string> relation_names_;

    std::vector<GraphEdge> edges_;
    std::vector<std::vector<uint32_t>> out_adj_;
    std::vector<std::vector<uint32_t>> in_adj_;
};

} // namespace cognicore
