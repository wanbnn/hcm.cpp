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

struct GraphNode {
    uint32_t id = 0;
    std::string name;
    uint32_t frequency = 1;
};

struct GraphEdge {
    uint32_t src = 0;
    uint32_t rel = 0; // index into relation_names_
    uint32_t dst = 0;
    RelationType type = RelationType::CAUSAL;
    uint32_t frequency = 1;
    float confidence = 0.8f;
    bool user_taught = false;
};

struct CausalStep {
    std::string src;
    std::string rel;
    std::string dst;
    RelationType type = RelationType::CAUSAL;
    float confidence = 0.8f;
};

struct CausalPath {
    std::vector<CausalStep> steps;
    float score = 0.0f;
};

class CausalGraph {
public:
    uint32_t get_or_add_node(const std::string& name);
    uint32_t find_node(const std::string& name) const; // UINT32_MAX if not found
    uint32_t get_or_add_relation(const std::string& rel);

    void add_fact(const std::string& subj,
                  const std::string& rel,
                  const std::string& obj,
                  RelationType type = RelationType::CAUSAL,
                  float confidence = 0.8f,
                  bool user_taught = false);

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
