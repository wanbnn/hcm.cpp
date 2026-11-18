#pragma once

#include "core/causal_graph.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cognicore {

/// Read-only causal graph backed directly by a memory-mapped compact index.
/// The textual model remains the source of truth for writes; this view serves
/// path/adjacency queries without rebuilding nodes/edges on the heap.
class MappedCausalGraph {
public:
    static bool write(const std::string& path, const CausalGraph& graph);
    static std::shared_ptr<MappedCausalGraph> open(const std::string& path);

    ~MappedCausalGraph();
    MappedCausalGraph(const MappedCausalGraph&) = delete;
    MappedCausalGraph& operator=(const MappedCausalGraph&) = delete;

    bool valid() const { return data_ != nullptr; }
    size_t num_nodes() const noexcept;
    size_t num_edges() const noexcept;
    size_t mapped_bytes() const noexcept { return size_; }

    uint32_t find_node(const std::string& name) const;
    std::string node_name(uint32_t id) const;
    const std::vector<std::string>& relations() const { return relations_; }

    std::vector<CausalStep> query_direct(const std::string& concept_name, size_t limit = 8) const;
    std::vector<CausalPath> find_paths(const std::vector<std::string>& seed_concepts,
                                       size_t max_hops = 3,
                                       size_t max_results = 8) const;

private:
    MappedCausalGraph() = default;
    struct Record;
    const Record* edge(uint32_t index) const;
    uint32_t edge_src(const Record& record) const;
    uint32_t edge_dst(const Record& record) const;
    uint32_t edge_rel(const Record& record) const;
    uint8_t edge_type(const Record& record) const;
    bool edge_negated(const Record& record) const;
    std::string edge_modality(const Record& record) const;
    float edge_confidence(const Record& record) const;
    uint32_t edge_frequency(const Record& record) const;

    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
    std::vector<std::string> relations_;
    const uint8_t* nodes_ = nullptr;
    const uint8_t* edges_ = nullptr;
    const uint8_t* index_ = nullptr;
    const char* strings_ = nullptr;
    uint32_t node_count_ = 0, edge_count_ = 0, relation_count_ = 0, index_count_ = 0;
    uint32_t string_bytes_ = 0;
};

} // namespace cognicore
