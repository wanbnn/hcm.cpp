#include "core/causal_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <unordered_set>

namespace cognicore {

uint32_t CausalGraph::get_or_add_node(const std::string& name,
                                      GramGender gender,
                                      GramNumber number) {
    auto it = node_map_.find(name);
    if (it != node_map_.end()) {
        nodes_[it->second].frequency += 1;
        return it->second;
    }
    const uint32_t id = static_cast<uint32_t>(nodes_.size());
    node_map_.emplace(name, id);
    nodes_.push_back(GraphNode{id, name, 1, gender, number});
    out_adj_.emplace_back();
    in_adj_.emplace_back();
    return id;
}

uint32_t CausalGraph::find_node(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it != node_map_.end()) return it->second;
    return UINT32_MAX;
}

uint32_t CausalGraph::get_or_add_relation(const std::string& rel) {
    auto it = rel_map_.find(rel);
    if (it != rel_map_.end()) return it->second;
    const uint32_t id = static_cast<uint32_t>(relation_names_.size());
    rel_map_.emplace(rel, id);
    relation_names_.push_back(rel);
    return id;
}

void CausalGraph::add_fact(const std::string& subj,
                           const std::string& rel,
                           const std::string& obj,
                           RelationType type,
                           float confidence,
                           bool user_taught,
                           bool negated,
                           GramTense tense,
                           const std::string& modality,
                           GramGender subj_gender,
                           GramNumber subj_number,
                           GramGender obj_gender,
                           GramNumber obj_number) {
    if (subj.empty() || obj.empty() || subj == obj) return;
    const uint32_t s = get_or_add_node(subj, subj_gender, subj_number);
    const uint32_t r = get_or_add_relation(rel);
    const uint32_t d = get_or_add_node(obj, obj_gender, obj_number);

    // Check if edge already exists
    for (uint32_t eid : out_adj_[s]) {
        auto& e = edges_[eid];
        if (e.dst == d && e.rel == r && e.type == type && e.negated == negated &&
            e.tense == tense && std::string(e.modality) == modality.substr(0, 15)) {
            e.frequency += 1;
            e.confidence = std::min(1.0f, std::max(e.confidence, confidence) + 0.05f);
            if (user_taught) e.user_taught = true;
            return;
        }
    }

    const uint32_t eid = static_cast<uint32_t>(edges_.size());
    GraphEdge edge{s, r, d, type, 1, std::clamp(confidence, 0.05f, 1.0f), user_taught, negated, tense, {}};
    if (!modality.empty()) {
        std::strncpy(edge.modality, modality.c_str(), sizeof(edge.modality) - 1);
    }
    edges_.push_back(edge);
    out_adj_[s].push_back(eid);
    in_adj_[d].push_back(eid);
}

std::vector<CausalStep> CausalGraph::query_direct(const std::string& concept_name, size_t limit) const {
    std::vector<CausalStep> steps;
    const uint32_t nid = find_node(concept_name);
    if (nid == UINT32_MAX) return steps;

    auto make_step = [&](const GraphEdge& e) {
        return CausalStep{
            nodes_[e.src].name,
            relation_names_[e.rel],
            nodes_[e.dst].name,
            e.type,
            e.confidence,
            e.negated,
            e.tense,
            std::string(e.modality),
            nodes_[e.src].gender,
            nodes_[e.src].number,
            nodes_[e.dst].gender,
            nodes_[e.dst].number
        };
    };

    for (uint32_t eid : out_adj_[nid]) {
        steps.push_back(make_step(edges_[eid]));
    }
    for (uint32_t eid : in_adj_[nid]) {
        steps.push_back(make_step(edges_[eid]));
    }

    std::sort(steps.begin(), steps.end(),
              [](const CausalStep& a, const CausalStep& b) { return a.confidence > b.confidence; });
    if (steps.size() > limit) steps.resize(limit);
    return steps;
}

std::vector<CausalPath> CausalGraph::find_paths(const std::vector<std::string>& seed_concepts,
                                                size_t max_hops,
                                                size_t max_results,
                                                const Codebook* codebook) const {
    std::unordered_set<uint32_t> seed_ids;
    for (const auto& s : seed_concepts) {
        const uint32_t id = find_node(s);
        if (id != UINT32_MAX) {
            seed_ids.insert(id);
        } else if (codebook != nullptr) {
            // Try VSA nearest neighbors in graph
            auto hv = HyperVector::from_token(s);
            auto near = codebook->query_nearest(hv, 3, true);
            for (const auto& [cand, sim] : near) {
                if (sim > 0.25) {
                    const uint32_t cid = find_node(cand);
                    if (cid != UINT32_MAX) seed_ids.insert(cid);
                }
            }
        }
    }

    if (seed_ids.empty()) return {};

    struct PartialPath {
        std::vector<uint32_t> node_seq;
        std::vector<uint32_t> edge_seq;
        float score = 1.0f;
    };

    std::vector<CausalPath> results;
    std::queue<PartialPath> q;
    for (uint32_t sid : seed_ids) {
        q.push(PartialPath{{sid}, {}, 1.0f});
    }

    while (!q.empty()) {
        PartialPath cur = std::move(q.front());
        q.pop();

        const uint32_t tail = cur.node_seq.back();
        if (!cur.edge_seq.empty()) {
            CausalPath cp;
            cp.score = cur.score;
            for (uint32_t eid : cur.edge_seq) {
                const auto& e = edges_[eid];
                cp.steps.push_back(CausalStep{
                    nodes_[e.src].name,
                    relation_names_[e.rel],
                    nodes_[e.dst].name,
                    e.type,
                    e.confidence,
                    e.negated,
                    e.tense,
                    std::string(e.modality),
                    nodes_[e.src].gender,
                    nodes_[e.src].number,
                    nodes_[e.dst].gender,
                    nodes_[e.dst].number
                });
            }
            // Bonus if path connects two distinct seed concepts
            if (seed_ids.count(cur.node_seq.front()) && seed_ids.count(tail) &&
                cur.node_seq.front() != tail) {
                cp.score += 2.5f;
            }
            // Slight bonus for multi-hop inference
            if (cp.steps.size() >= 2) {
                cp.score += 0.4f;
            }
            results.push_back(std::move(cp));
        }

        if (cur.edge_seq.size() >= max_hops) continue;
        // A denied relation may be reported, but cannot establish its target
        // as a premise for a downstream causal conclusion.
        if (!cur.edge_seq.empty() && edges_[cur.edge_seq.back()].negated) continue;

        // Expand outgoing edges (beam limit: top 6 edges per node)
        std::vector<uint32_t> cand_edges = out_adj_[tail];
        std::sort(cand_edges.begin(), cand_edges.end(), [&](uint32_t a, uint32_t b) {
            return edges_[a].confidence > edges_[b].confidence;
        });
        if (cand_edges.size() > 6) cand_edges.resize(6);

        for (uint32_t eid : cand_edges) {
            const uint32_t next_node = edges_[eid].dst;
            if (std::find(cur.node_seq.begin(), cur.node_seq.end(), next_node) != cur.node_seq.end()) {
                continue; // avoid cycles
            }
            PartialPath nxt = cur;
            nxt.node_seq.push_back(next_node);
            nxt.edge_seq.push_back(eid);
            nxt.score *= edges_[eid].confidence * (1.0f + 0.05f * std::log1p(static_cast<float>(edges_[eid].frequency)));
            q.push(std::move(nxt));
        }
    }

    // If no outgoing paths from seeds, check incoming edges as fallback
    if (results.empty()) {
        for (uint32_t sid : seed_ids) {
            for (uint32_t eid : in_adj_[sid]) {
                const auto& e = edges_[eid];
                CausalPath cp;
                cp.score = e.confidence;
                cp.steps.push_back(CausalStep{
                    nodes_[e.src].name,
                    relation_names_[e.rel],
                    nodes_[e.dst].name,
                    e.type,
                    e.confidence,
                    e.negated,
                    e.tense,
                    std::string(e.modality),
                    nodes_[e.src].gender,
                    nodes_[e.src].number,
                    nodes_[e.dst].gender,
                    nodes_[e.dst].number
                });
                results.push_back(std::move(cp));
            }
        }
    }

    std::sort(results.begin(), results.end(),
              [](const CausalPath& a, const CausalPath& b) { return a.score > b.score; });

    if (results.size() > max_results) {
        results.resize(max_results);
    }
    return results;
}

size_t CausalGraph::consolidate(uint32_t min_freq, float min_conf) {
    const size_t before = edges_.size();

    // Merge duplicates & filter noise
    struct EdgeKey {
        uint32_t s, r, d;
        uint8_t t;
        bool negated;
        GramTense tense;
        std::string modality;
        bool operator==(const EdgeKey& o) const noexcept {
            return s == o.s && r == o.r && d == o.d && t == o.t &&
                   negated == o.negated && tense == o.tense && modality == o.modality;
        }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& k) const noexcept {
            uint64_t h = (uint64_t(k.s) << 32) ^ (uint64_t(k.r) << 16) ^ uint64_t(k.d) ^ (uint64_t(k.t) << 48);
            return std::hash<uint64_t>{}(h) ^ std::hash<std::string>{}(k.modality) ^
                   (size_t(k.negated) << 8) ^ (size_t(k.tense) << 10);
        }
    };

    std::unordered_map<EdgeKey, GraphEdge, EdgeKeyHash> merged;
    for (const auto& e : edges_) {
        EdgeKey k{e.src, e.rel, e.dst, static_cast<uint8_t>(e.type), e.negated, e.tense, std::string(e.modality)};
        auto it = merged.find(k);
        if (it == merged.end()) {
            merged.emplace(k, e);
        } else {
            it->second.frequency += e.frequency;
            it->second.confidence = std::min(1.0f, std::max(it->second.confidence, e.confidence) + 0.05f);
            it->second.user_taught = it->second.user_taught || e.user_taught;
        }
    }

    edges_.clear();
    for (const auto& [k, e] : merged) {
        if (!e.user_taught && (e.frequency < min_freq || e.confidence < min_conf)) {
            continue;
        }
        edges_.push_back(e);
    }

    rebuild_adjacency();
    return before - edges_.size();
}

void CausalGraph::rebuild_adjacency() {
    out_adj_.assign(nodes_.size(), {});
    in_adj_.assign(nodes_.size(), {});
    for (uint32_t i = 0; i < edges_.size(); ++i) {
        out_adj_[edges_[i].src].push_back(i);
        in_adj_[edges_[i].dst].push_back(i);
    }
}

size_t CausalGraph::memory_bytes() const noexcept {
    size_t bytes = nodes_.size() * sizeof(GraphNode) + edges_.size() * sizeof(GraphEdge);
    for (const auto& n : nodes_) bytes += n.name.capacity();
    for (const auto& r : relation_names_) bytes += r.capacity();
    return bytes;
}

void CausalGraph::load_raw(std::vector<GraphNode> nodes,
                           std::vector<std::string> relation_names,
                           std::vector<GraphEdge> edges) {
    nodes_ = std::move(nodes);
    relation_names_ = std::move(relation_names);
    edges_ = std::move(edges);

    node_map_.clear();
    for (size_t i = 0; i < nodes_.size(); ++i) {
        node_map_[nodes_[i].name] = static_cast<uint32_t>(i);
    }
    rel_map_.clear();
    for (size_t i = 0; i < relation_names_.size(); ++i) {
        rel_map_[relation_names_[i]] = static_cast<uint32_t>(i);
    }
    rebuild_adjacency();
}

} // namespace cognicore
