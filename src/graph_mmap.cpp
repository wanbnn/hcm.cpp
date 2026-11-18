#include "core/graph_mmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <unordered_set>

namespace cognicore {
namespace {

struct Header {
    uint32_t magic = 0x474D4D31; // GMM1
    uint32_t version = 1;
    uint32_t nodes = 0, edges = 0, relations = 0;
    uint32_t strings_size = 0, index_count = 0, flags = 0;
};
struct NodeRec {
    uint32_t id = 0, name_off = 0, name_len = 0, frequency = 0;
    uint8_t gender = 0, number = 0;
    uint16_t pad = 0;
};
struct RelRec { uint32_t off = 0, len = 0; };
struct IndexRec { uint64_t hash = 0; uint32_t id = 0, pad = 0; };

uint64_t fnv1a(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char ch : text) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

struct MappedCausalGraph::Record { GraphEdge edge; };

MappedCausalGraph::~MappedCausalGraph() {
    if (data_) ::munmap(data_, size_);
    if (fd_ >= 0) ::close(fd_);
}

const MappedCausalGraph::Record* MappedCausalGraph::edge(uint32_t index) const {
    if (index >= edge_count_ || !edges_) return nullptr;
    return reinterpret_cast<const Record*>(edges_ + size_t(index) * sizeof(Record));
}

uint32_t MappedCausalGraph::edge_src(const Record& r) const { return r.edge.src; }
uint32_t MappedCausalGraph::edge_dst(const Record& r) const { return r.edge.dst; }
uint32_t MappedCausalGraph::edge_rel(const Record& r) const { return r.edge.rel; }
uint8_t MappedCausalGraph::edge_type(const Record& r) const { return static_cast<uint8_t>(r.edge.type); }
bool MappedCausalGraph::edge_negated(const Record& r) const { return r.edge.negated; }
std::string MappedCausalGraph::edge_modality(const Record& r) const {
    return std::string(r.edge.modality);
}
float MappedCausalGraph::edge_confidence(const Record& r) const { return r.edge.confidence; }
uint32_t MappedCausalGraph::edge_frequency(const Record& r) const { return r.edge.frequency; }

size_t MappedCausalGraph::num_nodes() const noexcept { return node_count_; }
size_t MappedCausalGraph::num_edges() const noexcept { return edge_count_; }

std::string MappedCausalGraph::node_name(uint32_t id) const {
    if (id >= node_count_ || !nodes_ || !strings_) return {};
    NodeRec rec;
    std::memcpy(&rec, nodes_ + size_t(id) * sizeof(NodeRec), sizeof(rec));
    if (rec.name_off + rec.name_len > string_bytes_) return {};
    return std::string(strings_ + rec.name_off, rec.name_len);
}

uint32_t MappedCausalGraph::find_node(const std::string& name) const {
    if (!index_ || !strings_ || !index_count_) return UINT32_MAX;
    const uint64_t hash = fnv1a(name);
    auto entry = [&](uint32_t at) {
        IndexRec rec;
        std::memcpy(&rec, index_ + size_t(at) * sizeof(IndexRec), sizeof(rec));
        return rec;
    };
    uint32_t low = 0, high = index_count_;
    while (low < high) {
        const uint32_t mid = low + (high - low) / 2;
        if (entry(mid).hash < hash) low = mid + 1; else high = mid;
    }
    for (uint32_t at = low; at < index_count_; ++at) {
        const auto rec = entry(at);
        if (rec.hash != hash) break;
        if (node_name(rec.id) == name) return rec.id;
    }
    return UINT32_MAX;
}

bool MappedCausalGraph::write(const std::string& path, const CausalGraph& graph) {
    std::string strings;
    std::vector<NodeRec> nodes;
    nodes.reserve(graph.nodes().size());
    for (const auto& node : graph.nodes()) {
        NodeRec rec;
        rec.id = node.id;
        rec.name_off = static_cast<uint32_t>(strings.size());
        rec.name_len = static_cast<uint32_t>(node.name.size());
        rec.frequency = node.frequency;
        rec.gender = static_cast<uint8_t>(node.gender);
        rec.number = static_cast<uint8_t>(node.number);
        strings += node.name;
        nodes.push_back(rec);
    }
    std::vector<RelRec> relations;
    relations.reserve(graph.relations().size());
    for (const auto& relation : graph.relations()) {
        RelRec rec;
        rec.off = static_cast<uint32_t>(strings.size());
        rec.len = static_cast<uint32_t>(relation.size());
        strings += relation;
        relations.push_back(rec);
    }
    std::vector<IndexRec> index;
    index.reserve(nodes.size());
    for (const auto& node : graph.nodes()) index.push_back({fnv1a(node.name), node.id, 0});
    std::sort(index.begin(), index.end(), [&](const IndexRec& a, const IndexRec& b) {
        if (a.hash != b.hash) return a.hash < b.hash;
        return graph.nodes()[a.id].name < graph.nodes()[b.id].name;
    });

    Header header;
    header.nodes = static_cast<uint32_t>(nodes.size());
    header.edges = static_cast<uint32_t>(graph.edges().size());
    header.relations = static_cast<uint32_t>(relations.size());
    header.strings_size = static_cast<uint32_t>(strings.size());
    header.index_count = static_cast<uint32_t>(index.size());

    const auto temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!nodes.empty()) out.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(NodeRec));
    if (!graph.edges().empty()) out.write(reinterpret_cast<const char*>(graph.edges().data()), graph.edges().size() * sizeof(GraphEdge));
    if (!relations.empty()) out.write(reinterpret_cast<const char*>(relations.data()), relations.size() * sizeof(RelRec));
    if (!index.empty()) out.write(reinterpret_cast<const char*>(index.data()), index.size() * sizeof(IndexRec));
    if (!strings.empty()) out.write(strings.data(), static_cast<std::streamsize>(strings.size()));
    out.close();
    if (!out) return false;
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    return !ec;
}

std::shared_ptr<MappedCausalGraph> MappedCausalGraph::open(const std::string& path) {
    auto mapped = std::shared_ptr<MappedCausalGraph>(new MappedCausalGraph());
    mapped->fd_ = ::open(path.c_str(), O_RDONLY);
    if (mapped->fd_ < 0) return nullptr;
    struct stat st {};
    if (::fstat(mapped->fd_, &st) != 0 || st.st_size < off_t(sizeof(Header))) {
        ::close(mapped->fd_); mapped->fd_ = -1; return nullptr;
    }
    mapped->size_ = static_cast<size_t>(st.st_size);
    mapped->data_ = ::mmap(nullptr, mapped->size_, PROT_READ, MAP_PRIVATE, mapped->fd_, 0);
    if (mapped->data_ == MAP_FAILED) {
        mapped->data_ = nullptr; ::close(mapped->fd_); mapped->fd_ = -1; return nullptr;
    }
    Header header;
    std::memcpy(&header, mapped->data_, sizeof(header));
    if (header.magic != 0x474D4D31 || header.version != 1 || header.nodes > 100000000 ||
        header.edges > 100000000 || header.relations > 10000000 || header.index_count > header.nodes) {
        return nullptr;
    }
    const size_t expected = sizeof(Header) + size_t(header.nodes) * sizeof(NodeRec) +
        size_t(header.edges) * sizeof(GraphEdge) + size_t(header.relations) * sizeof(RelRec) +
        size_t(header.index_count) * sizeof(IndexRec) + size_t(header.strings_size);
    if (expected != mapped->size_) return nullptr;
    auto* base = static_cast<uint8_t*>(mapped->data_) + sizeof(Header);
    mapped->nodes_ = base; base += size_t(header.nodes) * sizeof(NodeRec);
    mapped->edges_ = base; base += size_t(header.edges) * sizeof(GraphEdge);
    const auto* rel_base = base; base += size_t(header.relations) * sizeof(RelRec);
    mapped->index_ = base; base += size_t(header.index_count) * sizeof(IndexRec);
    mapped->strings_ = reinterpret_cast<const char*>(base);
    mapped->node_count_ = header.nodes; mapped->edge_count_ = header.edges;
    mapped->relation_count_ = header.relations; mapped->index_count_ = header.index_count;
    mapped->string_bytes_ = header.strings_size;
    mapped->relations_.resize(header.relations);
    for (uint32_t i = 0; i < header.relations; ++i) {
        RelRec rec;
        std::memcpy(&rec, rel_base + size_t(i) * sizeof(RelRec), sizeof(rec));
        if (rec.off + rec.len > mapped->string_bytes_) return nullptr;
        mapped->relations_[i] = std::string(mapped->strings_ + rec.off, rec.len);
    }
    return mapped;
}

std::vector<CausalStep> MappedCausalGraph::query_direct(const std::string& concept_name, size_t limit) const {
    std::vector<CausalStep> steps;
    const uint32_t id = find_node(concept_name);
    if (id == UINT32_MAX) return steps;
    for (uint32_t i = 0; i < edge_count_; ++i) {
        const auto* record = edge(i);
        if (!record || (edge_src(*record) != id && edge_dst(*record) != id)) continue;
        CausalStep step;
        step.src = node_name(edge_src(*record));
        step.dst = node_name(edge_dst(*record));
        const uint32_t rel = edge_rel(*record);
        step.rel = rel < relations_.size() ? relations_[rel] : "";
        step.type = static_cast<RelationType>(edge_type(*record));
        step.confidence = edge_confidence(*record);
        step.negated = edge_negated(*record);
        step.tense = record->edge.tense;
        step.modality = edge_modality(*record);
        step.edge_id = i;
        steps.push_back(std::move(step));
    }
    std::sort(steps.begin(), steps.end(), [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    if (steps.size() > limit) steps.resize(limit);
    return steps;
}

std::vector<CausalPath> MappedCausalGraph::find_paths(const std::vector<std::string>& seed_concepts,
                                                      size_t max_hops, size_t max_results) const {
    std::unordered_set<uint32_t> seeds;
    for (const auto& name : seed_concepts) {
        const auto id = find_node(name);
        if (id != UINT32_MAX) seeds.insert(id);
    }
    if (seeds.empty()) return {};
    struct Partial { std::vector<uint32_t> nodes, edges; float score = 1.0f; };
    std::vector<CausalPath> results;
    std::queue<Partial> queue;
    for (auto seed : seeds) queue.push(Partial{{seed}, {}, 1.0f});
    auto make_step = [&](uint32_t eid) {
        const auto* record = edge(eid);
        CausalStep step;
        if (!record) return step;
        step.src = node_name(edge_src(*record));
        step.dst = node_name(edge_dst(*record));
        const uint32_t rel = edge_rel(*record);
        step.rel = rel < relations_.size() ? relations_[rel] : "";
        step.type = static_cast<RelationType>(edge_type(*record));
        step.confidence = edge_confidence(*record);
        step.negated = edge_negated(*record);
        step.tense = record->edge.tense;
        step.modality = edge_modality(*record);
        step.edge_id = eid;
        return step;
    };
    while (!queue.empty()) {
        Partial current = std::move(queue.front()); queue.pop();
        const uint32_t tail = current.nodes.back();
        if (!current.edges.empty()) {
            CausalPath path;
            path.score = current.score;
            for (auto eid : current.edges) path.steps.push_back(make_step(eid));
            if (seeds.count(current.nodes.front()) && seeds.count(tail) && current.nodes.front() != tail)
                path.score += 2.5f;
            if (path.steps.size() >= 2) path.score += 0.4f;
            results.push_back(std::move(path));
        }
        if (current.edges.size() >= max_hops) continue;
        if (!current.edges.empty() && edge_negated(*edge(current.edges.back()))) continue;
        std::vector<uint32_t> candidates;
        for (uint32_t i = 0; i < edge_count_; ++i) {
            const auto* record = edge(i);
            if (record && edge_src(*record) == tail) candidates.push_back(i);
        }
        std::sort(candidates.begin(), candidates.end(), [&](uint32_t a, uint32_t b) {
            return edge_confidence(*edge(a)) > edge_confidence(*edge(b));
        });
        if (candidates.size() > 6) candidates.resize(6);
        for (auto eid : candidates) {
            const auto* record = edge(eid);
            const uint32_t next = edge_dst(*record);
            if (std::find(current.nodes.begin(), current.nodes.end(), next) != current.nodes.end()) continue;
            Partial next_path = current;
            next_path.nodes.push_back(next);
            next_path.edges.push_back(eid);
            next_path.score *= edge_confidence(*record) *
                (1.0f + 0.05f * std::log1p(static_cast<float>(edge_frequency(*record))));
            queue.push(std::move(next_path));
        }
    }
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    if (results.size() > max_results) results.resize(max_results);
    return results;
}

} // namespace cognicore
