#include "core/model_bundle.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace cognicore {

namespace {

class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& path) : ofs_(path, std::ios::binary) {}
    bool ok() const { return static_cast<bool>(ofs_); }

    template <typename T>
    void write_pod(const T& val) {
        ofs_.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    void write_bytes(const void* data, size_t n) {
        if (n > 0) ofs_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    }

    void write_string(const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        write_pod(len);
        write_bytes(s.data(), len);
    }

private:
    std::ofstream ofs_;
};

class MappedReader {
public:
    MappedReader(const std::string& path, bool use_mmap) : use_mmap_(use_mmap) {
        if (use_mmap_) {
            fd_ = ::open(path.c_str(), O_RDONLY);
            if (fd_ < 0) return;
            struct stat st{};
            if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
                ::close(fd_);
                fd_ = -1;
                return;
            }
            size_ = static_cast<size_t>(st.st_size);
            void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (p == MAP_FAILED) {
                ::close(fd_);
                fd_ = -1;
                return;
            }
            data_ = static_cast<const uint8_t*>(p);
            valid_ = true;
        } else {
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (!ifs) return;
            size_ = static_cast<size_t>(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            buf_.resize(size_);
            if (size_ > 0 && !ifs.read(reinterpret_cast<char*>(buf_.data()), static_cast<std::streamsize>(size_))) {
                return;
            }
            data_ = buf_.data();
            valid_ = true;
        }
    }

    ~MappedReader() {
        if (use_mmap_ && data_ != nullptr) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    bool valid() const { return valid_; }
    bool eof() const { return offset_ >= size_; }

    template <typename T>
    bool read_pod(T& out) {
        if (offset_ + sizeof(T) > size_) return false;
        std::memcpy(&out, data_ + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool read_bytes(void* dst, size_t n) {
        if (offset_ + n > size_) return false;
        if (n > 0) std::memcpy(dst, data_ + offset_, n);
        offset_ += n;
        return true;
    }

    bool read_string(std::string& out) {
        uint32_t len = 0;
        if (!read_pod(len)) return false;
        if (offset_ + len > size_) return false;
        out.assign(reinterpret_cast<const char*>(data_ + offset_), len);
        offset_ += len;
        return true;
    }

private:
    bool use_mmap_ = true;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
    bool valid_ = false;
    std::vector<uint8_t> buf_;
};

constexpr uint32_t MAGIC_CB = 0x434F4743; // COGC
constexpr uint32_t MAGIC_SDM = 0x434F4753; // COGS
constexpr uint32_t MAGIC_TM = 0x434F4754; // COGT
constexpr uint32_t MAGIC_GR = 0x434F4747; // COGG
constexpr uint32_t MAGIC_RL = 0x434F4752; // COGR

} // namespace

std::vector<uint64_t> ModelBundle::encode_dialogue_features(const std::string& user_text) const {
    const size_t n_feat = has_sparse_tsetlin ? sparse_tsetlin.num_features : tsetlin.config().num_features;
    const size_t n_words = (n_feat + 63) / 64;
    std::vector<uint64_t> feats(n_words, 0ULL);

    auto toks = parser.tokenize(user_text);
    HyperVector hv = codebook.encode_bag(toks, true);
    const size_t copy_words = std::min<size_t>(VSA_WORDS, n_words);
    for (size_t w = 0; w < copy_words; ++w) {
        feats[w] = hv.words[w];
    }

    // Add 64 context/graph state bits if num_features > VSA_BITS
    if (n_feat > VSA_BITS && n_words > VSA_WORDS) {
        uint64_t ctx_bits = 0;
        if (user_text.find('?') != std::string::npos) ctx_bits |= (1ULL << 0);
        for (const auto& t : toks) {
            if (t == "por" || t == "porque" || t == "causa" || t == "why") ctx_bits |= (1ULL << 1);
            if (t == "o" || t == "que" || t == "qual" || t == "what") ctx_bits |= (1ULL << 2);
            if (t == "compare" || t == "diferenca" || t == "diferença" || t == "igual") ctx_bits |= (1ULL << 3);
            if (t == "se" || t == "hipotese" || t == "hipótese" || t == "if") ctx_bits |= (1ULL << 4);
            if (t == "ola" || t == "olá" || t == "oi" || t == "hello" || t == "hi") ctx_bits |= (1ULL << 5);
            if (graph.find_node(t) != UINT32_MAX) ctx_bits |= (1ULL << 6);
        }
        feats[VSA_WORDS] = ctx_bits;
    }
    return feats;
}

ModelBundle::InferenceTrace ModelBundle::infer_and_respond(const std::string& user_text, bool online_learn) {
    InferenceTrace trace;
    auto all_toks = parser.tokenize(user_text);
    auto content_toks = parser.extract_content_tokens(user_text);

    // Online one-shot learning if user states a declarative fact (no '?')
    const bool is_question = (user_text.find('?') != std::string::npos);
    bool learned_fact = false;
    if (online_learn && !is_question) {
        auto parsed = parser.parse_text(user_text);
        if (!parsed.triples.empty()) {
            for (const auto& tr : parsed.triples) {
                graph.add_fact(tr.subject, tr.relation, tr.object, tr.type, tr.confidence, true);
                codebook.get_or_add(tr.subject);
                codebook.get_or_add(tr.object);
                HyperVector addr = HyperVector::from_token(tr.subject);
                HyperVector data = HyperVector::from_token(tr.object);
                sdm.write(addr, data);
                learned_fact = true;
            }
            for (const auto& cf : parsed.frames) {
                realizer.add_frame(cf);
            }
        }
    }

    // 1. VSA + SDM concept activation
    for (const auto& t : content_toks) {
        if (graph.find_node(t) != UINT32_MAX) {
            trace.activated_concepts.push_back(t);
        }
    }
    // Also check multi-word phrases in graph
    for (size_t i = 0; i + 1 < all_toks.size(); ++i) {
        std::string bi = all_toks[i] + " " + all_toks[i + 1];
        if (graph.find_node(bi) != UINT32_MAX) {
            trace.activated_concepts.push_back(bi);
        }
    }

    // Query SDM associative memory for related concepts
    if (!content_toks.empty()) {
        HyperVector q_hv = codebook.encode_bag(content_toks, true);
        HyperVector recalled = sdm.read(q_hv);
        auto near = codebook.query_nearest(recalled, 3, true);
        for (const auto& [cand, sim] : near) {
            if (sim > 0.20 && graph.find_node(cand) != UINT32_MAX) {
                if (std::find(trace.activated_concepts.begin(), trace.activated_concepts.end(), cand) ==
                    trace.activated_concepts.end()) {
                    trace.activated_concepts.push_back(cand);
                }
            }
        }
    }

    // 2. Tsetlin Machine dialog intent
    if (learned_fact) {
        trace.intent = DialogIntent::AGREE;
    } else {
        auto feats = encode_dialogue_features(user_text);
        size_t pred = has_sparse_tsetlin
            ? sparse_tsetlin.predict(feats.data())
            : tsetlin.predict(feats.data());
        trace.intent = static_cast<DialogIntent>(pred);
    }

    // 3. Causal Graph multi-hop reasoning
    if (!trace.activated_concepts.empty()) {
        trace.paths = graph.find_paths(trace.activated_concepts, 3, 5, &codebook);
    }

    // 4. Surface Realizer compositional synthesis
    SemanticPlan plan;
    plan.intent = trace.intent;
    plan.paths = trace.paths;
    plan.query_concepts = trace.activated_concepts.empty() ? content_toks : trace.activated_concepts;
    trace.response = realizer.realize(plan);

    return trace;
}

bool ModelBundle::save(const std::string& dir_path) const {
    std::error_code ec;
    fs::create_directories(dir_path, ec);
    if (ec) return false;

    // 1. Codebook
    {
        BinaryWriter w(dir_path + "/codebook.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_CB);
        const uint32_t n = static_cast<uint32_t>(codebook.size());
        w.write_pod(n);
        for (const auto& tok : codebook.tokens()) w.write_string(tok);
        if (n > 0) {
            w.write_bytes(codebook.base_vectors().data(), n * sizeof(HyperVector));
            w.write_bytes(codebook.semantic_vectors().data(), n * sizeof(HyperVector));
        }
    }

    // 2. SDM
    {
        BinaryWriter w(dir_path + "/sdm.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_SDM);
        w.write_pod(sdm.config());
        uint8_t bin = sdm.is_binarized() ? 1 : 0;
        w.write_pod(bin);
        uint64_t wc = sdm.write_count();
        w.write_pod(wc);
        const size_t m = sdm.num_locations();
        w.write_bytes(sdm.addresses().data(), m * sizeof(HyperVector));
        w.write_bytes(sdm.touched_flags().data(), m * sizeof(uint8_t));
        if (bin) {
            w.write_bytes(sdm.binarized_contents().data(), m * sizeof(HyperVector));
        } else {
            w.write_bytes(sdm.counters().data(), m * VSA_BITS * sizeof(int8_t));
        }
    }

    // 3. Tsetlin (save sparse if consolidated, else full)
    {
        BinaryWriter w(dir_path + "/tsetlin.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_TM);
        uint8_t is_sparse = has_sparse_tsetlin ? 1 : 0;
        w.write_pod(is_sparse);
        if (is_sparse) {
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.num_features));
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.num_classes));
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.clauses_per_class));
            for (const auto& cl : sparse_tsetlin.clauses) {
                w.write_pod(cl.sign);
                uint32_t np = static_cast<uint32_t>(cl.pos_literals.size());
                uint32_t nn = static_cast<uint32_t>(cl.neg_literals.size());
                w.write_pod(np);
                w.write_pod(nn);
                w.write_bytes(cl.pos_literals.data(), np * sizeof(uint16_t));
                w.write_bytes(cl.neg_literals.data(), nn * sizeof(uint16_t));
            }
        } else {
            w.write_pod(tsetlin.config());
            w.write_bytes(tsetlin.ta_states().data(), tsetlin.ta_states().size() * sizeof(int8_t));
            w.write_bytes(tsetlin.clause_signs().data(), tsetlin.clause_signs().size() * sizeof(int8_t));
        }
    }

    // 4. Causal Graph
    {
        BinaryWriter w(dir_path + "/graph.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_GR);
        uint32_t nn = static_cast<uint32_t>(graph.nodes().size());
        uint32_t nr = static_cast<uint32_t>(graph.relations().size());
        uint32_t ne = static_cast<uint32_t>(graph.edges().size());
        w.write_pod(nn);
        w.write_pod(nr);
        w.write_pod(ne);
        for (const auto& node : graph.nodes()) {
            w.write_pod(node.id);
            w.write_pod(node.frequency);
            w.write_string(node.name);
        }
        for (const auto& rel : graph.relations()) {
            w.write_string(rel);
        }
        if (ne > 0) {
            w.write_bytes(graph.edges().data(), ne * sizeof(GraphEdge));
        }
    }

    // 5. Realizer
    {
        BinaryWriter w(dir_path + "/realizer.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_RL);
        w.write_pod(realizer.total_tokens());
        uint32_t nf = static_cast<uint32_t>(realizer.frames().size());
        w.write_pod(nf);
        for (const auto& f : realizer.frames()) {
            w.write_pod(static_cast<uint8_t>(f.rel_type));
            w.write_pod(f.frequency);
            w.write_string(f.pattern);
        }
        auto write_map = [&](const std::unordered_map<std::string, uint32_t>& m) {
            uint32_t sz = static_cast<uint32_t>(m.size());
            w.write_pod(sz);
            for (const auto& [k, v] : m) {
                w.write_string(k);
                w.write_pod(v);
            }
        };
        write_map(realizer.unigrams());
        write_map(realizer.bigrams());
        write_map(realizer.trigrams());
    }

    return true;
}

bool ModelBundle::load(const std::string& dir_path, bool use_mmap) {
    // 1. Codebook
    {
        MappedReader r(dir_path + "/codebook.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, n = 0;
        if (!r.read_pod(magic) || magic != MAGIC_CB || !r.read_pod(n)) return false;
        std::vector<std::string> toks(n);
        for (uint32_t i = 0; i < n; ++i) {
            if (!r.read_string(toks[i])) return false;
        }
        std::vector<HyperVector> base_v(n), sem_v(n);
        if (n > 0) {
            if (!r.read_bytes(base_v.data(), n * sizeof(HyperVector))) return false;
            if (!r.read_bytes(sem_v.data(), n * sizeof(HyperVector))) return false;
        }
        codebook.load_raw(std::move(toks), std::move(base_v), std::move(sem_v));
    }

    // 2. SDM
    {
        MappedReader r(dir_path + "/sdm.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0;
        SDMConfig cfg;
        uint8_t bin = 0;
        uint64_t wc = 0;
        if (!r.read_pod(magic) || magic != MAGIC_SDM || !r.read_pod(cfg) ||
            !r.read_pod(bin) || !r.read_pod(wc)) return false;
        const size_t m = cfg.num_locations;
        std::vector<HyperVector> addrs(m);
        std::vector<uint8_t> touched(m);
        if (!r.read_bytes(addrs.data(), m * sizeof(HyperVector))) return false;
        if (!r.read_bytes(touched.data(), m * sizeof(uint8_t))) return false;
        std::vector<int8_t> counters;
        std::vector<HyperVector> bin_c;
        if (bin) {
            bin_c.resize(m);
            if (!r.read_bytes(bin_c.data(), m * sizeof(HyperVector))) return false;
        } else {
            counters.resize(m * VSA_BITS);
            if (!r.read_bytes(counters.data(), m * VSA_BITS * sizeof(int8_t))) return false;
        }
        sdm.load_raw(cfg, std::move(addrs), std::move(counters), std::move(bin_c),
                     std::move(touched), bin != 0, wc);
    }

    // 3. Tsetlin
    {
        MappedReader r(dir_path + "/tsetlin.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0;
        uint8_t is_sparse = 0;
        if (!r.read_pod(magic) || magic != MAGIC_TM || !r.read_pod(is_sparse)) return false;
        if (is_sparse) {
            uint32_t nf = 0, nc = 0, cpc = 0;
            if (!r.read_pod(nf) || !r.read_pod(nc) || !r.read_pod(cpc)) return false;
            sparse_tsetlin.num_features = nf;
            sparse_tsetlin.num_classes = nc;
            sparse_tsetlin.clauses_per_class = cpc;
            sparse_tsetlin.clauses.resize(static_cast<size_t>(nc) * cpc);
            for (auto& cl : sparse_tsetlin.clauses) {
                uint32_t np = 0, nn = 0;
                if (!r.read_pod(cl.sign) || !r.read_pod(np) || !r.read_pod(nn)) return false;
                cl.pos_literals.resize(np);
                cl.neg_literals.resize(nn);
                if (!r.read_bytes(cl.pos_literals.data(), np * sizeof(uint16_t))) return false;
                if (!r.read_bytes(cl.neg_literals.data(), nn * sizeof(uint16_t))) return false;
            }
            has_sparse_tsetlin = true;
        } else {
            TsetlinConfig cfg;
            if (!r.read_pod(cfg)) return false;
            std::vector<int8_t> ta(cfg.num_classes * cfg.clauses_per_class * 2 * cfg.num_features);
            std::vector<int8_t> signs(cfg.num_classes * cfg.clauses_per_class);
            if (!r.read_bytes(ta.data(), ta.size() * sizeof(int8_t))) return false;
            if (!r.read_bytes(signs.data(), signs.size() * sizeof(int8_t))) return false;
            tsetlin.load_raw(cfg, std::move(ta), std::move(signs));
            has_sparse_tsetlin = false;
        }
    }

    // 4. Causal Graph
    {
        MappedReader r(dir_path + "/graph.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, nn = 0, nr = 0, ne = 0;
        if (!r.read_pod(magic) || magic != MAGIC_GR ||
            !r.read_pod(nn) || !r.read_pod(nr) || !r.read_pod(ne)) return false;
        std::vector<GraphNode> nodes(nn);
        for (uint32_t i = 0; i < nn; ++i) {
            if (!r.read_pod(nodes[i].id) || !r.read_pod(nodes[i].frequency) ||
                !r.read_string(nodes[i].name)) return false;
        }
        std::vector<std::string> rels(nr);
        for (uint32_t i = 0; i < nr; ++i) {
            if (!r.read_string(rels[i])) return false;
        }
        std::vector<GraphEdge> edges(ne);
        if (ne > 0 && !r.read_bytes(edges.data(), ne * sizeof(GraphEdge))) return false;
        graph.load_raw(std::move(nodes), std::move(rels), std::move(edges));
    }

    // 5. Realizer
    {
        MappedReader r(dir_path + "/realizer.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, nf = 0;
        uint64_t total_toks = 0;
        if (!r.read_pod(magic) || magic != MAGIC_RL ||
            !r.read_pod(total_toks) || !r.read_pod(nf)) return false;
        std::vector<ConstructionFrame> frames(nf);
        for (uint32_t i = 0; i < nf; ++i) {
            uint8_t rt = 0;
            if (!r.read_pod(rt) || !r.read_pod(frames[i].frequency) ||
                !r.read_string(frames[i].pattern)) return false;
            frames[i].rel_type = static_cast<RelationType>(rt);
        }
        auto read_map = [&](std::unordered_map<std::string, uint32_t>& m) -> bool {
            uint32_t sz = 0;
            if (!r.read_pod(sz)) return false;
            m.reserve(sz);
            for (uint32_t i = 0; i < sz; ++i) {
                std::string k;
                uint32_t v = 0;
                if (!r.read_string(k) || !r.read_pod(v)) return false;
                m.emplace(std::move(k), v);
            }
            return true;
        };
        std::unordered_map<std::string, uint32_t> uni, bi, tri;
        if (!read_map(uni) || !read_map(bi) || !read_map(tri)) return false;
        realizer.load_raw(std::move(frames), std::move(uni), std::move(bi), std::move(tri), total_toks);
    }

    return true;
}

size_t ModelBundle::disk_size_bytes(const std::string& dir_path) {
    size_t total = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir_path, ec)) {
        if (entry.is_regular_file()) {
            total += entry.file_size();
        }
    }
    return total;
}

size_t ModelBundle::estimated_ram_bytes() const noexcept {
    size_t total = codebook.size() * (2 * sizeof(HyperVector) + 32);
    total += sdm.memory_bytes();
    total += has_sparse_tsetlin ? sparse_tsetlin.memory_bytes() : tsetlin.memory_bytes();
    total += graph.memory_bytes();
    total += realizer.memory_bytes();
    return total;
}

} // namespace cognicore
