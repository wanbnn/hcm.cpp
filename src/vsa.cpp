#include "core/vsa.hpp"

#include <algorithm>
#include <cstring>

namespace cognicore {

namespace {

inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline uint64_t fnv1a_64(const std::string& s) noexcept {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : s) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

HyperVector HyperVector::zero() noexcept {
    HyperVector v;
    v.words.fill(0ULL);
    return v;
}

HyperVector HyperVector::random(uint64_t seed) noexcept {
    HyperVector v;
    uint64_t state = seed;
    for (size_t i = 0; i < VSA_WORDS; ++i) {
        state = splitmix64(state + i * 0x9E3779B97F4A7C15ULL);
        v.words[i] = state;
    }
    return v;
}

HyperVector HyperVector::from_token(const std::string& token) noexcept {
    return random(fnv1a_64(token));
}

bool HyperVector::operator==(const HyperVector& other) const noexcept {
    return words == other.words;
}

bool HyperVector::operator!=(const HyperVector& other) const noexcept {
    return words != other.words;
}

bool HyperVector::get_bit(size_t idx) const noexcept {
    return ((words[idx >> 6] >> (idx & 63)) & 1ULL) != 0;
}

void HyperVector::set_bit(size_t idx, bool val) noexcept {
    const uint64_t mask = 1ULL << (idx & 63);
    if (val) {
        words[idx >> 6] |= mask;
    } else {
        words[idx >> 6] &= ~mask;
    }
}

HyperVector HyperVector::bind(const HyperVector& other) const noexcept {
    HyperVector out;
    for (size_t i = 0; i < VSA_WORDS; ++i) {
        out.words[i] = words[i] ^ other.words[i];
    }
    return out;
}

HyperVector HyperVector::permute(int k) const noexcept {
    constexpr int TOTAL_BITS = static_cast<int>(VSA_BITS);
    int shift = ((k % TOTAL_BITS) + TOTAL_BITS) % TOTAL_BITS;
    if (shift == 0) {
        return *this;
    }

    const size_t word_shift = static_cast<size_t>(shift / 64);
    const unsigned bit_shift = static_cast<unsigned>(shift % 64);

    HyperVector out;
    if (bit_shift == 0) {
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            const size_t src = (w + VSA_WORDS - word_shift) % VSA_WORDS;
            out.words[w] = words[src];
        }
    } else {
        const unsigned inv_shift = 64U - bit_shift;
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            const size_t src_lo = (w + VSA_WORDS - word_shift) % VSA_WORDS;
            const size_t src_hi = (src_lo + VSA_WORDS - 1) % VSA_WORDS;
            out.words[w] = (words[src_lo] << bit_shift) | (words[src_hi] >> inv_shift);
        }
    }
    return out;
}

uint32_t HyperVector::hamming(const HyperVector& other) const noexcept {
    uint32_t dist = 0;
    for (size_t i = 0; i < VSA_WORDS; ++i) {
        dist += static_cast<uint32_t>(__builtin_popcountll(words[i] ^ other.words[i]));
    }
    return dist;
}

double HyperVector::similarity(const HyperVector& other) const noexcept {
    const uint32_t d = hamming(other);
    return 1.0 - (2.0 * static_cast<double>(d)) / static_cast<double>(VSA_BITS);
}

HyperVector HyperVector::bundle(const std::vector<HyperVector>& vecs, uint64_t tie_seed) {
    if (vecs.empty()) {
        return HyperVector::zero();
    }
    if (vecs.size() == 1) {
        return vecs[0];
    }

    std::array<uint32_t, VSA_BITS> counts{};
    for (const auto& v : vecs) {
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            uint64_t word = v.words[w];
            while (word != 0) {
                const unsigned tz = static_cast<unsigned>(__builtin_ctzll(word));
                counts[(w << 6) + tz] += 1;
                word &= (word - 1);
            }
        }
    }

    const uint32_t n = static_cast<uint32_t>(vecs.size());
    HyperVector out;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t out_word = 0;
        uint64_t tie_rand = splitmix64(tie_seed ^ w);
        for (size_t b = 0; b < 64; ++b) {
            const uint32_t c2 = counts[(w << 6) + b] * 2U;
            bool bit = false;
            if (c2 > n) {
                bit = true;
            } else if (c2 == n) {
                bit = ((tie_rand >> b) & 1ULL) != 0;
            }
            if (bit) {
                out_word |= (1ULL << b);
            }
        }
        out.words[w] = out_word;
    }
    return out;
}

uint32_t Codebook::get_or_add(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    const uint32_t id = static_cast<uint32_t>(id_to_token_.size());
    token_to_id_.emplace(token, id);
    id_to_token_.push_back(token);
    const HyperVector bv = HyperVector::from_token(token);
    base_vecs_.push_back(bv);
    semantic_vecs_.push_back(bv);
    return id;
}

bool Codebook::contains(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

uint32_t Codebook::find_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : UINT32_MAX;
}

void Codebook::observe_context(const std::string& center_token,
                               const std::vector<std::string>& context_tokens,
                               int32_t weight) {
    const uint32_t cid = get_or_add(center_token);
    if (accumulators_.size() <= cid) {
        accumulators_.resize(id_to_token_.size());
    }
    auto& acc = accumulators_[cid];
    for (const auto& ctx : context_tokens) {
        const uint32_t ctx_id = get_or_add(ctx);
        if (accumulators_.size() < id_to_token_.size()) {
            accumulators_.resize(id_to_token_.size());
        }
        const HyperVector& bv = base_vecs_[ctx_id];
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            const uint64_t word = bv.words[w];
            for (size_t b = 0; b < 64; ++b) {
                const int32_t delta = ((word >> b) & 1ULL) ? weight : -weight;
                accumulators_[cid][(w << 6) + b] += delta;
            }
        }
    }
    (void)acc;
    accumulators_dirty_ = true;
}

void Codebook::finalize_semantics() {
    if (!accumulators_dirty_ && accumulators_.empty()) {
        return;
    }
    const size_t n = std::min(id_to_token_.size(), accumulators_.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& acc = accumulators_[i];
        bool any_nonzero = false;
        HyperVector sv;
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; ++b) {
                const int32_t val = acc[(w << 6) + b];
                if (val != 0) any_nonzero = true;
                if (val > 0 || (val == 0 && base_vecs_[i].get_bit((w << 6) + b))) {
                    word |= (1ULL << b);
                }
            }
            sv.words[w] = word;
        }
        // An untouched token may already have learned semantics loaded from disk.
        if (any_nonzero) semantic_vecs_[i] = sv;
    }
    accumulators_dirty_ = false;
}

HyperVector Codebook::encode_sequence(const std::vector<std::string>& tokens,
                                      bool use_semantic) const {
    if (tokens.empty()) return HyperVector::zero();
    std::vector<HyperVector> shifted;
    shifted.reserve(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        const uint32_t id = find_id(tokens[i]);
        const HyperVector v = (id != UINT32_MAX)
            ? (use_semantic ? semantic_vecs_[id] : base_vecs_[id])
            : HyperVector::from_token(tokens[i]);
        shifted.push_back(v.permute(static_cast<int>(i)));
    }
    return HyperVector::bundle(shifted);
}

HyperVector Codebook::encode_bag(const std::vector<std::string>& tokens,
                                 bool use_semantic) const {
    if (tokens.empty()) return HyperVector::zero();
    std::vector<HyperVector> vecs;
    vecs.reserve(tokens.size());
    for (const auto& t : tokens) {
        const uint32_t id = find_id(t);
        vecs.push_back((id != UINT32_MAX)
            ? (use_semantic ? semantic_vecs_[id] : base_vecs_[id])
            : HyperVector::from_token(t));
    }
    return HyperVector::bundle(vecs);
}

std::vector<std::pair<std::string, double>> Codebook::query_nearest(const HyperVector& query,
                                                                    size_t top_k,
                                                                    bool use_semantic,
                                                                    bool use_gpu) const {
    const size_t n = id_to_token_.size();
    if (n == 0 || top_k == 0) return {};

    const auto& db_vecs = use_semantic ? semantic_vecs_ : base_vecs_;
    std::vector<uint32_t> dists(n);
    backend::batch_hamming(query.words.data(),
                           db_vecs[0].words.data(),
                           n,
                           dists.data(),
                           use_gpu);

    std::vector<uint32_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = static_cast<uint32_t>(i);

    const size_t k = std::min(top_k, n);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](uint32_t a, uint32_t b) { return dists[a] < dists[b]; });

    std::vector<std::pair<std::string, double>> result;
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        const uint32_t idx = indices[i];
        const double sim = 1.0 - (2.0 * static_cast<double>(dists[idx])) / static_cast<double>(VSA_BITS);
        result.emplace_back(id_to_token_[idx], sim);
    }
    return result;
}

void Codebook::load_raw(std::vector<std::string> tokens,
                        std::vector<HyperVector> base_vecs,
                        std::vector<HyperVector> semantic_vecs) {
    id_to_token_ = std::move(tokens);
    base_vecs_ = std::move(base_vecs);
    semantic_vecs_ = std::move(semantic_vecs);
    token_to_id_.clear();
    token_to_id_.reserve(id_to_token_.size());
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<uint32_t>(i);
    }
    accumulators_.clear();
    accumulators_dirty_ = false;
}

} // namespace cognicore
