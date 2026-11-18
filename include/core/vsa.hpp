#pragma once

#include "core/backend.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cognicore {

inline constexpr size_t VSA_BITS = backend::VSA_BITS;
inline constexpr size_t VSA_WORDS = backend::VSA_WORDS;

struct alignas(64) HyperVector {
    std::array<uint64_t, VSA_WORDS> words{};

    static HyperVector zero() noexcept;
    static HyperVector random(uint64_t seed) noexcept;
    static HyperVector from_token(const std::string& token) noexcept;

    bool operator==(const HyperVector& other) const noexcept;
    bool operator!=(const HyperVector& other) const noexcept;

    bool get_bit(size_t idx) const noexcept;
    void set_bit(size_t idx, bool val) noexcept;

    /// Bitwise XOR binding (self-inverse: a.bind(b).bind(b) == a).
    HyperVector bind(const HyperVector& other) const noexcept;
    HyperVector unbind(const HyperVector& other) const noexcept { return bind(other); }

    /// Circular bit rotation across all 4096 bits. Positive k shifts left; negative shifts right.
    HyperVector permute(int k) const noexcept;

    /// Hamming distance in [0, 4096].
    uint32_t hamming(const HyperVector& other) const noexcept;

    /// Cosine-like binary similarity in [-1.0, 1.0]: 1.0 - 2.0 * hamming / VSA_BITS.
    double similarity(const HyperVector& other) const noexcept;

    /// Majority-rule bundling with deterministic tie-breaking for even N.
    static HyperVector bundle(const std::vector<HyperVector>& vecs,
                              uint64_t tie_seed = 0x9E3779B97F4A7C15ULL);
};

/// Random Indexing Codebook: maps tokens to fixed random base vectors and
/// context-accumulated semantic vectors.
class Codebook {
public:
    uint32_t get_or_add(const std::string& token);
    bool contains(const std::string& token) const;
    uint32_t find_id(const std::string& token) const; // returns UINT32_MAX if missing

    const HyperVector& base_vector(uint32_t id) const { return base_vecs_.at(id); }
    const HyperVector& semantic_vector(uint32_t id) const { return semantic_vecs_.at(id); }
    const std::string& token(uint32_t id) const { return id_to_token_.at(id); }
    size_t size() const noexcept { return id_to_token_.size(); }

    /// Accumulates neighbor base vectors into center_token's semantic accumulator.
    void observe_context(const std::string& center_token,
                         const std::vector<std::string>& context_tokens,
                         int32_t weight = 1);

    /// Binarizes accumulated integer counters into `semantic_vecs_`.
    void finalize_semantics();

    /// Encodes a sequence of tokens into a single HyperVector using permute+bundle.
    HyperVector encode_sequence(const std::vector<std::string>& tokens,
                                bool use_semantic = true) const;

    /// Encodes a bag of tokens (order-independent) using bundle.
    HyperVector encode_bag(const std::vector<std::string>& tokens,
                           bool use_semantic = true) const;

    /// Finds top-K nearest tokens to `query` using batch Hamming distance.
    std::vector<std::pair<std::string, double>> query_nearest(const HyperVector& query,
                                                              size_t top_k = 5,
                                                              bool use_semantic = true,
                                                              bool use_gpu = false) const;

    // Direct accessors for serialization
    const std::vector<std::string>& tokens() const { return id_to_token_; }
    const std::vector<HyperVector>& base_vectors() const { return base_vecs_; }
    const std::vector<HyperVector>& semantic_vectors() const { return semantic_vecs_; }
    void load_raw(std::vector<std::string> tokens,
                  std::vector<HyperVector> base_vecs,
                  std::vector<HyperVector> semantic_vecs);

private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<HyperVector> base_vecs_;
    std::vector<HyperVector> semantic_vecs_;
    std::vector<std::array<int32_t, VSA_BITS>> accumulators_;
    bool accumulators_dirty_ = false;
};

} // namespace cognicore
