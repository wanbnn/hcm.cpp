#include "core/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cognicore::backend {

namespace {

inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline float hash_to_float(uint64_t seed, uint64_t a, uint64_t b) noexcept {
    uint64_t h = splitmix64(seed ^ (a * 0xD6E8FEB86659FD93ULL) ^ (b * 0xA5A3564E27F8862DULL));
    // Upper 24 bits mapped to [0, 1)
    return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);
}

inline bool get_bit(const uint64_t* words, size_t bit_idx) noexcept {
    return ((words[bit_idx >> 6] >> (bit_idx & 63)) & 1ULL) != 0;
}

inline int8_t clamp_int8(int32_t v) noexcept {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return static_cast<int8_t>(v);
}

} // namespace

bool is_gpu_available() {
#if COGNICORE_USE_ROCM
    return hip::is_gpu_available();
#else
    return false;
#endif
}

void batch_hamming(const uint64_t* query,
                   const uint64_t* db,
                   size_t num_vecs,
                   uint32_t* out_dists,
                   bool use_gpu) {
#if COGNICORE_USE_ROCM
    if (use_gpu && hip::is_gpu_available()) {
        hip::batch_hamming(query, db, num_vecs, out_dists);
        return;
    }
#else
    (void)use_gpu;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_vecs >= 256)
#endif
    for (size_t i = 0; i < num_vecs; ++i) {
        const uint64_t* vec = db + i * VSA_WORDS;
        uint32_t dist = 0;
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            dist += static_cast<uint32_t>(__builtin_popcountll(query[w] ^ vec[w]));
        }
        out_dists[i] = dist;
    }
}

void sdm_write_active(int8_t* counters,
                      const uint32_t* active_indices,
                      size_t num_active,
                      const uint64_t* data_words,
                      bool use_gpu) {
#if COGNICORE_USE_ROCM
    if (use_gpu && hip::is_gpu_available()) {
        hip::sdm_write_active(counters, active_indices, num_active, data_words);
        return;
    }
#else
    (void)use_gpu;
#endif

    if (num_active == 0) return;

    int8_t deltas[VSA_BITS];
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        const uint64_t word = data_words[w];
        for (size_t b = 0; b < 64; ++b) {
            deltas[(w << 6) + b] = ((word >> b) & 1ULL) ? int8_t{1} : int8_t{-1};
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_active >= 64)
#endif
    for (size_t a = 0; a < num_active; ++a) {
        int8_t* row = counters + static_cast<size_t>(active_indices[a]) * VSA_BITS;
        for (size_t d = 0; d < VSA_BITS; ++d) {
            row[d] = clamp_int8(static_cast<int32_t>(row[d]) + static_cast<int32_t>(deltas[d]));
        }
    }
}

void sdm_read_active(const int8_t* counters,
                     const uint32_t* active_indices,
                     size_t num_active,
                     int32_t* out_sums,
                     bool use_gpu) {
#if COGNICORE_USE_ROCM
    if (use_gpu && hip::is_gpu_available()) {
        hip::sdm_read_active(counters, active_indices, num_active, out_sums);
        return;
    }
#else
    (void)use_gpu;
#endif

    std::memset(out_sums, 0, VSA_BITS * sizeof(int32_t));
    if (num_active == 0) return;

    for (size_t a = 0; a < num_active; ++a) {
        const int8_t* row = counters + static_cast<size_t>(active_indices[a]) * VSA_BITS;
        for (size_t d = 0; d < VSA_BITS; ++d) {
            out_sums[d] += static_cast<int32_t>(row[d]);
        }
    }
}

void tsetlin_eval_clauses(const int8_t* ta_states,
                          const uint64_t* input_bits,
                          size_t num_clauses,
                          size_t num_features,
                          int8_t threshold,
                          uint8_t* clause_outputs,
                          bool use_gpu) {
#if COGNICORE_USE_ROCM
    if (use_gpu && hip::is_gpu_available()) {
        hip::tsetlin_eval_clauses(ta_states, input_bits, num_clauses, num_features, threshold, clause_outputs);
        return;
    }
#else
    (void)use_gpu;
#endif

    const size_t stride = 2 * num_features;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_clauses * num_features >= 32768)
#endif
    for (size_t c = 0; c < num_clauses; ++c) {
        const int8_t* row = ta_states + c * stride;
        uint8_t out = 1;
        for (size_t f = 0; f < num_features; ++f) {
            const bool x = get_bit(input_bits, f);
            const bool inc_pos = row[f] > threshold;
            const bool inc_neg = row[num_features + f] > threshold;
            if ((inc_pos && !x) || (inc_neg && x)) {
                out = 0;
                break;
            }
        }
        clause_outputs[c] = out;
    }
}

void tsetlin_update_clauses(int8_t* ta_states,
                            const uint64_t* input_bits,
                            const uint8_t* clause_outputs,
                            const int8_t* clause_signs,
                            size_t num_clauses,
                            size_t num_features,
                            int8_t n_states,
                            int32_t class_sum,
                            int32_t target_sum,
                            int32_t margin_T,
                            float s_param,
                            uint64_t step_seed,
                            bool use_gpu) {
#if COGNICORE_USE_ROCM
    if (use_gpu && hip::is_gpu_available()) {
        hip::tsetlin_update_clauses(ta_states, input_bits, clause_outputs, clause_signs,
                                    num_clauses, num_features, n_states, class_sum,
                                    target_sum, margin_T, s_param, step_seed);
        return;
    }
#else
    (void)use_gpu;
#endif

    const int32_t max_state = 2 * static_cast<int32_t>(n_states);
    const int32_t clamped_sum = std::max(-margin_T, std::min(margin_T, class_sum));
    const float update_prob = (target_sum > 0)
        ? static_cast<float>(margin_T - clamped_sum) / static_cast<float>(2 * margin_T)
        : static_cast<float>(margin_T + clamped_sum) / static_cast<float>(2 * margin_T);

    const float p_reward = (s_param - 1.0f) / s_param;
    const float p_penalty = 1.0f / s_param;
    const size_t stride = 2 * num_features;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_clauses * num_features >= 32768)
#endif
    for (size_t c = 0; c < num_clauses; ++c) {
        if (hash_to_float(step_seed, c, 0xFFFFFFFFULL) > update_prob) {
            continue;
        }

        int8_t* row = ta_states + c * stride;
        const bool c_out = clause_outputs[c] != 0;
        const bool same_polarity = (target_sum > 0 && clause_signs[c] > 0) ||
                                   (target_sum < 0 && clause_signs[c] < 0);

        if (same_polarity) {
            // Type I feedback
            if (c_out) {
                for (size_t f = 0; f < num_features; ++f) {
                    const bool x = get_bit(input_bits, f);
                    const float r_pos = hash_to_float(step_seed, c, f * 2);
                    const float r_neg = hash_to_float(step_seed, c, f * 2 + 1);
                    if (x) {
                        if (row[f] < max_state && r_pos <= p_reward) row[f]++;
                        if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
                    } else {
                        if (row[num_features + f] < max_state && r_neg <= p_reward) row[num_features + f]++;
                        if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
                    }
                }
            } else {
                for (size_t f = 0; f < num_features; ++f) {
                    const float r_pos = hash_to_float(step_seed, c, f * 2);
                    const float r_neg = hash_to_float(step_seed, c, f * 2 + 1);
                    if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
                    if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
                }
            }
        } else {
            // Type II feedback
            if (c_out) {
                for (size_t f = 0; f < num_features; ++f) {
                    const bool x = get_bit(input_bits, f);
                    if (!x && row[f] <= n_states) {
                        if (row[f] < max_state) row[f]++;
                    }
                    if (x && row[num_features + f] <= n_states) {
                        if (row[num_features + f] < max_state) row[num_features + f]++;
                    }
                }
            }
        }
    }
}

} // namespace cognicore::backend
