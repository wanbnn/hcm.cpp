#pragma once

#include <cstddef>
#include <cstdint>

namespace cognicore::backend {

inline constexpr size_t VSA_BITS = 4096;
inline constexpr size_t VSA_WORDS = VSA_BITS / 64;

/// Returns true if a HIP/ROCm GPU device is compiled in and available at runtime.
bool is_gpu_available();

/// Computes Hamming distance between a single 4096-bit query vector and `num_vecs`
/// packed 4096-bit database vectors (each `VSA_WORDS` uint64_t).
void batch_hamming(const uint64_t* query,
                   const uint64_t* db,
                   size_t num_vecs,
                   uint32_t* out_dists,
                   bool use_gpu = false);

/// SDM Write: for each active location index in `active_indices[0..num_active-1]`,
/// updates the `VSA_BITS` int8_t counters in `counters` (layout: `M x VSA_BITS`)
/// by +1 where `data` bit is 1, and -1 where `data` bit is 0, saturating at [-127, 127].
void sdm_write_active(int8_t* counters,
                      const uint32_t* active_indices,
                      size_t num_active,
                      const uint64_t* data_words,
                      bool use_gpu = false);

/// SDM Read: sums int8_t counters across `active_indices[0..num_active-1]` into
/// `out_sums[0..VSA_BITS-1]` (int32_t).
void sdm_read_active(const int8_t* counters,
                     const uint32_t* active_indices,
                     size_t num_active,
                     int32_t* out_sums,
                     bool use_gpu = false);

/// Tsetlin Machine: evaluates `num_clauses` clauses for one input sample.
/// `ta_states` has shape [num_clauses, 2 * num_features] (positive then negated literals).
/// `input_bits` has `num_features` bits packed in uint64_t words.
/// `clause_outputs` receives 1 if all included literals match, else 0.
void tsetlin_eval_clauses(const int8_t* ta_states,
                          const uint64_t* input_bits,
                          size_t num_clauses,
                          size_t num_features,
                          int8_t threshold,
                          uint8_t* clause_outputs,
                          bool use_gpu = false);

/// Tsetlin Machine: updates TA states for one class/sample using Type I and Type II feedback.
/// `clause_signs` is +1 or -1 per clause. `target_sign` is +1 (positive class) or -1 (negative).
/// `rand_seeds` provides per-clause deterministic PRNG seeds for stochastic feedback.
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
                            bool use_gpu = false);

#if COGNICORE_USE_ROCM
namespace hip {
bool is_gpu_available();
void batch_hamming(const uint64_t* query,
                   const uint64_t* db,
                   size_t num_vecs,
                   uint32_t* out_dists);
void sdm_write_active(int8_t* counters,
                      const uint32_t* active_indices,
                      size_t num_active,
                      const uint64_t* data_words);
void sdm_read_active(const int8_t* counters,
                     const uint32_t* active_indices,
                     size_t num_active,
                     int32_t* out_sums);
void tsetlin_eval_clauses(const int8_t* ta_states,
                          const uint64_t* input_bits,
                          size_t num_clauses,
                          size_t num_features,
                          int8_t threshold,
                          uint8_t* clause_outputs);
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
                            uint64_t step_seed);
} // namespace hip
#endif

} // namespace cognicore::backend
