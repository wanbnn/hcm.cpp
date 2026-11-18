#include "core/backend.hpp"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>

namespace cognicore::backend::hip {

namespace {

inline void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "[HIP ERROR] %s: %s\n", what, hipGetErrorString(err));
    }
}

__device__ __forceinline__ uint64_t splitmix64_dev(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline uint64_t splitmix64_host(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float hash_to_float_dev(uint64_t seed, uint64_t a, uint64_t b) {
    uint64_t h = splitmix64_dev(seed ^ (a * 0xD6E8FEB86659FD93ULL) ^ (b * 0xA5A3564E27F8862DULL));
    return static_cast<float>(h >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ bool get_bit_dev(const uint64_t* words, size_t bit_idx) {
    return ((words[bit_idx >> 6] >> (bit_idx & 63)) & 1ULL) != 0;
}

__device__ __forceinline__ int8_t clamp_int8_dev(int32_t v) {
    if (v > 127) return 127;
    if (v < -127) return -127;
    return static_cast<int8_t>(v);
}

__global__ void kernel_batch_hamming(const uint64_t* __restrict__ query,
                                     const uint64_t* __restrict__ db,
                                     size_t num_vecs,
                                     uint32_t* __restrict__ out_dists) {
    // 1 block of 64 threads per database vector (64 words == VSA_WORDS)
    const size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const unsigned tid = threadIdx.x; // 0..63
    __shared__ uint32_t s_partial[64];

    const uint64_t q_w = query[tid];
    const uint64_t d_w = db[vec_idx * VSA_WORDS + tid];
    s_partial[tid] = static_cast<uint32_t>(__popcll(q_w ^ d_w));
    __syncthreads();

    for (unsigned stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid] += s_partial[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_dists[vec_idx] = s_partial[0];
    }
}

__global__ void kernel_sdm_write_active(int8_t* __restrict__ counters,
                                        const uint32_t* __restrict__ active_indices,
                                        size_t num_active,
                                        const uint64_t* __restrict__ data_words) {
    const size_t a = blockIdx.y;
    const size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (a >= num_active || d >= VSA_BITS) return;

    const uint32_t loc = active_indices[a];
    const int32_t delta = get_bit_dev(data_words, d) ? 1 : -1;
    int8_t* cell = counters + static_cast<size_t>(loc) * VSA_BITS + d;
    *cell = clamp_int8_dev(static_cast<int32_t>(*cell) + delta);
}

__global__ void kernel_sdm_read_active(const int8_t* __restrict__ counters,
                                       const uint32_t* __restrict__ active_indices,
                                       size_t num_active,
                                       int32_t* __restrict__ out_sums) {
    const size_t d = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (d >= VSA_BITS) return;

    int32_t sum = 0;
    for (size_t a = 0; a < num_active; ++a) {
        const uint32_t loc = active_indices[a];
        sum += static_cast<int32_t>(counters[static_cast<size_t>(loc) * VSA_BITS + d]);
    }
    out_sums[d] = sum;
}

__global__ void kernel_tsetlin_eval(const int8_t* __restrict__ ta_states,
                                    const uint64_t* __restrict__ input_bits,
                                    size_t num_clauses,
                                    size_t num_features,
                                    int8_t threshold,
                                    uint8_t* __restrict__ clause_outputs) {
    const size_t c = blockIdx.x;
    if (c >= num_clauses) return;

    const size_t stride = 2 * num_features;
    const int8_t* row = ta_states + c * stride;

    __shared__ uint32_t s_ok;
    if (threadIdx.x == 0) s_ok = 1;
    __syncthreads();

    uint32_t local_ok = 1;
    for (size_t f = threadIdx.x; f < num_features; f += blockDim.x) {
        const bool x = get_bit_dev(input_bits, f);
        const bool inc_pos = row[f] > threshold;
        const bool inc_neg = row[num_features + f] > threshold;
        if ((inc_pos && !x) || (inc_neg && x)) {
            local_ok = 0;
            break;
        }
    }

    if (local_ok == 0) {
        atomicAnd(&s_ok, 0U);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        clause_outputs[c] = static_cast<uint8_t>(s_ok);
    }
}

__global__ void kernel_tsetlin_update(int8_t* __restrict__ ta_states,
                                      const uint64_t* __restrict__ input_bits,
                                      const uint8_t* __restrict__ clause_outputs,
                                      const int8_t* __restrict__ clause_signs,
                                      size_t num_clauses,
                                      size_t num_features,
                                      int8_t n_states,
                                      float update_prob,
                                      int32_t target_sum,
                                      float p_reward,
                                      float p_penalty,
                                      uint64_t step_seed) {
    const size_t c = blockIdx.y;
    const size_t f = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= num_clauses || f >= num_features) return;

    if (hash_to_float_dev(step_seed, c, 0xFFFFFFFFULL) > update_prob) {
        return;
    }

    const int32_t max_state = 2 * static_cast<int32_t>(n_states);
    const size_t stride = 2 * num_features;
    int8_t* row = ta_states + c * stride;

    const bool c_out = clause_outputs[c] != 0;
    const bool same_polarity = (target_sum > 0 && clause_signs[c] > 0) ||
                               (target_sum < 0 && clause_signs[c] < 0);
    const bool x = get_bit_dev(input_bits, f);

    if (same_polarity) {
        if (c_out) {
            const float r_pos = hash_to_float_dev(step_seed, c, f * 2);
            const float r_neg = hash_to_float_dev(step_seed, c, f * 2 + 1);
            if (x) {
                if (row[f] < max_state && r_pos <= p_reward) row[f]++;
                if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
            } else {
                if (row[num_features + f] < max_state && r_neg <= p_reward) row[num_features + f]++;
                if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
            }
        } else {
            const float r_pos = hash_to_float_dev(step_seed, c, f * 2);
            const float r_neg = hash_to_float_dev(step_seed, c, f * 2 + 1);
            if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
            if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
        }
    } else {
        if (c_out) {
            if (!x && row[f] <= n_states && row[f] < max_state) {
                row[f]++;
            }
            if (x && row[num_features + f] <= n_states && row[num_features + f] < max_state) {
                row[num_features + f]++;
            }
        }
    }
}

__global__ void kernel_tsetlin_update_resident(int8_t* __restrict__ ta_states,
                                               const uint64_t* __restrict__ input_bits,
                                               const uint8_t* __restrict__ clause_outputs,
                                               const int8_t* __restrict__ clause_signs,
                                               size_t num_clauses,
                                               size_t num_features,
                                               int8_t n_states,
                                               int32_t target_sum,
                                               int32_t margin_T,
                                               float s_param,
                                               uint64_t step_seed) {
    const size_t c = blockIdx.y;
    const size_t f = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= num_clauses || f >= num_features) return;

    // class_sum is identical in every block and avoids a host round-trip.
    int32_t class_sum = 0;
    for (size_t k = 0; k < num_clauses; ++k) {
        class_sum += static_cast<int32_t>(clause_outputs[k]) *
                     static_cast<int32_t>(clause_signs[k]);
    }
    const int32_t clamped_sum = class_sum > margin_T ? margin_T
                              : (class_sum < -margin_T ? -margin_T : class_sum);
    const float update_prob = (target_sum > 0)
        ? static_cast<float>(margin_T - clamped_sum) / static_cast<float>(2 * margin_T)
        : static_cast<float>(margin_T + clamped_sum) / static_cast<float>(2 * margin_T);
    const float p_reward = (s_param - 1.0f) / s_param;
    const float p_penalty = 1.0f / s_param;

    if (hash_to_float_dev(step_seed, c, 0xFFFFFFFFULL) > update_prob) return;

    const int32_t max_state = 2 * static_cast<int32_t>(n_states);
    const size_t stride = 2 * num_features;
    int8_t* row = ta_states + c * stride;

    const bool c_out = clause_outputs[c] != 0;
    const bool same_polarity = (target_sum > 0 && clause_signs[c] > 0) ||
                               (target_sum < 0 && clause_signs[c] < 0);
    const bool x = get_bit_dev(input_bits, f);

    if (same_polarity) {
        if (c_out) {
            const float r_pos = hash_to_float_dev(step_seed, c, f * 2);
            const float r_neg = hash_to_float_dev(step_seed, c, f * 2 + 1);
            if (x) {
                if (row[f] < max_state && r_pos <= p_reward) row[f]++;
                if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
            } else {
                if (row[num_features + f] < max_state && r_neg <= p_reward) row[num_features + f]++;
                if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
            }
        } else {
            const float r_pos = hash_to_float_dev(step_seed, c, f * 2);
            const float r_neg = hash_to_float_dev(step_seed, c, f * 2 + 1);
            if (row[f] > 1 && r_pos <= p_penalty) row[f]--;
            if (row[num_features + f] > 1 && r_neg <= p_penalty) row[num_features + f]--;
        }
    } else {
        if (c_out) {
            if (!x && row[f] <= n_states && row[f] < max_state) row[f]++;
            if (x && row[num_features + f] <= n_states && row[num_features + f] < max_state) {
                row[num_features + f]++;
            }
        }
    }
}

} // namespace

bool is_gpu_available() {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    cached = (err == hipSuccess && count > 0) ? 1 : 0;
    return cached == 1;
}

void batch_hamming(const uint64_t* query,
                   const uint64_t* db,
                   size_t num_vecs,
                   uint32_t* out_dists) {
    if (num_vecs == 0) return;

    uint64_t* d_query = nullptr;
    uint64_t* d_db = nullptr;
    uint32_t* d_out = nullptr;

    check_hip(hipMalloc(&d_query, VSA_WORDS * sizeof(uint64_t)), "hipMalloc d_query");
    check_hip(hipMalloc(&d_db, num_vecs * VSA_WORDS * sizeof(uint64_t)), "hipMalloc d_db");
    check_hip(hipMalloc(&d_out, num_vecs * sizeof(uint32_t)), "hipMalloc d_out");

    check_hip(hipMemcpy(d_query, query, VSA_WORDS * sizeof(uint64_t), hipMemcpyHostToDevice), "H2D query");
    check_hip(hipMemcpy(d_db, db, num_vecs * VSA_WORDS * sizeof(uint64_t), hipMemcpyHostToDevice), "H2D db");

    hipLaunchKernelGGL(kernel_batch_hamming,
                       dim3(static_cast<unsigned>(num_vecs)),
                       dim3(64),
                       0, 0,
                       d_query, d_db, num_vecs, d_out);
    check_hip(hipDeviceSynchronize(), "kernel_batch_hamming sync");
    check_hip(hipMemcpy(out_dists, d_out, num_vecs * sizeof(uint32_t), hipMemcpyDeviceToHost), "D2H out");

    hipFree(d_query);
    hipFree(d_db);
    hipFree(d_out);
}

void sdm_write_active(int8_t* counters,
                      const uint32_t* active_indices,
                      size_t num_active,
                      const uint64_t* data_words) {
    if (num_active == 0) return;

    // Find max index to allocate only needed slice
    uint32_t max_idx = 0;
    for (size_t i = 0; i < num_active; ++i) {
        if (active_indices[i] > max_idx) max_idx = active_indices[i];
    }
    const size_t num_rows = static_cast<size_t>(max_idx) + 1;
    const size_t counters_bytes = num_rows * VSA_BITS * sizeof(int8_t);

    int8_t* d_counters = nullptr;
    uint32_t* d_active = nullptr;
    uint64_t* d_data = nullptr;

    check_hip(hipMalloc(&d_counters, counters_bytes), "hipMalloc d_counters");
    check_hip(hipMalloc(&d_active, num_active * sizeof(uint32_t)), "hipMalloc d_active");
    check_hip(hipMalloc(&d_data, VSA_WORDS * sizeof(uint64_t)), "hipMalloc d_data");

    check_hip(hipMemcpy(d_counters, counters, counters_bytes, hipMemcpyHostToDevice), "H2D counters");
    check_hip(hipMemcpy(d_active, active_indices, num_active * sizeof(uint32_t), hipMemcpyHostToDevice), "H2D active");
    check_hip(hipMemcpy(d_data, data_words, VSA_WORDS * sizeof(uint64_t), hipMemcpyHostToDevice), "H2D data");

    dim3 block(256);
    dim3 grid(static_cast<unsigned>(VSA_BITS / 256), static_cast<unsigned>(num_active));
    hipLaunchKernelGGL(kernel_sdm_write_active, grid, block, 0, 0,
                       d_counters, d_active, num_active, d_data);
    check_hip(hipDeviceSynchronize(), "kernel_sdm_write_active sync");
    check_hip(hipMemcpy(counters, d_counters, counters_bytes, hipMemcpyDeviceToHost), "D2H counters");

    hipFree(d_counters);
    hipFree(d_active);
    hipFree(d_data);
}

void sdm_read_active(const int8_t* counters,
                     const uint32_t* active_indices,
                     size_t num_active,
                     int32_t* out_sums) {
    if (num_active == 0) {
        std::fill(out_sums, out_sums + VSA_BITS, 0);
        return;
    }

    uint32_t max_idx = 0;
    for (size_t i = 0; i < num_active; ++i) {
        if (active_indices[i] > max_idx) max_idx = active_indices[i];
    }
    const size_t num_rows = static_cast<size_t>(max_idx) + 1;
    const size_t counters_bytes = num_rows * VSA_BITS * sizeof(int8_t);

    int8_t* d_counters = nullptr;
    uint32_t* d_active = nullptr;
    int32_t* d_sums = nullptr;

    check_hip(hipMalloc(&d_counters, counters_bytes), "hipMalloc d_counters");
    check_hip(hipMalloc(&d_active, num_active * sizeof(uint32_t)), "hipMalloc d_active");
    check_hip(hipMalloc(&d_sums, VSA_BITS * sizeof(int32_t)), "hipMalloc d_sums");

    check_hip(hipMemcpy(d_counters, counters, counters_bytes, hipMemcpyHostToDevice), "H2D counters");
    check_hip(hipMemcpy(d_active, active_indices, num_active * sizeof(uint32_t), hipMemcpyHostToDevice), "H2D active");

    hipLaunchKernelGGL(kernel_sdm_read_active,
                       dim3(static_cast<unsigned>(VSA_BITS / 256)),
                       dim3(256),
                       0, 0,
                       d_counters, d_active, num_active, d_sums);
    check_hip(hipDeviceSynchronize(), "kernel_sdm_read_active sync");
    check_hip(hipMemcpy(out_sums, d_sums, VSA_BITS * sizeof(int32_t), hipMemcpyDeviceToHost), "D2H sums");

    hipFree(d_counters);
    hipFree(d_active);
    hipFree(d_sums);
}

void tsetlin_eval_clauses(const int8_t* ta_states,
                          const uint64_t* input_bits,
                          size_t num_clauses,
                          size_t num_features,
                          int8_t threshold,
                          uint8_t* clause_outputs) {
    if (num_clauses == 0) return;

    const size_t ta_bytes = num_clauses * 2 * num_features * sizeof(int8_t);
    const size_t input_words = (num_features + 63) / 64;

    int8_t* d_ta = nullptr;
    uint64_t* d_in = nullptr;
    uint8_t* d_out = nullptr;

    check_hip(hipMalloc(&d_ta, ta_bytes), "hipMalloc d_ta");
    check_hip(hipMalloc(&d_in, input_words * sizeof(uint64_t)), "hipMalloc d_in");
    check_hip(hipMalloc(&d_out, num_clauses * sizeof(uint8_t)), "hipMalloc d_out");

    check_hip(hipMemcpy(d_ta, ta_states, ta_bytes, hipMemcpyHostToDevice), "H2D ta");
    check_hip(hipMemcpy(d_in, input_bits, input_words * sizeof(uint64_t), hipMemcpyHostToDevice), "H2D in");

    hipLaunchKernelGGL(kernel_tsetlin_eval,
                       dim3(static_cast<unsigned>(num_clauses)),
                       dim3(256),
                       0, 0,
                       d_ta, d_in, num_clauses, num_features, threshold, d_out);
    check_hip(hipDeviceSynchronize(), "kernel_tsetlin_eval sync");
    check_hip(hipMemcpy(clause_outputs, d_out, num_clauses * sizeof(uint8_t), hipMemcpyDeviceToHost), "D2H out");

    hipFree(d_ta);
    hipFree(d_in);
    hipFree(d_out);
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
                            uint64_t step_seed) {
    if (num_clauses == 0) return;

    const int32_t clamped_sum = std::max(-margin_T, std::min(margin_T, class_sum));
    const float update_prob = (target_sum > 0)
        ? static_cast<float>(margin_T - clamped_sum) / static_cast<float>(2 * margin_T)
        : static_cast<float>(margin_T + clamped_sum) / static_cast<float>(2 * margin_T);
    const float p_reward = (s_param - 1.0f) / s_param;
    const float p_penalty = 1.0f / s_param;

    const size_t ta_bytes = num_clauses * 2 * num_features * sizeof(int8_t);
    const size_t input_words = (num_features + 63) / 64;

    int8_t* d_ta = nullptr;
    uint64_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    int8_t* d_signs = nullptr;

    check_hip(hipMalloc(&d_ta, ta_bytes), "hipMalloc d_ta");
    check_hip(hipMalloc(&d_in, input_words * sizeof(uint64_t)), "hipMalloc d_in");
    check_hip(hipMalloc(&d_out, num_clauses * sizeof(uint8_t)), "hipMalloc d_out");
    check_hip(hipMalloc(&d_signs, num_clauses * sizeof(int8_t)), "hipMalloc d_signs");

    check_hip(hipMemcpy(d_ta, ta_states, ta_bytes, hipMemcpyHostToDevice), "H2D ta");
    check_hip(hipMemcpy(d_in, input_bits, input_words * sizeof(uint64_t), hipMemcpyHostToDevice), "H2D in");
    check_hip(hipMemcpy(d_out, clause_outputs, num_clauses * sizeof(uint8_t), hipMemcpyHostToDevice), "H2D out");
    check_hip(hipMemcpy(d_signs, clause_signs, num_clauses * sizeof(int8_t), hipMemcpyHostToDevice), "H2D signs");

    dim3 block(256);
    dim3 grid(static_cast<unsigned>((num_features + 255) / 256), static_cast<unsigned>(num_clauses));
    hipLaunchKernelGGL(kernel_tsetlin_update, grid, block, 0, 0,
                       d_ta, d_in, d_out, d_signs,
                       num_clauses, num_features, n_states,
                       update_prob, target_sum, p_reward, p_penalty, step_seed);
    check_hip(hipDeviceSynchronize(), "kernel_tsetlin_update sync");
    check_hip(hipMemcpy(ta_states, d_ta, ta_bytes, hipMemcpyDeviceToHost), "D2H ta");

    hipFree(d_ta);
    hipFree(d_in);
    hipFree(d_out);
    hipFree(d_signs);
}

bool tsetlin_train_device(std::vector<int8_t>& ta_states,
                          std::vector<int8_t>& clause_signs,
                          const std::vector<std::vector<uint64_t>>& X_bits,
                          const std::vector<size_t>& y,
                          size_t epochs,
                          const TsetlinTrainConfig& cfg) {
    if (!is_gpu_available()) return false;
    const size_t n = std::min(X_bits.size(), y.size());
    if (n == 0 || cfg.num_classes == 0 || cfg.clauses_per_class == 0) return false;
    const size_t words = (cfg.num_features + 63) / 64;
    const size_t classes = cfg.num_classes;
    const size_t clauses = cfg.clauses_per_class;
    const size_t stride_ta = clauses * 2 * cfg.num_features;
    if (ta_states.size() != classes * stride_ta ||
        clause_signs.size() != classes * clauses) return false;

    // Flatten the packed bit-vectors once; it stays in VRAM for every epoch.
    std::vector<uint64_t> flat(n * words, 0);
    for (size_t i = 0; i < n; ++i) {
        const size_t copy = std::min(words, X_bits[i].size());
        if (copy) std::copy(X_bits[i].begin(), X_bits[i].begin() + copy,
                            flat.begin() + static_cast<std::ptrdiff_t>(i * words));
    }

    int8_t* d_ta = nullptr;
    int8_t* d_signs = nullptr;
    uint64_t* d_x = nullptr;
    uint8_t* d_out = nullptr;
    auto cleanup = [&]() {
        if (d_ta) hipFree(d_ta);
        if (d_signs) hipFree(d_signs);
        if (d_x) hipFree(d_x);
        if (d_out) hipFree(d_out);
    };
    auto ok = [](hipError_t err, const char* what) {
        if (err != hipSuccess) {
            std::fprintf(stderr, "[HIP ERROR] %s: %s\n", what, hipGetErrorString(err));
            return false;
        }
        return true;
    };
    if (!ok(hipMalloc(&d_ta, ta_states.size() * sizeof(int8_t)), "hipMalloc d_ta") ||
        !ok(hipMalloc(&d_signs, clause_signs.size() * sizeof(int8_t)), "hipMalloc d_signs") ||
        !ok(hipMalloc(&d_x, flat.size() * sizeof(uint64_t)), "hipMalloc d_x") ||
        !ok(hipMalloc(&d_out, clauses * sizeof(uint8_t)), "hipMalloc d_out")) {
        cleanup();
        return false;
    }
    if (!ok(hipMemcpy(d_ta, ta_states.data(), ta_states.size() * sizeof(int8_t),
                      hipMemcpyHostToDevice), "H2D ta") ||
        !ok(hipMemcpy(d_signs, clause_signs.data(), clause_signs.size() * sizeof(int8_t),
                      hipMemcpyHostToDevice), "H2D signs") ||
        !ok(hipMemcpy(d_x, flat.data(), flat.size() * sizeof(uint64_t),
                      hipMemcpyHostToDevice), "H2D x")) {
        cleanup();
        return false;
    }

    std::fprintf(stderr,
                 "[HIP] Tsetlin residente na VRAM: TA=%.2f MB, X=%.2f MB, "
                 "amostras=%zu, epocas=%zu\n",
                 double(ta_states.size()) / 1e6, double(flat.size() * 8) / 1e6,
                 n, epochs);

    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    uint64_t seed = cfg.seed;
    const dim3 eval_grid(static_cast<unsigned>(clauses));
    const dim3 eval_block(256);
    const dim3 update_grid(static_cast<unsigned>((cfg.num_features + 255) / 256),
                           static_cast<unsigned>(clauses));
    const dim3 update_block(256);

    for (size_t ep = 0; ep < epochs; ++ep) {
        for (size_t i = n; i > 1; --i) {
            seed = splitmix64_host(seed);
            const size_t j = static_cast<size_t>(seed % i);
            std::swap(order[i - 1], order[j]);
        }
        for (size_t idx : order) {
            seed = splitmix64_host(seed);
            const size_t target = y[idx];
            if (target >= classes) continue;
            const uint64_t* d_sample = d_x + idx * words;
            auto step = [&](size_t klass, uint64_t class_seed, int32_t target_sum) {
                int8_t* class_ta = d_ta + klass * stride_ta;
                const int8_t* class_signs = d_signs + klass * clauses;
                hipLaunchKernelGGL(kernel_tsetlin_eval, eval_grid, eval_block, 0, 0,
                                   class_ta, d_sample, clauses, cfg.num_features,
                                   cfg.n_states, d_out);
                hipLaunchKernelGGL(kernel_tsetlin_update_resident, update_grid, update_block,
                                   0, 0, class_ta, d_sample, d_out, class_signs, clauses,
                                   cfg.num_features, cfg.n_states, target_sum, cfg.margin_T,
                                   cfg.s_param, class_seed);
            };
            step(target, seed ^ (target * 0x9E3779B97F4A7C15ULL), cfg.margin_T);
            if (classes <= 8) {
                for (size_t neg = 0; neg < classes; ++neg) {
                    if (neg == target) continue;
                    step(neg, seed ^ (neg * 0xBF58476D1CE4E5B9ULL), -cfg.margin_T);
                }
            } else {
                size_t neg = static_cast<size_t>(splitmix64_host(seed) % (classes - 1));
                if (neg >= target) ++neg;
                step(neg, seed ^ (neg * 0xBF58476D1CE4E5B9ULL), -cfg.margin_T);
            }
        }
    }

    if (!ok(hipDeviceSynchronize(), "tsetlin_train_device sync") ||
        !ok(hipMemcpy(ta_states.data(), d_ta, ta_states.size() * sizeof(int8_t),
                      hipMemcpyDeviceToHost), "D2H ta") ||
        !ok(hipMemcpy(clause_signs.data(), d_signs, clause_signs.size() * sizeof(int8_t),
                      hipMemcpyDeviceToHost), "D2H signs")) {
        cleanup();
        return false;
    }
    cleanup();
    return true;
}

} // namespace cognicore::backend::hip
