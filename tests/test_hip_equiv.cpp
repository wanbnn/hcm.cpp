#include "core/backend.hpp"
#include "core/vsa.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace cognicore;

int main() {
    if (!backend::is_gpu_available()) {
        std::cout << "[SKIP] HIP GPU not available (CPU fallback active).\n";
        return 0;
    }

    std::cout << "[INFO] HIP GPU detected! Running exact numerical equivalence tests...\n";

    // 1. Batch Hamming equivalence
    {
        const size_t M = 512;
        HyperVector query = HyperVector::random(111);
        std::vector<HyperVector> db(M);
        for (size_t i = 0; i < M; ++i) db[i] = HyperVector::random(1000 + i);

        std::vector<uint32_t> dist_cpu(M), dist_gpu(M);
        backend::batch_hamming(query.words.data(), db[0].words.data(), M, dist_cpu.data(), false);
        backend::batch_hamming(query.words.data(), db[0].words.data(), M, dist_gpu.data(), true);
        assert(dist_cpu == dist_gpu);
        std::cout << "[PASS] HIP batch_hamming == CPU\n";
    }

    // 2. SDM write & read equivalence
    {
        const size_t M = 256;
        std::vector<int8_t> c_cpu(M * VSA_BITS, 0);
        std::vector<int8_t> c_gpu(M * VSA_BITS, 0);
        std::vector<uint32_t> active = {3, 17, 42, 99, 128, 250};
        HyperVector data = HyperVector::random(777);

        backend::sdm_write_active(c_cpu.data(), active.data(), active.size(), data.words.data(), false);
        backend::sdm_write_active(c_gpu.data(), active.data(), active.size(), data.words.data(), true);
        assert(c_cpu == c_gpu);

        std::vector<int32_t> sum_cpu(VSA_BITS), sum_gpu(VSA_BITS);
        backend::sdm_read_active(c_cpu.data(), active.data(), active.size(), sum_cpu.data(), false);
        backend::sdm_read_active(c_gpu.data(), active.data(), active.size(), sum_gpu.data(), true);
        assert(sum_cpu == sum_gpu);
        std::cout << "[PASS] HIP sdm_write_active & sdm_read_active == CPU\n";
    }

    // 3. Tsetlin eval & update equivalence
    {
        const size_t num_clauses = 40;
        const size_t num_features = 512;
        const int8_t n_states = 30;

        std::vector<int8_t> ta_cpu(num_clauses * 2 * num_features, n_states);
        for (size_t i = 0; i < ta_cpu.size(); i += 7) ta_cpu[i] = n_states + 1;
        std::vector<int8_t> ta_gpu = ta_cpu;

        std::vector<int8_t> signs(num_clauses);
        for (size_t c = 0; c < num_clauses; ++c) signs[c] = (c % 2 == 0) ? 1 : -1;

        HyperVector in_hv = HyperVector::random(999);
        std::vector<uint8_t> out_cpu(num_clauses), out_gpu(num_clauses);

        backend::tsetlin_eval_clauses(ta_cpu.data(), in_hv.words.data(), num_clauses, num_features, n_states, out_cpu.data(), false);
        backend::tsetlin_eval_clauses(ta_gpu.data(), in_hv.words.data(), num_clauses, num_features, n_states, out_gpu.data(), true);
        assert(out_cpu == out_gpu);

        int32_t class_sum = 0;
        for (size_t c = 0; c < num_clauses; ++c) class_sum += out_cpu[c] * signs[c];

        backend::tsetlin_update_clauses(ta_cpu.data(), in_hv.words.data(), out_cpu.data(), signs.data(),
                                        num_clauses, num_features, n_states, class_sum, 10, 10, 4.0f, 12345ULL, false);
        backend::tsetlin_update_clauses(ta_gpu.data(), in_hv.words.data(), out_gpu.data(), signs.data(),
                                        num_clauses, num_features, n_states, class_sum, 10, 10, 4.0f, 12345ULL, true);
        assert(ta_cpu == ta_gpu);
        std::cout << "[PASS] HIP tsetlin_eval_clauses & tsetlin_update_clauses == CPU\n";
    }

    return 0;
}
