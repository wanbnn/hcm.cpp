#include "core/sdm.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace cognicore;

HyperVector flip_random_bits(const HyperVector& v, size_t num_flips, uint64_t seed) {
    HyperVector noisy = v;
    for (size_t i = 0; i < num_flips; ++i) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        const size_t bit = (seed >> 32) % VSA_BITS;
        noisy.set_bit(bit, !noisy.get_bit(bit));
    }
    return noisy;
}

void test_sdm_storage_and_noise_recovery() {
    SDMConfig cfg;
    cfg.num_locations = 2048;
    cfg.top_k = 48;
    KanervaSDM sdm(cfg);

    const size_t N = 50;
    std::vector<HyperVector> addrs(N);
    std::vector<HyperVector> datas(N);
    for (size_t i = 0; i < N; ++i) {
        addrs[i] = HyperVector::random(1000 + i);
        datas[i] = HyperVector::random(5000 + i);
        sdm.write(addrs[i], datas[i]);
    }

    assert(sdm.write_count() == N);
    assert(sdm.occupied_locations() > 0);

    double avg_exact_sim = 0.0;
    double avg_noisy_sim = 0.0;
    for (size_t i = 0; i < N; ++i) {
        auto rec = sdm.read(addrs[i]);
        double sim = rec.similarity(datas[i]);
        avg_exact_sim += sim;
        assert(sim > 0.80);

        // 10% bit noise (~410 flipped bits)
        auto noisy_addr = flip_random_bits(addrs[i], 410, 9000 + i);
        auto rec_noisy = sdm.read(noisy_addr);
        double nsim = rec_noisy.similarity(datas[i]);
        avg_noisy_sim += nsim;
        assert(nsim > 0.65);
    }
    avg_exact_sim /= N;
    avg_noisy_sim /= N;

    std::cout << "[PASS] test_sdm_storage_and_noise_recovery (avg_exact_sim="
              << avg_exact_sim << ", avg_noisy_sim=" << avg_noisy_sim << ")\n";

    // Test binarization compaction
    const size_t bytes_before = sdm.memory_bytes();
    sdm.binarize_for_inference();
    const size_t bytes_after = sdm.memory_bytes();
    assert(sdm.is_binarized());
    assert(bytes_after < bytes_before / 4);

    double avg_bin_sim = 0.0;
    for (size_t i = 0; i < N; ++i) {
        auto rec = sdm.read(addrs[i]);
        double sim = rec.similarity(datas[i]);
        avg_bin_sim += sim;
        assert(sim > 0.70);
    }
    avg_bin_sim /= N;

    std::cout << "[PASS] test_sdm_binarize (bytes: " << bytes_before << " -> " << bytes_after
              << ", avg_bin_sim=" << avg_bin_sim << ")\n";
}

int main() {
    test_sdm_storage_and_noise_recovery();
    return 0;
}
