#include "core/tsetlin.hpp"
#include "core/vsa.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace cognicore;

void test_noisy_xor_and_sparse_export() {
    TsetlinConfig cfg;
    cfg.num_features = 12; // bit 0 ^ bit 1, bits 2..11 are random noise
    cfg.num_classes = 2;
    cfg.clauses_per_class = 20;
    cfg.n_states = 40;
    cfg.margin_T = 10;
    cfg.s_param = 3.5f;
    cfg.seed = 42;

    TsetlinMachine tm(cfg);

    std::vector<std::vector<uint64_t>> X;
    std::vector<size_t> y;
    uint64_t rng = 123456789ULL;
    for (size_t i = 0; i < 400; ++i) {
        uint64_t bits = 0;
        const size_t b0 = i & 1ULL;
        const size_t b1 = (i >> 1) & 1ULL;
        if (b0) bits |= 1ULL;
        if (b1) bits |= 2ULL;
        for (size_t f = 2; f < 12; ++f) {
            rng = rng * 6364136223846793005ULL + 1ULL;
            if ((rng >> 63) & 1ULL) bits |= (1ULL << f);
        }
        X.push_back({bits});
        y.push_back(b0 ^ b1);
    }

    tm.fit(X, y, 30);

    size_t correct = 0;
    auto sparse = tm.export_sparse();
    for (size_t i = 0; i < X.size(); ++i) {
        auto s_full = tm.score_classes(X[i].data());
        auto s_sparse = sparse.score_classes(X[i].data());
        assert(s_full == s_sparse);

        size_t pred = tm.predict(X[i].data());
        assert(pred == sparse.predict(X[i].data()));
        if (pred == y[i]) ++correct;
    }

    double acc = static_cast<double>(correct) / X.size();
    assert(acc >= 0.99);
    std::cout << "[PASS] test_noisy_xor_and_sparse_export (acc=" << acc
              << ", active_literals=" << sparse.active_literal_count() << ")\n";
}

void test_vsa_4096_multiclass_and_compression() {
    // 4 prototype hypervectors, noisy samples (5% bit flips)
    const size_t num_classes = 4;
    std::vector<HyperVector> protos(num_classes);
    for (size_t k = 0; k < num_classes; ++k) {
        protos[k] = HyperVector::random(7000 + k);
    }

    std::vector<std::vector<uint64_t>> X_train, X_test;
    std::vector<size_t> y_train, y_test;

    uint64_t rng = 987654321ULL;
    auto make_noisy = [&](const HyperVector& p) {
        HyperVector v = p;
        for (size_t f = 0; f < 200; ++f) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            size_t bit = (rng >> 32) % VSA_BITS;
            v.set_bit(bit, !v.get_bit(bit));
        }
        return std::vector<uint64_t>(v.words.begin(), v.words.end());
    };

    for (size_t i = 0; i < 80; ++i) {
        size_t c = i % num_classes;
        X_train.push_back(make_noisy(protos[c]));
        y_train.push_back(c);
    }
    for (size_t i = 0; i < 40; ++i) {
        size_t c = i % num_classes;
        X_test.push_back(make_noisy(protos[c]));
        y_test.push_back(c);
    }

    TsetlinConfig cfg;
    cfg.num_features = VSA_BITS;
    cfg.num_classes = num_classes;
    cfg.clauses_per_class = 20;
    cfg.n_states = 30;
    cfg.margin_T = 10;
    cfg.s_param = 5.0f;

    TsetlinMachine tm(cfg);
    tm.fit(X_train, y_train, 15);

    auto sparse = tm.export_sparse();
    size_t correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        assert(tm.score_classes(X_test[i].data()) == sparse.score_classes(X_test[i].data()));
        if (sparse.predict(X_test[i].data()) == y_test[i]) ++correct;
    }

    double acc = static_cast<double>(correct) / X_test.size();
    assert(acc >= 0.95);
    assert(sparse.memory_bytes() < tm.memory_bytes() / 4);

    std::cout << "[PASS] test_vsa_4096_multiclass_and_compression (acc=" << acc
              << ", full_bytes=" << tm.memory_bytes()
              << ", sparse_bytes=" << sparse.memory_bytes() << ")\n";
}

int main() {
    test_noisy_xor_and_sparse_export();
    test_vsa_4096_multiclass_and_compression();
    return 0;
}
