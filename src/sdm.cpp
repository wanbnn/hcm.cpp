#include "core/sdm.hpp"

#include <algorithm>
#include <array>
#include <numeric>

namespace cognicore {

namespace {

inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

KanervaSDM::KanervaSDM(const SDMConfig& cfg)
    : cfg_(cfg),
      addresses_(cfg.num_locations),
      counters_(cfg.num_locations * VSA_BITS, 0),
      touched_(cfg.num_locations, 0) {
    for (size_t i = 0; i < cfg_.num_locations; ++i) {
        addresses_[i] = HyperVector::random(splitmix64(cfg_.seed + i));
    }
}

std::vector<uint32_t> KanervaSDM::activate(const HyperVector& address, bool use_gpu) const {
    const size_t m = cfg_.num_locations;
    if (m == 0) return {};

    std::vector<uint32_t> dists(m);
    backend::batch_hamming(address.words.data(),
                           addresses_[0].words.data(),
                           m,
                           dists.data(),
                           use_gpu);

    std::vector<uint32_t> active;
    if (cfg_.top_k > 0) {
        const size_t k = std::min(cfg_.top_k, m);
        std::vector<uint32_t> idx(m);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](uint32_t a, uint32_t b) { return dists[a] < dists[b]; });
        active.assign(idx.begin(), idx.begin() + k);
    } else {
        for (size_t i = 0; i < m; ++i) {
            if (dists[i] <= cfg_.hamming_radius) {
                active.push_back(static_cast<uint32_t>(i));
            }
        }
    }
    return active;
}

void KanervaSDM::write(const HyperVector& address, const HyperVector& data, bool use_gpu) {
    if (binarized_) {
        // Expand binarized contents back to int8 counters for online learning
        counters_.resize(cfg_.num_locations * VSA_BITS, 0);
        for (size_t i = 0; i < cfg_.num_locations; ++i) {
            if (!touched_[i]) continue;
            int8_t* row = counters_.data() + i * VSA_BITS;
            for (size_t d = 0; d < VSA_BITS; ++d) {
                row[d] = binarized_contents_[i].get_bit(d) ? int8_t{1} : int8_t{-1};
            }
        }
        binarized_contents_.clear();
        binarized_ = false;
    }

    auto active = activate(address, use_gpu);
    if (active.empty()) return;

    for (uint32_t idx : active) {
        touched_[idx] = 1;
    }

    backend::sdm_write_active(counters_.data(),
                              active.data(),
                              active.size(),
                              data.words.data(),
                              use_gpu);
    ++write_count_;
}

HyperVector KanervaSDM::read(const HyperVector& address, bool use_gpu) const {
    auto active = activate(address, use_gpu);
    if (active.empty()) {
        return HyperVector::zero();
    }

    if (binarized_) {
        std::vector<HyperVector> active_vecs;
        active_vecs.reserve(active.size());
        for (uint32_t idx : active) {
            if (touched_[idx]) {
                active_vecs.push_back(binarized_contents_[idx]);
            }
        }
        return HyperVector::bundle(active_vecs);
    }

    std::array<int32_t, VSA_BITS> sums{};
    backend::sdm_read_active(counters_.data(),
                             active.data(),
                             active.size(),
                             sums.data(),
                             use_gpu);

    HyperVector out;
    for (size_t w = 0; w < VSA_WORDS; ++w) {
        uint64_t word = 0;
        for (size_t b = 0; b < 64; ++b) {
            if (sums[(w << 6) + b] > 0) {
                word |= (1ULL << b);
            }
        }
        out.words[w] = word;
    }
    return out;
}

void KanervaSDM::binarize_for_inference() {
    if (binarized_) return;
    binarized_contents_.resize(cfg_.num_locations);
    for (size_t i = 0; i < cfg_.num_locations; ++i) {
        const int8_t* row = counters_.data() + i * VSA_BITS;
        HyperVector hv;
        for (size_t w = 0; w < VSA_WORDS; ++w) {
            uint64_t word = 0;
            for (size_t b = 0; b < 64; ++b) {
                if (row[(w << 6) + b] > 0) {
                    word |= (1ULL << b);
                }
            }
            hv.words[w] = word;
        }
        binarized_contents_[i] = hv;
    }
    std::vector<int8_t>().swap(counters_);
    binarized_ = true;
}

size_t KanervaSDM::occupied_locations() const noexcept {
    size_t count = 0;
    for (uint8_t t : touched_) {
        if (t) ++count;
    }
    return count;
}

size_t KanervaSDM::memory_bytes() const noexcept {
    size_t total = addresses_.size() * sizeof(HyperVector);
    total += counters_.size() * sizeof(int8_t);
    total += binarized_contents_.size() * sizeof(HyperVector);
    total += touched_.size() * sizeof(uint8_t);
    return total;
}

void KanervaSDM::load_raw(SDMConfig cfg,
                          std::vector<HyperVector> addresses,
                          std::vector<int8_t> counters,
                          std::vector<HyperVector> binarized_contents,
                          std::vector<uint8_t> touched,
                          bool binarized,
                          size_t write_count) {
    cfg_ = cfg;
    addresses_ = std::move(addresses);
    counters_ = std::move(counters);
    binarized_contents_ = std::move(binarized_contents);
    touched_ = std::move(touched);
    binarized_ = binarized;
    write_count_ = write_count;
}

} // namespace cognicore
