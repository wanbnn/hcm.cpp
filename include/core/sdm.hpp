#pragma once

#include "core/vsa.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cognicore {

struct SDMConfig {
    size_t num_locations = 2048;   // M hard locations
    size_t top_k = 16;             // If > 0, activate top-K nearest locations; else use radius
    uint32_t hamming_radius = 1920;// Used when top_k == 0
    uint64_t seed = 0x517CC1B727220A95ULL;
};

class KanervaSDM {
public:
    explicit KanervaSDM(const SDMConfig& cfg = SDMConfig{});

    void write(const HyperVector& address, const HyperVector& data, bool use_gpu = false);
    HyperVector read(const HyperVector& address, bool use_gpu = false) const;

    /// Compacts int8 counters into 1-bit HyperVectors (8x smaller) for CPU-only inference.
    void binarize_for_inference();
    bool is_binarized() const noexcept { return binarized_; }

    size_t num_locations() const noexcept { return cfg_.num_locations; }
    size_t write_count() const noexcept { return write_count_; }
    size_t occupied_locations() const noexcept;
    size_t memory_bytes() const noexcept;
    const SDMConfig& config() const noexcept { return cfg_; }

    /// Finds active hard-location indices for a given address.
    std::vector<uint32_t> activate(const HyperVector& address, bool use_gpu = false) const;

    // Raw accessors for serialization / mmap
    const std::vector<HyperVector>& addresses() const { return addresses_; }
    const std::vector<int8_t>& counters() const { return counters_; }
    const std::vector<HyperVector>& binarized_contents() const { return binarized_contents_; }
    const std::vector<uint8_t>& touched_flags() const { return touched_; }

    void load_raw(SDMConfig cfg,
                  std::vector<HyperVector> addresses,
                  std::vector<int8_t> counters,
                  std::vector<HyperVector> binarized_contents,
                  std::vector<uint8_t> touched,
                  bool binarized,
                  size_t write_count);

private:
    SDMConfig cfg_;
    std::vector<HyperVector> addresses_;
    std::vector<int8_t> counters_;              // M * VSA_BITS (when not binarized)
    std::vector<HyperVector> binarized_contents_; // M * HyperVector (when binarized)
    std::vector<uint8_t> touched_;
    bool binarized_ = false;
    size_t write_count_ = 0;
};

} // namespace cognicore
