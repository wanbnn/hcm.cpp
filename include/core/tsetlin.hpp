#pragma once

#include "core/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cognicore {

enum class DialogIntent : uint8_t {
    EXPLAIN_CAUSE = 0,
    ANSWER_FACT   = 1,
    COMPARE       = 2,
    ASK_CLARIFY   = 3,
    AGREE         = 4,
    HYPOTHESIZE   = 5,
    GREET         = 6,
    THANK         = 7,
    FAREWELL      = 8,
    INTRODUCE     = 9,
    HELP          = 10,
    SMALLTALK     = 11,
    COUNT         = 12
};

const char* intent_to_string(DialogIntent intent) noexcept;
DialogIntent string_to_intent(const std::string& s) noexcept;

struct TsetlinConfig {
    size_t num_features = 4096 + 64; // VSA bits + context/graph state bits
    size_t num_classes = static_cast<size_t>(DialogIntent::COUNT);
    size_t clauses_per_class = 40;   // Must be even (half +1, half -1)
    int8_t n_states = 50;            // States in [1, 2*n_states], threshold = n_states
    int32_t margin_T = 15;
    float s_param = 4.0f;
    uint64_t seed = 0x123456789ABCDEF0ULL;
};

struct SparseClause {
    int8_t sign = 1;
    std::vector<uint16_t> pos_literals; // Indices f where x[f] must be 1
    std::vector<uint16_t> neg_literals; // Indices f where x[f] must be 0
};

struct SparseTsetlinModel {
    size_t num_features = 0;
    size_t num_classes = 0;
    size_t clauses_per_class = 0;
    std::vector<SparseClause> clauses; // num_classes * clauses_per_class

    /// Evaluates class scores using only included literal indices.
    std::vector<int32_t> score_classes(const uint64_t* input_bits) const;
    size_t predict(const uint64_t* input_bits) const;
    size_t active_literal_count() const noexcept;
    size_t non_empty_clause_count() const noexcept;
    size_t memory_bytes() const noexcept;
};

class TsetlinMachine {
public:
    explicit TsetlinMachine(const TsetlinConfig& cfg = TsetlinConfig{});

    /// Trains on one sample (packed uint64_t bit-vector of `num_features` bits).
    void fit_sample(const uint64_t* input_bits,
                    size_t target_class,
                    uint64_t step_seed,
                    bool use_gpu = false);

    /// Trains for `epochs` over a dataset.
    void fit(const std::vector<std::vector<uint64_t>>& X_bits,
             const std::vector<size_t>& y,
             size_t epochs,
             bool use_gpu = false);

    /// Returns raw class vote sums.
    std::vector<int32_t> score_classes(const uint64_t* input_bits, bool use_gpu = false) const;

    /// Returns predicted class index.
    size_t predict(const uint64_t* input_bits, bool use_gpu = false) const;

    /// Exports only included literals (ta_state > n_states) into compact uint16_t lists.
    SparseTsetlinModel export_sparse() const;

    const TsetlinConfig& config() const noexcept { return cfg_; }
    size_t total_clauses() const noexcept { return cfg_.num_classes * cfg_.clauses_per_class; }
    size_t memory_bytes() const noexcept;

    const std::vector<int8_t>& ta_states() const { return ta_states_; }
    const std::vector<int8_t>& clause_signs() const { return clause_signs_; }
    void load_raw(TsetlinConfig cfg,
                  std::vector<int8_t> ta_states,
                  std::vector<int8_t> clause_signs);

private:
    TsetlinConfig cfg_;
    // Layout: [class][clause][2 * num_features]
    std::vector<int8_t> ta_states_;
    // Layout: [class][clause] (+1 or -1)
    std::vector<int8_t> clause_signs_;
};

} // namespace cognicore
