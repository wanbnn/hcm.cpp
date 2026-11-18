#include "core/tsetlin.hpp"

#include <algorithm>
#include <cstring>

namespace cognicore {

namespace {

inline uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline bool get_bit(const uint64_t* words, size_t idx) noexcept {
    return ((words[idx >> 6] >> (idx & 63)) & 1ULL) != 0;
}

} // namespace

const char* intent_to_string(DialogIntent intent) noexcept {
    switch (intent) {
        case DialogIntent::EXPLAIN_CAUSE: return "EXPLAIN_CAUSE";
        case DialogIntent::ANSWER_FACT:   return "ANSWER_FACT";
        case DialogIntent::COMPARE:       return "COMPARE";
        case DialogIntent::ASK_CLARIFY:   return "ASK_CLARIFY";
        case DialogIntent::AGREE:         return "AGREE";
        case DialogIntent::HYPOTHESIZE:   return "HYPOTHESIZE";
        case DialogIntent::GREET:         return "GREET";
        default:                          return "ANSWER_FACT";
    }
}

DialogIntent string_to_intent(const std::string& s) noexcept {
    if (s == "EXPLAIN_CAUSE") return DialogIntent::EXPLAIN_CAUSE;
    if (s == "ANSWER_FACT")   return DialogIntent::ANSWER_FACT;
    if (s == "COMPARE")       return DialogIntent::COMPARE;
    if (s == "ASK_CLARIFY")   return DialogIntent::ASK_CLARIFY;
    if (s == "AGREE")         return DialogIntent::AGREE;
    if (s == "HYPOTHESIZE")   return DialogIntent::HYPOTHESIZE;
    if (s == "GREET")         return DialogIntent::GREET;
    return DialogIntent::ANSWER_FACT;
}

std::vector<int32_t> SparseTsetlinModel::score_classes(const uint64_t* input_bits) const {
    std::vector<int32_t> scores(num_classes, 0);
    for (size_t k = 0; k < num_classes; ++k) {
        int32_t sum = 0;
        const size_t base = k * clauses_per_class;
        for (size_t c = 0; c < clauses_per_class; ++c) {
            const auto& cl = clauses[base + c];
            bool active = true;
            for (uint16_t f : cl.pos_literals) {
                if (!get_bit(input_bits, f)) {
                    active = false;
                    break;
                }
            }
            if (active) {
                for (uint16_t f : cl.neg_literals) {
                    if (get_bit(input_bits, f)) {
                        active = false;
                        break;
                    }
                }
            }
            if (active) {
                sum += cl.sign;
            }
        }
        scores[k] = sum;
    }
    return scores;
}

size_t SparseTsetlinModel::predict(const uint64_t* input_bits) const {
    auto scores = score_classes(input_bits);
    return static_cast<size_t>(
        std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
}

size_t SparseTsetlinModel::active_literal_count() const noexcept {
    size_t total = 0;
    for (const auto& cl : clauses) {
        total += cl.pos_literals.size() + cl.neg_literals.size();
    }
    return total;
}

size_t SparseTsetlinModel::non_empty_clause_count() const noexcept {
    size_t count = 0;
    for (const auto& cl : clauses) {
        if (!cl.pos_literals.empty() || !cl.neg_literals.empty()) {
            ++count;
        }
    }
    return count;
}

size_t SparseTsetlinModel::memory_bytes() const noexcept {
    size_t bytes = sizeof(*this) + clauses.size() * sizeof(SparseClause);
    for (const auto& cl : clauses) {
        bytes += (cl.pos_literals.size() + cl.neg_literals.size()) * sizeof(uint16_t);
    }
    return bytes;
}

TsetlinMachine::TsetlinMachine(const TsetlinConfig& cfg)
    : cfg_(cfg),
      ta_states_(cfg.num_classes * cfg.clauses_per_class * 2 * cfg.num_features),
      clause_signs_(cfg.num_classes * cfg.clauses_per_class) {
    uint64_t rng = cfg_.seed;
    if (cfg_.num_features <= 64) {
        for (size_t i = 0; i < ta_states_.size(); ++i) {
            rng = splitmix64(rng);
            ta_states_[i] = (rng & 1ULL) ? cfg_.n_states : static_cast<int8_t>(cfg_.n_states + 1);
        }
    } else {
        // High-dimensional initialization: start just below threshold so clauses are not deadlocked
        std::fill(ta_states_.begin(), ta_states_.end(), cfg_.n_states);
    }
    for (size_t k = 0; k < cfg_.num_classes; ++k) {
        for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
            clause_signs_[k * cfg_.clauses_per_class + c] = (c % 2 == 0) ? int8_t{1} : int8_t{-1};
        }
    }
}

void TsetlinMachine::fit_sample(const uint64_t* input_bits,
                                size_t target_class,
                                uint64_t step_seed,
                                bool use_gpu) {
    const size_t stride_class_ta = cfg_.clauses_per_class * 2 * cfg_.num_features;
    std::vector<uint8_t> clause_outputs(cfg_.clauses_per_class);

    // Update target class (+margin_T)
    {
        int8_t* ta_ptr = ta_states_.data() + target_class * stride_class_ta;
        const int8_t* sign_ptr = clause_signs_.data() + target_class * cfg_.clauses_per_class;
        backend::tsetlin_eval_clauses(ta_ptr, input_bits, cfg_.clauses_per_class,
                                      cfg_.num_features, cfg_.n_states,
                                      clause_outputs.data(), use_gpu);
        int32_t class_sum = 0;
        for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
            class_sum += static_cast<int32_t>(clause_outputs[c]) * sign_ptr[c];
        }
        backend::tsetlin_update_clauses(ta_ptr, input_bits, clause_outputs.data(), sign_ptr,
                                        cfg_.clauses_per_class, cfg_.num_features,
                                        cfg_.n_states, class_sum, cfg_.margin_T,
                                        cfg_.margin_T, cfg_.s_param,
                                        step_seed ^ (target_class * 0x9E3779B97F4A7C15ULL),
                                        use_gpu);
    }

    // Update negative classes (-margin_T)
    if (cfg_.num_classes > 1) {
        if (cfg_.num_classes <= 8) {
            for (size_t neg = 0; neg < cfg_.num_classes; ++neg) {
                if (neg == target_class) continue;
                int8_t* ta_ptr = ta_states_.data() + neg * stride_class_ta;
                const int8_t* sign_ptr = clause_signs_.data() + neg * cfg_.clauses_per_class;
                backend::tsetlin_eval_clauses(ta_ptr, input_bits, cfg_.clauses_per_class,
                                              cfg_.num_features, cfg_.n_states,
                                              clause_outputs.data(), use_gpu);
                int32_t class_sum = 0;
                for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
                    class_sum += static_cast<int32_t>(clause_outputs[c]) * sign_ptr[c];
                }
                backend::tsetlin_update_clauses(ta_ptr, input_bits, clause_outputs.data(), sign_ptr,
                                                cfg_.clauses_per_class, cfg_.num_features,
                                                cfg_.n_states, class_sum, -cfg_.margin_T,
                                                cfg_.margin_T, cfg_.s_param,
                                                step_seed ^ (neg * 0xBF58476D1CE4E5B9ULL),
                                                use_gpu);
            }
        } else {
            size_t neg = static_cast<size_t>(splitmix64(step_seed) % (cfg_.num_classes - 1));
            if (neg >= target_class) ++neg;
            int8_t* ta_ptr = ta_states_.data() + neg * stride_class_ta;
            const int8_t* sign_ptr = clause_signs_.data() + neg * cfg_.clauses_per_class;
            backend::tsetlin_eval_clauses(ta_ptr, input_bits, cfg_.clauses_per_class,
                                          cfg_.num_features, cfg_.n_states,
                                          clause_outputs.data(), use_gpu);
            int32_t class_sum = 0;
            for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
                class_sum += static_cast<int32_t>(clause_outputs[c]) * sign_ptr[c];
            }
            backend::tsetlin_update_clauses(ta_ptr, input_bits, clause_outputs.data(), sign_ptr,
                                            cfg_.clauses_per_class, cfg_.num_features,
                                            cfg_.n_states, class_sum, -cfg_.margin_T,
                                            cfg_.margin_T, cfg_.s_param,
                                            step_seed ^ (neg * 0xBF58476D1CE4E5B9ULL),
                                            use_gpu);
        }
    }
}

void TsetlinMachine::fit(const std::vector<std::vector<uint64_t>>& X_bits,
                         const std::vector<size_t>& y,
                         size_t epochs,
                         bool use_gpu) {
    const size_t n = std::min(X_bits.size(), y.size());
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;

    uint64_t seed = cfg_.seed;
    for (size_t ep = 0; ep < epochs; ++ep) {
        // Fisher-Yates shuffle
        for (size_t i = n; i > 1; --i) {
            seed = splitmix64(seed);
            const size_t j = static_cast<size_t>(seed % i);
            std::swap(order[i - 1], order[j]);
        }
        for (size_t idx : order) {
            seed = splitmix64(seed);
            fit_sample(X_bits[idx].data(), y[idx], seed, use_gpu);
        }
    }
}

std::vector<int32_t> TsetlinMachine::score_classes(const uint64_t* input_bits, bool use_gpu) const {
    std::vector<int32_t> scores(cfg_.num_classes, 0);
    std::vector<uint8_t> clause_outputs(cfg_.clauses_per_class);
    const size_t stride_class_ta = cfg_.clauses_per_class * 2 * cfg_.num_features;

    for (size_t k = 0; k < cfg_.num_classes; ++k) {
        const int8_t* ta_ptr = ta_states_.data() + k * stride_class_ta;
        const int8_t* sign_ptr = clause_signs_.data() + k * cfg_.clauses_per_class;
        backend::tsetlin_eval_clauses(ta_ptr, input_bits, cfg_.clauses_per_class,
                                      cfg_.num_features, cfg_.n_states,
                                      clause_outputs.data(), use_gpu);
        int32_t sum = 0;
        for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
            sum += static_cast<int32_t>(clause_outputs[c]) * sign_ptr[c];
        }
        scores[k] = sum;
    }
    return scores;
}

size_t TsetlinMachine::predict(const uint64_t* input_bits, bool use_gpu) const {
    auto scores = score_classes(input_bits, use_gpu);
    return static_cast<size_t>(
        std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
}

SparseTsetlinModel TsetlinMachine::export_sparse() const {
    SparseTsetlinModel sparse;
    sparse.num_features = cfg_.num_features;
    sparse.num_classes = cfg_.num_classes;
    sparse.clauses_per_class = cfg_.clauses_per_class;
    sparse.clauses.resize(cfg_.num_classes * cfg_.clauses_per_class);

    const size_t stride = 2 * cfg_.num_features;
    for (size_t k = 0; k < cfg_.num_classes; ++k) {
        for (size_t c = 0; c < cfg_.clauses_per_class; ++c) {
            const size_t global_c = k * cfg_.clauses_per_class + c;
            const int8_t* row = ta_states_.data() + global_c * stride;
            auto& sc = sparse.clauses[global_c];
            sc.sign = clause_signs_[global_c];
            for (size_t f = 0; f < cfg_.num_features; ++f) {
                if (row[f] > cfg_.n_states) {
                    sc.pos_literals.push_back(static_cast<uint16_t>(f));
                }
                if (row[cfg_.num_features + f] > cfg_.n_states) {
                    sc.neg_literals.push_back(static_cast<uint16_t>(f));
                }
            }
        }
    }
    return sparse;
}

size_t TsetlinMachine::memory_bytes() const noexcept {
    return ta_states_.size() * sizeof(int8_t) + clause_signs_.size() * sizeof(int8_t);
}

void TsetlinMachine::load_raw(TsetlinConfig cfg,
                              std::vector<int8_t> ta_states,
                              std::vector<int8_t> clause_signs) {
    cfg_ = cfg;
    ta_states_ = std::move(ta_states);
    clause_signs_ = std::move(clause_signs);
}

} // namespace cognicore
