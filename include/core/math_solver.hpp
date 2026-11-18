#pragma once

#include "core/language_rules.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace cognicore {

struct MathResult {
    int64_t value = 0;
    std::string expression;
};

/// Small symbolic arithmetic/word-problem solver (GSM8K/MathVista-inspired
/// benchmark support). Integer arithmetic only; no language model.
class MathSolver {
public:
    MathSolver();
    explicit MathSolver(const LanguageRules& rules) : rules_(rules) {}
    std::optional<MathResult> solve(const std::string& text) const;
private:
    LanguageRules rules_;
};

} // namespace cognicore
