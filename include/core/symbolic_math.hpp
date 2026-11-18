#pragma once

#include "core/language_rules.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace cognicore {

struct Rational {
    int64_t numerator = 0;
    int64_t denominator = 1;
    double value() const { return denominator ? double(numerator) / double(denominator) : 0.0; }
    std::string to_string() const;
};

struct SymbolicResult {
    std::string expression;
    std::string steps;
    Rational value;
    std::string variable;
};

/// Symbolic arithmetic/equation engine with fractions, parentheses, powers and
/// linear equation solving. No floating-point neural machinery and no CAS
/// dependency.
class SymbolicMath {
public:
    explicit SymbolicMath(const LanguageRules& rules = LanguageRules{});
    std::optional<SymbolicResult> evaluate(const std::string& text) const;
    /// Extracts a bare arithmetic expression from natural-language wrapper.
    static std::string extract_expression(const std::string& text, const LanguageRules& rules);
    std::optional<SymbolicResult> solve(const std::string& text) const;
    bool verify_linear(const std::string& equation, const Rational& x) const;
private:
    LanguageRules rules_;
};

} // namespace cognicore
