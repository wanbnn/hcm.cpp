#include "core/language_rules.hpp"
#include "core/symbolic_math.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

int main() {
    LanguageRules rules;
    assert(rules.load_default());
    SymbolicMath math(rules);
    const auto fraction = math.evaluate("1/2 + 1/3");
    assert(fraction && fraction->value.numerator == 5 && fraction->value.denominator == 6);
    const auto powers = math.evaluate("(2 + 3) * 4^2");
    assert(powers && powers->value.numerator == 80 && powers->value.denominator == 1);
    const auto percent = math.evaluate("10%");
    assert(percent && percent->value.numerator == 1 && percent->value.denominator == 10);
    const auto equation = math.solve("2x + 3 = 11");
    assert(equation && equation->variable == "x" && equation->value.numerator == 4);
    assert(math.verify_linear("2x + 3 = 11", {4, 1}));
    assert(!math.verify_linear("2x + 3 = 11", {3, 1}));
    const auto impossible = math.solve("2x + 3 = 2x + 5");
    assert(!impossible);
    std::cout << "[PASS] marco16 symbolic math\n";
    return 0;
}
