#include "core/symbolic_math.hpp"
#include "core/text_parser.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <regex>
#include <sstream>
#include <vector>

namespace cognicore {
namespace {

int64_t gcd64(int64_t a, int64_t b) {
    while (b) { const int64_t next = a % b; a = b; b = next; }
    return a < 0 ? -a : a;
}

Rational normalize(Rational value) {
    if (value.denominator == 0) return value;
    if (value.denominator < 0) { value.numerator = -value.numerator; value.denominator = -value.denominator; }
    const auto divisor = std::gcd(value.numerator < 0 ? -value.numerator : value.numerator, value.denominator);
    if (divisor > 1) { value.numerator /= divisor; value.denominator /= divisor; }
    return value;
}

Rational add(const Rational& a, const Rational& b) {
    return Rational{a.numerator * b.denominator + b.numerator * a.denominator, a.denominator * b.denominator};
}
Rational sub(const Rational& a, const Rational& b) {
    return Rational{a.numerator * b.denominator - b.numerator * a.denominator, a.denominator * b.denominator};
}
Rational mul(const Rational& a, const Rational& b) {
    return Rational{a.numerator * b.numerator, a.denominator * b.denominator};
}
Rational div(const Rational& a, const Rational& b) {
    return b.numerator == 0 ? Rational{0, 0} : Rational{a.numerator * b.denominator, a.denominator * b.numerator};
}

std::string sanitize(const std::string& text, const LanguageRules& rules) {
    std::string out = TextParser::utf8_tolower(text);
    for (const auto& [word, op] : rules.math_word_operators) {
        const auto needle = " " + word + " ";
        size_t at;
        while ((at = out.find(needle)) != std::string::npos) out.replace(at, needle.size(), " " + op + " ");
    }
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
             {"÷", "/"}, {"×", "*"}, {"−", "-"}, {"**", "^"}}) {
        size_t at;
        while ((at = out.find(from)) != std::string::npos) out.replace(at, from.size(), to);
    }
    // Percentages become fractions.
    std::regex percent(R"((\d+(?:\.\d+)?)\s*%)");
    out = std::regex_replace(out, percent, "($1/100)");
    std::string clean;
    for (char ch : out) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || std::isalpha(static_cast<unsigned char>(ch)) ||
            ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == '(' || ch == ')' ||
            ch == '=' || ch == '.' || ch == ' ')
            clean.push_back(ch);
    }
    // Implicit multiplication: 2x, 3(x+1), (x+1)(x-1), 2 x.
    std::string with_mul;
    for (size_t i = 0; i < clean.size(); ++i) {
        if (i && (std::isdigit(static_cast<unsigned char>(clean[i])) || clean[i] == '(' ||
                  std::isalpha(static_cast<unsigned char>(clean[i])))) {
            const char previous = clean[i - 1];
            if (clean[i] == '(' && (std::isdigit(static_cast<unsigned char>(previous)) ||
                                     std::isalpha(static_cast<unsigned char>(previous)) || previous == ')'))
                with_mul.push_back('*');
            else if (std::isalpha(static_cast<unsigned char>(clean[i])) &&
                     (std::isdigit(static_cast<unsigned char>(previous)) || previous == ')'))
                with_mul.push_back('*');
        }
        with_mul.push_back(clean[i]);
    }
    return with_mul;
}

struct Token {
    bool number = false;
    Rational value;
    char op = 0;
    std::string name;
};

std::vector<Token> lex(const std::string& expression) {
    std::vector<Token> tokens;
    for (size_t i = 0; i < expression.size();) {
        const unsigned char ch = static_cast<unsigned char>(expression[i]);
        if (std::isspace(ch)) { ++i; continue; }
        if (std::isdigit(ch)) {
            std::string number;
            while (i < expression.size() && (std::isdigit(static_cast<unsigned char>(expression[i])) || expression[i] == '.'))
                number.push_back(expression[i++]);
            const auto dot = number.find('.');
            if (dot == std::string::npos) {
                tokens.push_back({true, Rational{std::stoll(number), 1}, 0, ""});
            } else {
                const int decimals = static_cast<int>(number.size() - dot - 1);
                int64_t denominator = 1;
                for (int d = 0; d < decimals; ++d) denominator *= 10;
                number.erase(dot, 1);
                tokens.push_back({true, Rational{std::stoll(number), denominator}, 0, ""});
            }
            continue;
        }
        if (std::isalpha(ch)) {
            std::string name;
            while (i < expression.size() && std::isalpha(static_cast<unsigned char>(expression[i])))
                name.push_back(expression[i++]);
            tokens.push_back({false, Rational{}, 0, name});
            continue;
        }
        tokens.push_back({false, Rational{}, expression[i], ""});
        ++i;
    }
    return tokens;
}

class Parser {
public:
    Parser(std::vector<Token> tokens, std::string variable, Rational variable_value)
        : tokens_(std::move(tokens)), variable_(std::move(variable)), variable_value_(variable_value) {}

    bool parse(Rational& value) {
        value = parse_expression();
        return ok_ && position_ == tokens_.size();
    }

private:
    const Token* peek() const { return position_ < tokens_.size() ? &tokens_[position_] : nullptr; }
    const Token* consume() { return position_ < tokens_.size() ? &tokens_[position_++] : nullptr; }

    Rational parse_expression() {
        auto value = parse_term();
        while (ok_ && peek() && !peek()->number && (peek()->op == '+' || peek()->op == '-')) {
            const char op = consume()->op;
            const auto right = parse_term();
            value = op == '+' ? add(value, right) : sub(value, right);
        }
        return normalize(value);
    }
    Rational parse_term() {
        auto value = parse_power();
        while (ok_ && peek() && !peek()->number && (peek()->op == '*' || peek()->op == '/')) {
            const char op = consume()->op;
            const auto right = parse_power();
            if (op == '/') value = div(value, right);
            else value = mul(value, right);
            if (value.denominator == 0) ok_ = false;
        }
        return normalize(value);
    }
    Rational parse_power() {
        auto value = parse_unary();
        if (ok_ && peek() && !peek()->number && peek()->op == '^') {
            consume();
            const auto exponent = parse_power();
            if (exponent.denominator != 1 || std::abs(exponent.numerator) > 12) { ok_ = false; return value; }
            int64_t power = exponent.numerator;
            Rational result{1, 1};
            bool invert = power < 0;
            power = std::abs(power);
            for (int64_t i = 0; i < power; ++i) result = mul(result, value);
            if (invert) result = div(Rational{1, 1}, result);
            value = result;
        }
        return normalize(value);
    }
    Rational parse_unary() {
        if (peek() && !peek()->number && (peek()->op == '+' || peek()->op == '-')) {
            const char op = consume()->op;
            const auto value = parse_unary();
            return op == '-' ? Rational{-value.numerator, value.denominator} : value;
        }
        return parse_primary();
    }
    Rational parse_primary() {
        const auto* token = consume();
        if (!token) { ok_ = false; return {}; }
        if (token->number) return normalize(token->value);
        if (token->op == '(') {
            auto value = parse_expression();
            if (!peek() || peek()->op != ')') ok_ = false;
            else consume();
            return value;
        }
        if (!token->name.empty()) {
            if (token->name == variable_) return variable_value_;
            ok_ = false; return {};
        }
        ok_ = false; return {};
    }

    std::vector<Token> tokens_;
    size_t position_ = 0;
    std::string variable_;
    Rational variable_value_;
    bool ok_ = true;
};

} // namespace

std::string Rational::to_string() const {
    if (denominator == 1) return std::to_string(numerator);
    if (denominator == 0) return "indefinido";
    return std::to_string(numerator) + "/" + std::to_string(denominator);
}

SymbolicMath::SymbolicMath(const LanguageRules& rules) : rules_(rules) {}

std::string SymbolicMath::extract_expression(const std::string& text, const LanguageRules& rules) {
    auto expression = sanitize(text, rules);
    const auto equals = expression.find('=');
    if (equals != std::string::npos) return {};
    expression.erase(std::remove_if(expression.begin(), expression.end(), [](unsigned char ch) {
        return std::isalpha(ch) != 0;
    }), expression.end());
    while (!expression.empty() && (expression.front() == ' ' || expression.front() == '+' ||
                                   expression.front() == '*' || expression.front() == '/'))
        expression.erase(expression.begin());
    while (!expression.empty() && (expression.back() == ' ' || expression.back() == '+' ||
                                   expression.back() == '-' || expression.back() == '*' || expression.back() == '/'))
        expression.pop_back();
    return expression;
}

std::optional<SymbolicResult> SymbolicMath::evaluate(const std::string& text) const {
    auto expression = sanitize(text, rules_);
    if (expression.find('=') != std::string::npos) return std::nullopt;
    // Natural-language wrappers ("quanto é", "calcule") are not variables.
    expression.erase(std::remove_if(expression.begin(), expression.end(), [](unsigned char ch) {
        return std::isalpha(ch) != 0;
    }), expression.end());
    auto tokens = lex(expression);
    if (tokens.empty()) return std::nullopt;
    Parser parser(std::move(tokens), "", {});
    Rational value;
    if (!parser.parse(value)) return std::nullopt;
    SymbolicResult result;
    result.expression = expression;
    result.value = value;
    result.steps = expression + " = " + value.to_string();
    return result;
}

std::optional<SymbolicResult> SymbolicMath::solve(const std::string& text) const {
    auto expression = sanitize(text, rules_);
    // Remove natural-language words but keep single-letter variables (x, y, z).
    std::regex words(R"([a-zA-Z]{2,})");
    expression = std::regex_replace(expression, words, " ");
    const auto equals = expression.find('=');
    if (equals == std::string::npos || expression.find('=', equals + 1) != std::string::npos)
        return std::nullopt;
    const auto left = expression.substr(0, equals);
    const auto right = expression.substr(equals + 1);
    std::string variable;
    for (char ch : expression) if (std::isalpha(static_cast<unsigned char>(ch))) { variable = std::string(1, ch); break; }
    if (variable.empty()) return std::nullopt;
    auto evaluate_at = [&](Rational x) -> std::optional<Rational> {
        const auto tokens = lex(left);
        Parser left_parser(tokens, variable, x);
        Rational lv;
        if (!left_parser.parse(lv)) return std::nullopt;
        const auto right_tokens = lex(right);
        Parser right_parser(right_tokens, variable, x);
        Rational rv;
        if (!right_parser.parse(rv)) return std::nullopt;
        return sub(lv, rv);
    };
    const auto f0 = evaluate_at({0, 1});
    const auto f1 = evaluate_at({1, 1});
    if (!f0 || !f1) return std::nullopt;
    const auto slope = sub(*f1, *f0);
    if (slope.numerator == 0) return std::nullopt;
    const auto solution = div(Rational{-f0->numerator, f0->denominator}, slope);
    if (solution.denominator == 0) return std::nullopt;
    SymbolicResult result;
    result.expression = expression;
    result.variable = variable;
    result.value = normalize(solution);
    result.steps = "f(" + variable + ")=" + left + "-(" + right + "); f(0)=" + f0->to_string() +
                   "; f(1)=" + f1->to_string() + "; " + variable + "=" + result.value.to_string();
    return result;
}

bool SymbolicMath::verify_linear(const std::string& equation, const Rational& x) const {
    auto expression = sanitize(equation, rules_);
    std::regex words(R"([a-zA-Z]{2,})");
    expression = std::regex_replace(expression, words, " ");
    const auto equals = expression.find('=');
    if (equals == std::string::npos) return false;
    std::string variable;
    for (char ch : expression) if (std::isalpha(static_cast<unsigned char>(ch))) { variable = std::string(1, ch); break; }
    if (variable.empty()) return false;
    auto value_at = [&](const std::string& side) -> std::optional<Rational> {
        Parser parser(lex(side), variable, x);
        Rational value;
        if (!parser.parse(value)) return std::nullopt;
        return value;
    };
    const auto left = value_at(expression.substr(0, equals));
    const auto right = value_at(expression.substr(equals + 1));
    if (!left || !right) return false;
    const auto difference = sub(*left, *right);
    return difference.numerator == 0;
}

} // namespace cognicore
