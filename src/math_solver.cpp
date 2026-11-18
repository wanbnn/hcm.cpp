#include "core/math_solver.hpp"
#include "core/text_parser.hpp"

#include <cctype>
#include <regex>
#include <sstream>
#include <vector>

namespace cognicore {

MathSolver::MathSolver() { rules_.load_default(); }

namespace {

struct Token { bool number = false; int64_t value = 0; char op = 0; };

std::vector<Token> lex_expression(const std::string& text) {
    std::vector<Token> tokens;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (std::isdigit(c)) {
            int64_t value = 0;
            while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
                value = value * 10 + (text[i] - '0');
                ++i;
            }
            tokens.push_back({true, value, 0});
            continue;
        }
        if (text[i] == '+' || text[i] == '-' || text[i] == '*' || text[i] == '/') {
            tokens.push_back({false, 0, text[i]});
            ++i;
            continue;
        }
        ++i;
    }
    return tokens;
}

int64_t apply(int64_t a, int64_t b, char op, bool& ok) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) { ok = false; return 0; }
            return a / b;
        default: ok = false; return 0;
    }
}

} // namespace

std::optional<MathResult> MathSolver::solve(const std::string& text) const {
    std::string normalized = TextParser::utf8_tolower(text);
    // Normalize common arithmetic words to symbols.
    for (const auto& [word, op] : rules_.math_word_operators) {
        const auto needle = " " + word + " ";
        size_t at;
        while ((at = normalized.find(needle)) != std::string::npos)
            normalized.replace(at, needle.size(), " " + op + " ");
    }
    auto tokens = lex_expression(normalized);
    std::vector<int64_t> numbers;
    bool has_operator = false;
    for (const auto& token : tokens) {
        if (token.number) numbers.push_back(token.value);
        else has_operator = true;
    }
    if (numbers.size() < 2) return std::nullopt;

    if (has_operator && tokens.size() >= 3) {
        // Shunting-yard-lite: multiplication/division first, then addition.
        std::vector<Token> reduced;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].number) { reduced.push_back(tokens[i]); continue; }
            if ((tokens[i].op == '*' || tokens[i].op == '/') && !reduced.empty() && reduced.back().number &&
                i + 1 < tokens.size() && tokens[i + 1].number) {
                bool ok = true;
                reduced.back().value = apply(reduced.back().value, tokens[i + 1].value, tokens[i].op, ok);
                if (!ok) return std::nullopt;
                ++i;
                continue;
            }
            reduced.push_back(tokens[i]);
        }
        if (reduced.empty() || !reduced.front().number) return std::nullopt;
        int64_t value = reduced.front().value;
        for (size_t i = 1; i + 1 < reduced.size(); i += 2) {
            if (reduced[i].number || !reduced[i + 1].number) return std::nullopt;
            bool ok = true;
            value = apply(value, reduced[i + 1].value, reduced[i].op, ok);
            if (!ok) return std::nullopt;
        }
        MathResult result;
        result.value = value;
        result.expression = text;
        return result;
    }

    // GSM8K-like word problem: pick the operation from explicit cue words and
    // apply it to the first two quantities mentioned.
    char op = 0;
    auto has = [&](const std::vector<std::string>& cues) { return rules_.contains_any(normalized, cues); };
    if (has(rules_.math_multiply_cues)) op = '*';
    else if (has(rules_.math_subtract_cues)) op = '-';
    else if (has(rules_.math_divide_cues)) op = '/';
    else if (has(rules_.math_add_cues)) op = '+';
    if (!op) return std::nullopt;
    bool ok = true;
    MathResult result;
    result.value = apply(numbers[0], numbers[1], op, ok);
    if (!ok) return std::nullopt;
    result.expression = text;
    return result;
}

} // namespace cognicore
