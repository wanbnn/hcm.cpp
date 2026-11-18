#include "core/assistant.hpp"
#include "core/math_solver.hpp"
#include "core/morphological_realizer.hpp"
#include "core/symbolic_math.hpp"
#include <algorithm>
#include <cctype>
#include <optional>
#include <regex>
#include <sstream>
#include <iomanip>

namespace cognicore {
namespace {
TsetlinConfig router_config() {
    TsetlinConfig cfg;
    cfg.num_features = 8; cfg.num_classes = 5; cfg.clauses_per_class = 40;
    cfg.n_states = 30; cfg.margin_T = 15; cfg.seed = 8123;
    return cfg;
}
std::string short_name(const std::string& name) {
    auto at = name.rfind("::");
    if (at != std::string::npos) return name.substr(at + 2);
    at = name.rfind('.');
    return at == std::string::npos ? name : name.substr(at + 1);
}

CodeLanguage detect_requested_language(const std::string& lower, CodeLanguage fallback,
                                        const LanguageRules& rules) {
    const auto tag = rules.requested_language(lower, "");
    if (tag == "python") return CodeLanguage::PYTHON;
    if (tag == "javascript") return CodeLanguage::JAVASCRIPT;
    if (tag == "typescript") return CodeLanguage::TYPESCRIPT;
    if (tag == "cpp") return CodeLanguage::CPP;
    return fallback;
}

std::optional<std::string> arithmetic_expression(const std::string& text, const LanguageRules& rules) {
    std::string normalized = TextParser::utf8_tolower(text);
    for (const auto& [word, op] : rules.math_word_operators) {
        const auto needle = " " + word + " ";
        size_t at;
        while ((at = normalized.find(needle)) != std::string::npos)
            normalized.replace(at, needle.size(), " " + op + " ");
    }
    std::string expression;
    for (char ch : normalized) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '+' || ch == '-' ||
            ch == '*' || ch == '/' || ch == '(' || ch == ')' || ch == '.' || ch == ' ')
            expression.push_back(ch);
    }
    auto trim = [](std::string value) {
        while (!value.empty() && value.front() == ' ') value.erase(value.begin());
        while (!value.empty() && value.back() == ' ') value.pop_back();
        return value;
    };
    expression = trim(expression);
    while (!expression.empty() && (expression.front() == ' ' || expression.front() == '+' ||
                                   expression.front() == '*' || expression.front() == '/'))
        expression.erase(expression.begin());
    while (!expression.empty() && (expression.back() == ' ' || expression.back() == '+' ||
                                   expression.back() == '-' || expression.back() == '*' || expression.back() == '/'))
        expression.pop_back();
    const auto operators = expression.find_first_of("+-*/");
    const auto digits = expression.find_first_of("0123456789");
    if (operators == std::string::npos || digits == std::string::npos) return std::nullopt;
    std::vector<int64_t> numbers;
    std::regex number_pattern(R"(\d+)");
    for (std::sregex_iterator it(expression.begin(), expression.end(), number_pattern), end; it != end; ++it)
        numbers.push_back(std::stoll(it->str()));
    if (numbers.size() < 2) return std::nullopt;
    return expression;
}

std::optional<std::string> arithmetic_script(const std::string& text, CodeLanguage lang, const LanguageRules& rules) {
    const auto expression = arithmetic_expression(text, rules);
    if (!expression) return std::nullopt;
    std::ostringstream out;
    if (lang == CodeLanguage::PYTHON) {
        out << "def synthesized_script():\n"
            << "    return " << *expression << "\n\n"
            << "if __name__ == \"__main__\":\n"
            << "    print(synthesized_script())\n";
    } else if (lang == CodeLanguage::JAVASCRIPT || lang == CodeLanguage::TYPESCRIPT) {
        out << "function synthesizedScript() {\n"
            << "  return " << *expression << ";\n"
            << "}\n\n"
            << "console.log(synthesizedScript());\n";
    } else {
        out << "#include <iostream>\n\n"
            << "int main() {\n"
            << "    std::cout << (" << *expression << ") << \"\\n\";\n"
            << "    return 0;\n"
            << "}\n";
    }
    const auto code = out.str();
    if (!TreeSitterCodeParser::validate_syntax(code, lang)) return std::nullopt;
    return code;
}

bool code_followup(const std::string& lower, const LanguageRules& rules) {
    return rules.is_code_followup(lower);
}
}
const char* route_name(AssistantRoute route) {
    static const char* names[] = {"CONVERSATIONAL_GENERAL", "CAUSAL_REASONING", "CODE_QUERY_OR_EXPLAIN",
                                 "CODE_GENERATE_OR_PATCH", "FACT_UPDATE"};
    return names[static_cast<size_t>(route)];
}
AssistantRouter::AssistantRouter() : router_(router_config()) {
    // Train the symbolic router's five-way decision from explicit context bits.
    // No language model is involved; keyword and graph evidence are input features.
    std::vector<std::vector<uint64_t>> x;
    std::vector<size_t> y;
    for (uint64_t mask = 0; mask < 16; ++mask) {
        size_t label = mask & 4 ? 3 : mask & 8 ? 2 : mask & 1 ? 4 : mask & 2 ? 1 : 0;
        x.push_back({mask}); y.push_back(label);
    }
    router_.fit(x, y, 500);
}
AssistantRoute AssistantRouter::classify(const std::string& input, const ModelBundle& model, std::string* evidence) const {
    const auto text = TextParser::utf8_tolower(input);
    uint64_t features = 0;
    const auto& rules = model.language_rules;
    const bool hypothetical = rules.is_hypothetical(text);
    const bool leading_rule = rules.is_leading_rule(text);
    const bool capacity = parse_memory_quantity(text).has_value() &&
        rules.contains_any(text, rules.resource_cues);
    const bool conditional_rule = leading_rule &&
        rules.contains_any(text, rules.consequence_markers);
    const auto parsed = model.parser.parse_text(input);
    const bool has_rule = !parsed.conditionals.empty();
    const bool question_form = input.find('?') != std::string::npos ||
        rules.starts_with_any(text, rules.question_prefixes);
    const bool correction_turn = rules.is_instruction_or_correction(text);
    if (!question_form && !correction_turn && (!hypothetical || conditional_rule) &&
        (!parsed.triples.empty() || capacity || conditional_rule || has_rule)) features |= 1;
    if (rules.contains_any(text, rules.causal_cues) || hypothetical || has_rule) features |= 2;
    const bool code = rules.contains_any(text, rules.code_cues);
    const bool generation = rules.contains_any(text, rules.generation_cues);
    if (generation && (code || rules.contains_any(text, rules.code_generation_markers)))
        features |= 4;
    bool known_symbol = false;
    for (const auto& sym : model.code_graph.symbols()) {
        if (!sym.body_snippet.empty() && text.find(TextParser::utf8_tolower(short_name(sym.name))) != std::string::npos) {
            known_symbol = true; break;
        }
    }
    if (!(features & 4) && (code || known_symbol)) features |= 8;
    const auto votes = router_.score_classes(&features);
    if (evidence) {
        std::ostringstream out;
        out << "features=" << features << " [bit0=fato, bit1=causal, bit2=geração, bit3=consulta_código] votos=";
        for (auto v : votes) out << v << ",";
        *evidence = out.str();
    }
    return static_cast<AssistantRoute>(std::max_element(votes.begin(), votes.end()) - votes.begin());
}
AssistantReply AssistantSession::respond(const std::string& input, CodeLanguage language, const ISurfaceRealizer* mouth) {
    AssistantReply reply;
    std::string routing;
    reply.route = router_.classify(input, model_, &routing);
    const auto lower = TextParser::utf8_tolower(input);
    if (reply.route == AssistantRoute::CONVERSATIONAL_GENERAL &&
        last_route_ == AssistantRoute::CODE_GENERATE_OR_PATCH && code_followup(lower, model_.language_rules))
        reply.route = AssistantRoute::CODE_GENERATE_OR_PATCH;
    std::string effective_input = input;
    if (reply.route == AssistantRoute::CODE_GENERATE_OR_PATCH && code_followup(lower, model_.language_rules) &&
        !last_code_task_.empty())
        effective_input = last_code_task_;
    const bool math_cue = model_.language_rules.contains_any(lower, model_.language_rules.math_cues);
    const bool resource_context = model_.language_rules.contains_any(lower, model_.language_rules.resource_cues);
    std::optional<std::string> math_answer;
    if (math_cue && !resource_context &&
        (reply.route == AssistantRoute::CONVERSATIONAL_GENERAL || reply.route == AssistantRoute::FACT_UPDATE)) {
        const SymbolicMath symbolic(model_.language_rules);
        if (const auto equation = symbolic.solve(input)) {
            math_answer = "Resultado: " + equation->variable + " = " + equation->value.to_string() + ".";
            reply.trace.decision_override = "MATH: equação linear simbólica";
        } else if (const auto evaluated = symbolic.evaluate(input)) {
            math_answer = "Resultado: " + evaluated->value.to_string() + ".";
            reply.trace.decision_override = "MATH: aritmética racional simbólica";
        } else if (const auto solved = MathSolver{}.solve(input)) {
            math_answer = "Resultado: " + std::to_string(solved->value) + ".";
            reply.trace.decision_override = "MATH: aritmética inteira simbólica";
        }
    }
    if (math_answer) {
        reply.text = *math_answer;
    } else if (reply.route == AssistantRoute::CODE_GENERATE_OR_PATCH) {
        const auto requested_language = detect_requested_language(TextParser::utf8_tolower(effective_input), language, model_.language_rules);
        const auto result = synthesizer_.synthesize(effective_input, requested_language,
                                                    model_.code_graph, model_.codebook);
        std::ostringstream evidence;
        if (result.syntax_valid) {
            reply.text = result.code;
        } else if (auto script = arithmetic_script(effective_input, requested_language, model_.language_rules)) {
            reply.text = *script;
            reply.trace.decision_override = "CODE: script aritmético gerado sem APIs do workspace";
        } else {
            reply.text = result.error;
        }
        evidence << "padrão Tsetlin=" << unsigned(result.selected_pattern) << " votos=";
        for (auto v : result.pattern_votes) evidence << v << ",";
        evidence << " cláusulas=";
        for (auto id : result.fired_clauses) evidence << id << ",";
        evidence << " símbolos do grafo=";
        for (size_t i = 0; i < result.symbol_ids.size(); ++i)
            evidence << result.symbol_ids[i] << ":" << result.used_symbols[i] << ",";
        evidence << " Tree-sitter=" << (!reply.text.empty() ? "sem ERROR/MISSING" : "sem código validado")
                 << "; compatibilidade limitada às assinaturas indexadas; sem prova de tipos do compilador";
        reply.trace.code_evidence = evidence.str();
        last_code_task_ = effective_input;
    } else if (reply.route == AssistantRoute::CODE_QUERY_OR_EXPLAIN) {
        const auto lower = TextParser::utf8_tolower(input);
        std::ostringstream out;
        for (const auto& sym : model_.code_graph.symbols()) {
            if (sym.body_snippet.empty() || lower.find(TextParser::utf8_tolower(short_name(sym.name))) == std::string::npos) continue;
            out << sym.name << " (" << sym.file_path << ")\n  Retorno: " << sym.return_type << "; parâmetros: ";
            for (size_t i = 0; i < sym.param_types.size(); ++i)
                out << (i ? ", " : "") << sym.param_types[i] << " " << sym.param_names[i];
            out << "\n  Chamadores:";
            for (const auto& caller : model_.code_graph.find_callers(sym.name)) out << " " << caller.name;
            out << "\n  Chamadas:";
            for (const auto& edge : model_.code_graph.edges())
                if (edge.src == sym.id && edge.type == CodeEdgeType::CALLS) out << " " << model_.code_graph.symbols()[edge.dst].name;
            out << "\n";
        }
        reply.text = out.str().empty() ? "Não encontrei esse símbolo no índice do workspace." : out.str();
    } else {
        reply.trace = model_.infer_and_respond(input, reply.route == AssistantRoute::FACT_UPDATE, mouth);
        reply.text = reply.trace.response;

    }
    reply.trace.router_evidence = routing;
    model_.episodes.emplace_back(input, reply.text);
    if (model_.episodes.size() > 128) model_.episodes.erase(model_.episodes.begin());
    last_route_ = reply.route;
    return reply;
}
std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (unsigned char ch : text) {
        switch (ch) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(ch) << std::dec;
                else out << ch;
        }
    }
    return out.str();
}
std::string realizer_pair_json(const ParsedTriple& t) {
    SemanticPlan plan;
    plan.paths.push_back({{{t.subject, t.relation, t.object, t.type, t.confidence, t.negated, t.tense,
                           t.modality, t.subj_gender, t.subj_number, t.obj_gender, t.obj_number}}, 1});
    return "{\"plan\":\"" + json_escape(serialize_semantic_plan(plan)) +
           "\",\"text\":\"" + json_escape(t.source_text) + "\"}";
}
std::vector<std::string> realizer_document_pairs(const ParsedDocument& document) {
    std::vector<std::string> sources;
    std::vector<SemanticPlan> plans;
    for (const auto& t : document.triples) {
        if (t.source_text.empty()) continue;
        auto it = std::find(sources.begin(), sources.end(), t.source_text);
        size_t i = size_t(it - sources.begin());
        if (it == sources.end()) { sources.push_back(t.source_text); plans.emplace_back(); plans.back().paths.push_back({{}, 1}); }
        plans[i].paths[0].steps.push_back({t.subject, t.relation, t.object, t.type, t.confidence, t.negated,
            t.tense, t.modality, t.subj_gender, t.subj_number, t.obj_gender, t.obj_number});
    }
    std::vector<std::string> pairs;
    for (size_t i = 0; i < sources.size(); ++i)
        pairs.push_back("{\"plan\":\"" + json_escape(serialize_semantic_plan(plans[i])) +
                        "\",\"text\":\"" + json_escape(sources[i]) + "\"}");
    return pairs;
}

} // namespace cognicore
