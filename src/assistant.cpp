#include "core/assistant.hpp"
#include "core/document_planner.hpp"
#include "core/math_solver.hpp"
#include "core/morphological_realizer.hpp"
#include "core/symbolic_math.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
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


std::string document_topic(const std::string& lower, const LanguageRules& rules) {
    for (const auto& marker : rules.document_topic_markers) {
        const auto at = lower.find(" " + marker + " ");
        if (at == std::string::npos) continue;
        auto topic = lower.substr(at + marker.size() + 2);
        while (!topic.empty() && (topic.back() == '.' || topic.back() == '?' || topic.back() == ' ')) topic.pop_back();
        if (!topic.empty()) return topic;
    }
    for (const auto& prefix : rules.instruction_prefixes) {
        if (!lower.starts_with(prefix)) continue;
        auto topic = lower.substr(prefix.size());
        while (!topic.empty() && (topic.back() == '.' || topic.back() == '?' || topic.back() == ' ')) topic.pop_back();
        return topic;
    }
    return {};
}

bool code_followup(const std::string& lower, const LanguageRules& rules) {
    return rules.is_code_followup(lower);
}

bool contains_word_any(const std::string& text, const std::vector<std::string>& cues) {
    for (const auto& cue : cues) {
        if (cue.empty()) continue;
        if (cue.find(' ') != std::string::npos) {
            if (text.find(cue) != std::string::npos) return true;
            continue;
        }
        size_t at = 0;
        while ((at = text.find(cue, at)) != std::string::npos) {
            const bool left_ok = at == 0 || !std::isalnum(static_cast<unsigned char>(text[at - 1]));
            const size_t end = at + cue.size();
            const bool right_ok = end >= text.size() || !std::isalnum(static_cast<unsigned char>(text[end]));
            if (left_ok && right_ok) return true;
            at = end;
        }
    }
    return false;
}
}
const char* route_name(AssistantRoute route) {
    static const char* names[] = {"CONVERSATIONAL_GENERAL", "CAUSAL_REASONING", "CODE_QUERY_OR_EXPLAIN",
                                 "CODE_GENERATE_OR_PATCH", "FACT_UPDATE", "DOCUMENT_GENERATION",
                                 "MATH_REASONING"};
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
    const auto& rules = model_.language_rules;
    // A factual question ("qual é a capital da França?") must not be routed
    // to code-query merely because of an over-eager router feature.
    if (reply.route == AssistantRoute::CODE_QUERY_OR_EXPLAIN &&
        !contains_word_any(lower, rules.code_cues) &&
        !contains_word_any(lower, rules.generation_cues))
        reply.route = AssistantRoute::CONVERSATIONAL_GENERAL;
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
    if (math_cue && !resource_context && reply.route != AssistantRoute::CODE_GENERATE_OR_PATCH &&
        reply.route != AssistantRoute::CODE_QUERY_OR_EXPLAIN) {
        const SymbolicMath symbolic(model_.language_rules);
        if (const auto equation = symbolic.solve(input)) {
            math_answer = rules.render_template(rules.math_equation_templates, rules.language,
                                                equation->variable, {{"variable", equation->variable},
                                                                     {"value", equation->value.to_string()}});
            reply.trace.decision_override = "MATH: equação linear simbólica";
        } else if (const auto evaluated = symbolic.evaluate(input)) {
            math_answer = rules.render_template(rules.math_value_templates, rules.language,
                                                evaluated->value.to_string(),
                                                {{"value", evaluated->value.to_string()}});
            reply.trace.decision_override = "MATH: aritmética racional simbólica";
        } else if (const auto solved = MathSolver{}.solve(input)) {
            math_answer = rules.render_template(rules.math_value_templates, rules.language,
                                                std::to_string(solved->value),
                                                {{"value", std::to_string(solved->value)}});
            reply.trace.decision_override = "MATH: aritmética inteira simbólica";
        }
    }
    const bool document_request = !math_answer && !resource_context &&
        rules.contains_any(lower, rules.document_cues) &&
        !rules.contains_any(lower, rules.code_cues);
    ResponseConstraints constraints;
    constraints.force_document = document_request;
    if (shape_override_) {
        if (*shape_override_ == ResponseShape::BRIEF) constraints.force_brief = true;
        if (*shape_override_ == ResponseShape::DOCUMENT) constraints.force_document = true;
    }
    ContentPlan content_plan;
    const bool content_candidate = !math_answer && !resource_context &&
        reply.route != AssistantRoute::CODE_GENERATE_OR_PATCH &&
        reply.route != AssistantRoute::CODE_QUERY_OR_EXPLAIN;
    if (content_candidate) {
        ContentPlanner content;
        content_plan = content.plan(input, model_.graph, model_.codebook, rules,
                                    &model_.terms, constraints, &model_.parser);
    }
    const bool question_like = input.find('?') != std::string::npos ||
        lower.starts_with("qual ") || lower.starts_with("quais ") ||
        lower.starts_with("quem ") || lower.starts_with("onde ") ||
        lower.starts_with("quando ") || lower.starts_with("o que ") ||
        lower.starts_with("como ");
    const bool use_content = content_candidate && !content_plan.document.sections.empty() &&
        (content_plan.shape != ResponseShape::BRIEF || document_request || question_like);
    MorphologicalRealizer fallback(rules);
    const ISurfaceRealizer& surface = mouth ? *mouth : static_cast<const ISurfaceRealizer&>(fallback);
    if (math_answer) {
        reply.text = *math_answer;
        reply.route = AssistantRoute::MATH_REASONING;
    } else if (use_content) {
        reply.route = content_plan.shape == ResponseShape::DOCUMENT
            ? AssistantRoute::DOCUMENT_GENERATION : AssistantRoute::CAUSAL_REASONING;
        RenderOptions options;
        options.title = content_plan.shape == ResponseShape::DOCUMENT;
        options.headings = content_plan.shape == ResponseShape::DOCUMENT;
        options.natural = true;
        options.discourse = &model_.discourse;
        DocumentPlanner planner;
        reply.text = planner.realize(content_plan.document, surface, rules, options);
        reply.trace.decision_override = "CONTENT: " + content_plan.rationale;
    } else if (document_request) {
        reply.route = AssistantRoute::DOCUMENT_GENERATION;
        const auto topic = document_topic(lower, rules);
        reply.text = rules.render_template(rules.document_empty_templates, rules.language, topic,
                                           {{"topic", topic}});
        reply.trace.decision_override = "DOCUMENT: sem conteúdo suficiente";
    } else if (reply.route == AssistantRoute::CODE_GENERATE_OR_PATCH) {
        const auto requested_language = detect_requested_language(TextParser::utf8_tolower(effective_input), language, model_.language_rules);
        const auto result = synthesizer_.synthesize(effective_input, requested_language,
                                                    model_.code_graph, model_.codebook);
        std::ostringstream evidence;
        if (result.syntax_valid) {
            reply.text = result.code;
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
        reply.text = out.str().empty()
            ? rules.render_template(rules.api_not_found_templates, rules.language, input, {{"symbol", input}})
            : out.str();
    } else {
        reply.trace = model_.infer_and_respond(input, online_learning_, mouth);
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
