#include "core/assistant.hpp"
#include "core/morphological_realizer.hpp"
#include <algorithm>
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
AssistantRoute AssistantRouter::classify(const std::string& input, const ModelBundle& model) const {
    const auto text = TextParser::utf8_tolower(input);
    auto has = [&](std::initializer_list<const char*> needles) {
        for (auto needle : needles) if (text.find(needle) != std::string::npos) return true;
        return false;
    };
    uint64_t features = 0;
    if (input.find('?') == std::string::npos && !model.parser.parse_text(input).triples.empty()) features |= 1;
    if (has({"por que", "porque", "causa", "consequ", "why", "explain why"})) features |= 2;
    const bool code = has({"código", "codigo", "função", "funcao", "function", "módulo", "modulo", "code", "api", "workspace"});
    if (has({"gere", "gerar", "crie", "criar", "generate", "write", "refator", "patch", "combine", "componha", "sintet"}) &&
        (code || has({"combine", "componha", "sintet", "patch"}))) features |= 4;
    bool known_symbol = false;
    for (const auto& sym : model.code_graph.symbols()) {
        if (!sym.body_snippet.empty() && text.find(TextParser::utf8_tolower(short_name(sym.name))) != std::string::npos) {
            known_symbol = true; break;
        }
    }
    if (!(features & 4) && (code || known_symbol)) features |= 8;
    return static_cast<AssistantRoute>(router_.predict(&features));
}
AssistantReply AssistantSession::respond(const std::string& input, CodeLanguage language, const ISurfaceRealizer* mouth) {
    AssistantReply reply;
    reply.route = router_.classify(input, model_);
    if (reply.route == AssistantRoute::CODE_GENERATE_OR_PATCH) {
        const auto result = synthesizer_.synthesize(input, language, model_.code_graph, model_.codebook);
        reply.text = result.syntax_valid ? result.code : result.error;
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
        const auto lower = TextParser::utf8_tolower(input);
        if (reply.route == AssistantRoute::CONVERSATIONAL_GENERAL &&
            (lower.starts_with("olá") || lower.starts_with("oi") || lower.starts_with("hello"))) {
            SemanticPlan greeting; greeting.intent = DialogIntent::GREET;
            reply.text = mouth ? mouth->realize(greeting) : model_.realizer.realize(greeting);
        }
    }
    const auto lower = TextParser::utf8_tolower(input);
    if (lower.find("o que eu disse") != std::string::npos || lower.find("what did i say") != std::string::npos) {
        reply.text = model_.episodes.empty() ? "Ainda não há episódios anteriores nesta sessão." : model_.episodes.back().first;
    }
    model_.episodes.emplace_back(input, reply.text);
    if (model_.episodes.size() > 128) model_.episodes.erase(model_.episodes.begin());
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
