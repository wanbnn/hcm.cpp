#include "core/content_planner.hpp"

#include "core/text_parser.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace cognicore {
namespace {

std::string lower(std::string value) {
    return TextParser::utf8_tolower(value);
}

bool has_any(const std::string& text, const std::vector<std::string>& cues) {
    for (const auto& cue : cues) {
        if (!cue.empty() && text.find(lower(cue)) != std::string::npos) return true;
    }
    return false;
}

std::string unique_key(const CausalStep& step) {
    return lower(step.src) + "|" + lower(step.rel) + "|" + lower(step.dst);
}

void add_unique(std::vector<CausalStep>& values, std::set<std::string>& seen,
                const CausalStep& step) {
    if (step.src.empty() || step.dst.empty()) return;
    if (!seen.insert(unique_key(step)).second) return;
    values.push_back(step);
}

CausalPath single(const CausalStep& step) {
    CausalPath path;
    path.steps.push_back(step);
    path.score = step.confidence;
    return path;
}

std::string render_title(const LanguageRules& rules, const std::string& topic) {
    const auto tmpl = rules.document_title.empty() ? std::string("{topic}") : rules.document_title.front();
    auto title = tmpl;
    const auto at = title.find("{topic}");
    if (at != std::string::npos) title.replace(at, 7, topic);
    return title;
}

size_t requested_count(const TextParser& parser, const std::string& input,
                       const std::vector<std::string>& nouns) {
    const auto words = parser.tokenize(input);
    for (size_t i = 0; i + 1 < words.size(); ++i) {
        if (std::find(nouns.begin(), nouns.end(), words[i + 1]) == nouns.end()) continue;
        if (!words[i].empty() && std::all_of(words[i].begin(), words[i].end(),
                                             [](unsigned char c) { return std::isdigit(c); })) {
            const auto value = std::stoul(words[i]);
            if (value > 0) return std::min<size_t>(value, 8);
        }
    }
    return 0;
}

} // namespace

const char* goal_name(CommunicativeGoal goal) noexcept {
    switch (goal) {
        case CommunicativeGoal::GENERAL: return "GENERAL";
        case CommunicativeGoal::DEFINITION: return "DEFINITION";
        case CommunicativeGoal::MECHANISM: return "MECHANISM";
        case CommunicativeGoal::CAUSE: return "CAUSE";
        case CommunicativeGoal::COMPARISON: return "COMPARISON";
        case CommunicativeGoal::PROCEDURE: return "PROCEDURE";
        case CommunicativeGoal::ESSAY: return "ESSAY";
    }
    return "GENERAL";
}

const char* shape_name(ResponseShape shape) noexcept {
    switch (shape) {
        case ResponseShape::BRIEF: return "BRIEF";
        case ResponseShape::EXPLANATORY: return "EXPLANATORY";
        case ResponseShape::DOCUMENT: return "DOCUMENT";
    }
    return "BRIEF";
}

ContentPlan ContentPlanner::plan(const std::string& input,
                                 const CausalGraph& graph,
                                 const Codebook& codebook,
                                 const LanguageRules& rules,
                                 const TermResolver* terms,
                                 const ResponseConstraints& constraints,
                                 const TextParser* parser_override) const {
    ContentPlan plan;
    TextParser local_parser;
    const TextParser& parser = parser_override ? *parser_override : local_parser;
    const auto lowered = lower(input);
    const auto tokens = parser.extract_content_tokens(input);

    // Topic resolution: longest graph node mentioned in the input, then a
    // canonical alias, then the first content token.
    std::string topic;
    for (const auto& node : graph.nodes()) {
        if (node.name.size() < 2) continue;
        if (lowered.find(lower(node.name)) != std::string::npos &&
            (topic.empty() || node.name.size() > topic.size())) {
            topic = node.name;
        }
    }
    if (topic.empty() && terms) {
        for (const auto& token : tokens) {
            const auto resolved = terms->resolve(token);
            if (graph.find_node(resolved) != UINT32_MAX) { topic = resolved; break; }
        }
    }
    if (topic.empty() && !tokens.empty()) {
        const auto resolved = terms ? terms->resolve(tokens.front()) : tokens.front();
        if (graph.find_node(resolved) != UINT32_MAX) topic = resolved;
    }
    plan.answer.topic = topic;
    if (topic.empty()) {
        plan.rationale = "sem tópico ancorado no grafo";
        return plan;
    }

    std::set<std::string> seen_facts;
    const auto direct = graph.query_direct(topic, 48);
    for (const auto& step : direct)
        add_unique(plan.answer.facts, seen_facts, step);

    // Category buckets are independent of the global fact list; a fact can be
    // both an answer fact and a "function" used by the rhetorical schema.
    for (const auto& step : plan.answer.facts) {
        if (step.type == RelationType::TAXONOMIC) plan.answer.definitions.push_back(step);
        if (step.type == RelationType::PROPERTY) plan.answer.properties.push_back(step);
        if (step.type == RelationType::HAS_PART || step.type == RelationType::PART_OF)
            plan.answer.parts.push_back(step);
        if (step.type == RelationType::FUNCTION || step.type == RelationType::ENABLES)
            plan.answer.functions.push_back(step);
        if (step.type == RelationType::SEQUENCE || step.type == RelationType::PRECEDES)
            plan.answer.sequence.push_back(step);
        if (step.type == RelationType::CAUSAL || step.type == RelationType::ACTION) {
            if (lower(step.dst) == lower(topic)) plan.answer.causes.push_back(step);
            else plan.answer.functions.push_back(step);
        }
    }
    plan.answer.paths = graph.find_paths({topic}, 3, 8, &codebook);
    std::set<std::string> relations;
    for (const auto& step : plan.answer.facts) relations.insert(lower(step.rel));
    plan.answer.distinct_relations = relations.size();

    // Communicative goal.
    if (has_any(lowered, rules.document_cues)) plan.goal = CommunicativeGoal::ESSAY;
    else if (has_any(lowered, {"como funciona", "funcionamento", "mecanismo", "how does", "how it works"}))
        plan.goal = CommunicativeGoal::MECHANISM;
    else if (has_any(lowered, {"o que é", "o que e", "defina", "definição", "definicao", "what is"}))
        plan.goal = CommunicativeGoal::DEFINITION;
    else if (has_any(lowered, {"compare", "comparação", "comparacao", "diferença", "diferenca"}))
        plan.goal = CommunicativeGoal::COMPARISON;
    else if (has_any(lowered, {"passo a passo", "instruções", "instrucoes", "como fazer"}))
        plan.goal = CommunicativeGoal::PROCEDURE;
    else plan.goal = CommunicativeGoal::CAUSE;

    const bool brief_cue = constraints.force_brief ||
        has_any(lowered, {"resuma", "resumo", "em uma frase", "curto", "breve", "só o nome", "so o nome"});
    const bool document_cue = constraints.force_document ||
        has_any(lowered, {"em detalhes", "detalhado", "texto longo", "artigo", "redação", "redacao",
                          "relatório", "relatorio", "dissertação", "dissertacao"});
    size_t sections = constraints.sections ? constraints.sections
        : requested_count(parser, lowered, {"parágrafo", "paragrafo", "parágrafos", "paragrafos",
                                            "seção", "secao", "seções", "secoes"});
    if (sections == 0) sections = document_cue ? 4 : 0;

    const size_t facts = plan.answer.facts.size();
    const size_t parts = plan.answer.parts.size();
    const size_t functions = plan.answer.functions.size();
    const size_t sequence = plan.answer.sequence.size();
    if (brief_cue) {
        plan.shape = ResponseShape::BRIEF;
    } else if (constraints.force_document || plan.goal == CommunicativeGoal::ESSAY) {
        plan.shape = ResponseShape::DOCUMENT;
    } else if (plan.goal == CommunicativeGoal::MECHANISM) {
        plan.shape = (parts >= 2 || sequence >= 2 || facts >= 6)
            ? ResponseShape::DOCUMENT : (facts >= 2 || functions >= 2
                ? ResponseShape::EXPLANATORY : ResponseShape::BRIEF);
    } else if (plan.goal == CommunicativeGoal::DEFINITION) {
        plan.shape = (facts <= 1 && parts == 0) ? ResponseShape::BRIEF : ResponseShape::EXPLANATORY;
    } else if (plan.goal == CommunicativeGoal::COMPARISON || plan.goal == CommunicativeGoal::PROCEDURE) {
        plan.shape = facts >= 2 ? ResponseShape::EXPLANATORY : ResponseShape::BRIEF;
    } else {
        plan.shape = facts <= 1 ? ResponseShape::BRIEF : (facts >= 5 ? ResponseShape::DOCUMENT
                                                                    : ResponseShape::EXPLANATORY);
    }

    // Essay/document uses the existing generic planner for broad coverage.
    if (plan.shape == ResponseShape::DOCUMENT && plan.goal == CommunicativeGoal::ESSAY) {
        DocumentPlanner planner;
        plan.document = planner.plan(topic, graph, codebook, rules,
                                     sections ? sections : 5,
                                     constraints.paragraphs ? constraints.paragraphs : 2,
                                     terms, parser_override);
        plan.sections = plan.document.sections.size();
        plan.paragraphs = plan.sections ? plan.document.sections.front().paragraphs.size() : 0;
        std::ostringstream why;
        why << "goal=" << goal_name(plan.goal) << " shape=" << shape_name(plan.shape)
            << " fatos=" << facts << " seções=" << plan.sections;
        plan.rationale = why.str();
        return plan;
    }

    // Build rhetorical moves from the answer graph.
    std::set<std::string> used_moves;
    auto add_move = [&](SectionRole role, const std::string& heading,
                        std::vector<CausalStep> steps) {
        std::vector<CausalStep> fresh;
        for (const auto& step : steps)
            if (used_moves.insert(unique_key(step)).second) fresh.push_back(step);
        if (fresh.empty()) return;
        steps = fresh;
        DocumentSection section;
        section.role = role;
        section.heading = heading;
        DocumentParagraph paragraph;
        paragraph.topic = topic;
        for (const auto& step : steps) {
            DocumentSentence sentence;
            sentence.plan.intent = role == SectionRole::CAUSES ? DialogIntent::EXPLAIN_CAUSE
                                                              : DialogIntent::ANSWER_FACT;
            sentence.plan.language = rules.language;
            sentence.plan.paths.push_back(single(step));
            paragraph.sentences.push_back(std::move(sentence));
        }
        section.paragraphs.push_back(std::move(paragraph));
        plan.document.sections.push_back(std::move(section));
    };

    plan.document.topic = topic;
    plan.document.title = render_title(rules, topic);
    if (plan.goal == CommunicativeGoal::MECHANISM) {
        add_move(SectionRole::INTRODUCTION, "Definição", plan.answer.definitions);
        add_move(SectionRole::INTRODUCTION, "Partes", plan.answer.parts);
        add_move(SectionRole::CAUSES, "Funcionamento", plan.answer.functions);
        add_move(SectionRole::CONSEQUENCES, "Sequência", plan.answer.sequence);
        add_move(SectionRole::CONCLUSION, "Conclusão",
                 plan.answer.functions.empty() ? plan.answer.definitions
                     : std::vector<CausalStep>{plan.answer.functions.back()});
    } else if (plan.goal == CommunicativeGoal::DEFINITION) {
        add_move(SectionRole::INTRODUCTION, "Definição", plan.answer.definitions);
        add_move(SectionRole::INTRODUCTION, "Propriedades", plan.answer.properties);
        add_move(SectionRole::CAUSES, "Funcionamento", plan.answer.functions);
    } else {
        add_move(SectionRole::CAUSES, "Causas", plan.answer.causes);
        add_move(SectionRole::CONSEQUENCES, "Consequências", plan.answer.functions);
        add_move(SectionRole::CONCLUSION, "Conclusão",
                 plan.answer.functions.empty() ? plan.answer.definitions
                     : std::vector<CausalStep>{plan.answer.functions.back()});
    }
    if (plan.shape == ResponseShape::EXPLANATORY && plan.document.sections.size() > 2)
        plan.document.sections.resize(2);
    if (sections && plan.document.sections.size() > sections)
        plan.document.sections.resize(sections);
    plan.sections = plan.document.sections.size();
    plan.paragraphs = plan.sections ? plan.document.sections.front().paragraphs.size() : 0;

    std::ostringstream why;
    why << "goal=" << goal_name(plan.goal) << " shape=" << shape_name(plan.shape)
        << " fatos=" << facts << " partes=" << parts << " funções=" << functions
        << " sequência=" << sequence << " seções=" << plan.sections;
    plan.rationale = why.str();
    return plan;
}

} // namespace cognicore
