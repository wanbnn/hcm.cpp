#pragma once

#include "core/causal_graph.hpp"
#include "core/document_planner.hpp"
#include "core/language_rules.hpp"
#include "core/term_resolver.hpp"
#include "core/text_parser.hpp"

#include <string>
#include <vector>

namespace cognicore {

enum class CommunicativeGoal : uint8_t {
    GENERAL = 0,
    DEFINITION = 1,
    MECHANISM = 2,
    CAUSE = 3,
    COMPARISON = 4,
    PROCEDURE = 5,
    ESSAY = 6
};

enum class ResponseShape : uint8_t {
    BRIEF = 0,
    EXPLANATORY = 1,
    DOCUMENT = 2
};

struct ResponseConstraints {
    bool force_brief = false;
    bool force_document = false;
    size_t sections = 0;   // 0 = decide from content
    size_t paragraphs = 0; // 0 = decide from content
};

struct AnswerGraph {
    std::string topic;
    std::vector<CausalStep> facts;
    std::vector<CausalStep> definitions;
    std::vector<CausalStep> properties;
    std::vector<CausalStep> parts;
    std::vector<CausalStep> functions;
    std::vector<CausalStep> sequence;
    std::vector<CausalStep> causes;
    std::vector<CausalStep> consequences;
    std::vector<CausalPath> paths;
    size_t distinct_relations = 0;
};

struct ContentPlan {
    CommunicativeGoal goal = CommunicativeGoal::GENERAL;
    ResponseShape shape = ResponseShape::BRIEF;
    AnswerGraph answer;
    DocumentPlan document;
    size_t sections = 0;
    size_t paragraphs = 0;
    std::string rationale;
};

/// Content-first planner: resolves the topic, extracts the answer subgraph,
/// selects a rhetorical goal and only then decides how long the answer should
/// be. The surface layer is the existing DocumentPlanner.
class ContentPlanner {
public:
    ContentPlan plan(const std::string& input,
                     const CausalGraph& graph,
                     const Codebook& codebook,
                     const LanguageRules& rules,
                     const TermResolver* terms = nullptr,
                     const ResponseConstraints& constraints = {},
                     const TextParser* parser_override = nullptr) const;
};

const char* goal_name(CommunicativeGoal goal) noexcept;
const char* shape_name(ResponseShape shape) noexcept;

} // namespace cognicore
