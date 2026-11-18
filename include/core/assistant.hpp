#pragma once
#include "core/model_bundle.hpp"
#include "core/code_synthesizer.hpp"

namespace cognicore {
enum class AssistantRoute : uint8_t {
    CONVERSATIONAL_GENERAL, CAUSAL_REASONING, CODE_QUERY_OR_EXPLAIN,
    CODE_GENERATE_OR_PATCH, FACT_UPDATE, DOCUMENT_GENERATION, MATH_REASONING
};
const char* route_name(AssistantRoute route);
class AssistantRouter {
public:
    AssistantRouter();
    AssistantRoute classify(const std::string& input, const ModelBundle& model, std::string* evidence = nullptr) const;
private:
    TsetlinMachine router_;
};
struct AssistantReply {
    AssistantRoute route = AssistantRoute::CONVERSATIONAL_GENERAL;
    std::string text;
    ModelBundle::InferenceTrace trace;
};
class AssistantSession {
public:
    explicit AssistantSession(ModelBundle& model) : model_(model) {
        synthesizer_.set_rules(model_.language_rules);
        synthesizer_.set_patterns(model_.code_patterns);
    }
    AssistantReply respond(const std::string& input, CodeLanguage language = CodeLanguage::CPP,
                           const ISurfaceRealizer* mouth = nullptr);
    void set_online_learning(bool enabled) { online_learning_ = enabled; }
    bool online_learning() const { return online_learning_; }
private:
    ModelBundle& model_;
    AssistantRouter router_;
    CodeSynthesizer synthesizer_;
    AssistantRoute last_route_ = AssistantRoute::CONVERSATIONAL_GENERAL;
    std::string last_code_task_;
    bool online_learning_ = false;
};
std::string json_escape(const std::string& text);
std::string realizer_pair_json(const ParsedTriple& triple);
std::vector<std::string> realizer_document_pairs(const ParsedDocument& document);
} // namespace cognicore
