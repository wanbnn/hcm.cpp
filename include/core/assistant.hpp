#pragma once
#include "core/model_bundle.hpp"
#include "core/code_synthesizer.hpp"

namespace cognicore {
enum class AssistantRoute : uint8_t {
    CONVERSATIONAL_GENERAL, CAUSAL_REASONING, CODE_QUERY_OR_EXPLAIN,
    CODE_GENERATE_OR_PATCH, FACT_UPDATE
};
const char* route_name(AssistantRoute route);
class AssistantRouter {
public:
    AssistantRouter();
    AssistantRoute classify(const std::string& input, const ModelBundle& model) const;
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
    explicit AssistantSession(ModelBundle& model) : model_(model) {}
    AssistantReply respond(const std::string& input, CodeLanguage language = CodeLanguage::CPP,
                           const ISurfaceRealizer* mouth = nullptr);
private:
    ModelBundle& model_;
    AssistantRouter router_;
    CodeSynthesizer synthesizer_;
};
std::string json_escape(const std::string& text);
std::string realizer_pair_json(const ParsedTriple& triple);
std::vector<std::string> realizer_document_pairs(const ParsedDocument& document);
} // namespace cognicore
