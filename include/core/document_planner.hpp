#pragma once

#include "core/causal_graph.hpp"
#include "core/language_rules.hpp"
#include "core/realizer.hpp"
#include "core/term_resolver.hpp"
#include "core/text_parser.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cognicore {

enum class SectionRole : uint8_t {
    INTRODUCTION = 0,
    CAUSES = 1,
    CONSEQUENCES = 2,
    ALTERNATIVES = 3,
    CONCLUSION = 4
};

struct DocumentSentence {
    SemanticPlan plan;
    std::string text;
};

struct DocumentParagraph {
    std::string topic;
    std::vector<DocumentSentence> sentences;
};

struct DocumentSection {
    SectionRole role = SectionRole::INTRODUCTION;
    std::string heading;
    std::vector<DocumentParagraph> paragraphs;
};

struct DocumentPlan {
    std::string title;
    std::string topic;
    std::vector<DocumentSection> sections;
    std::vector<std::string> introduced_entities;
    std::vector<std::string> used_concepts;
    std::string rationale;
};

struct RenderOptions {
    bool title = true;
    bool headings = true;
    size_t max_sentences = 0; // 0 = no limit
};

/// Hierarchical document generation: goal -> outline -> paragraphs -> sentence
/// plans. It reasons over the graph before any word is generated, then a
/// revision pass removes duplicated sentences and mechanical repetition.
class DocumentPlanner {
public:
    DocumentPlan plan(const std::string& topic,
                      const CausalGraph& graph,
                      const Codebook& codebook,
                      const LanguageRules& rules,
                      size_t max_sections = 4,
                      size_t paragraphs_per_section = 2,
                      const TermResolver* terms = nullptr,
                      const TextParser* parser_override = nullptr) const;

    std::string realize(const DocumentPlan& plan,
                        const ISurfaceRealizer& realizer,
                        const LanguageRules& rules,
                        const RenderOptions& options = {}) const;

    /// Applies deterministic revision: duplicate removal, connector de-repetition,
    /// whitespace/punctuation normalization. It never adds facts.
    static std::string revise(const std::string& text, const LanguageRules& rules);
    static const char* role_name(SectionRole role) noexcept;
};

} // namespace cognicore
