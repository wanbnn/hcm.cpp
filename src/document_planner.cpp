#include "core/document_planner.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <unordered_set>

namespace cognicore {
namespace {

std::string lower_ascii(std::string value) {
    return TextParser::utf8_tolower(value);
}

bool contains_token(const std::string& text, const std::string& token) {
    return lower_ascii(text).find(lower_ascii(token)) != std::string::npos;
}

std::string first_or(const std::vector<std::string>& values, const std::string& fallback) {
    return values.empty() ? fallback : values.front();
}

std::string sentence_key(const CausalPath& path) {
    std::ostringstream out;
    for (const auto& step : path.steps)
        out << lower_ascii(step.src) << '|' << TextParser::lemmatize_simple(lower_ascii(step.rel))
            << '|' << lower_ascii(step.dst) << '|' << step.negated << ';';
    return out.str();
}

bool is_inhibitory(const CausalPath& path) {
    for (const auto& step : path.steps)
        if (step.negated || step.type == RelationType::INHIBITS) return true;
    return false;
}

bool has_taxonomic(const CausalPath& path) {
    for (const auto& step : path.steps)
        if (step.type == RelationType::TAXONOMIC || step.type == RelationType::PROPERTY) return true;
    return false;
}

void add_unique(std::vector<CausalPath>& values, std::set<std::string>& seen, CausalPath path) {
    if (path.steps.empty()) return;
    const auto key = sentence_key(path);
    if (!seen.insert(key).second) return;
    values.push_back(std::move(path));
}

DocumentParagraph make_paragraph(const std::string& topic,
                                 const std::vector<CausalPath>& paths,
                                 size_t begin, size_t end,
                                 DialogIntent intent,
                                 const std::string& language) {
    DocumentParagraph paragraph;
    paragraph.topic = topic;
    for (size_t i = begin; i < end && i < paths.size(); ++i) {
        DocumentSentence sentence;
        sentence.plan.intent = intent;
        sentence.plan.language = language;
        sentence.plan.paths.push_back(paths[i]);
        paragraph.sentences.push_back(std::move(sentence));
    }
    return paragraph;
}

std::vector<DocumentParagraph> split_paragraphs(const std::string& topic,
                                               const std::vector<CausalPath>& paths,
                                               size_t paragraph_count,
                                               DialogIntent intent,
                                               const std::string& language) {
    std::vector<DocumentParagraph> paragraphs;
    if (paths.empty()) return paragraphs;
    paragraph_count = std::max<size_t>(1, paragraph_count);
    const size_t per_paragraph = (paths.size() + paragraph_count - 1) / paragraph_count;
    for (size_t start = 0; start < paths.size(); start += per_paragraph) {
        paragraphs.push_back(make_paragraph(topic, paths, start,
                                            std::min(paths.size(), start + per_paragraph),
                                            intent, language));
        if (paragraphs.size() >= paragraph_count) break;
    }
    return paragraphs;
}

} // namespace

const char* DocumentPlanner::role_name(SectionRole role) noexcept {
    switch (role) {
        case SectionRole::INTRODUCTION: return "INTRODUCTION";
        case SectionRole::CAUSES: return "CAUSES";
        case SectionRole::CONSEQUENCES: return "CONSEQUENCES";
        case SectionRole::ALTERNATIVES: return "ALTERNATIVES";
        case SectionRole::CONCLUSION: return "CONCLUSION";
    }
    return "UNKNOWN";
}

DocumentPlan DocumentPlanner::plan(const std::string& topic,
                                   const CausalGraph& graph,
                                   const Codebook& codebook,
                                   const LanguageRules& rules,
                                   size_t max_sections,
                                   size_t paragraphs_per_section) const {
    DocumentPlan document;
    document.topic = topic;
    const auto title_template = first_or(rules.document_title, "{topic}");
    document.title = title_template;
    const auto title_at = document.title.find("{topic}");
    if (title_at != std::string::npos) document.title.replace(title_at, 7, topic);

    TextParser parser;
    const auto topic_tokens = parser.extract_content_tokens(topic);
    std::vector<std::string> seeds;
    for (const auto& node : graph.nodes()) {
        if (node.name.empty()) continue;
        if (lower_ascii(node.name) == lower_ascii(topic) ||
            std::find(topic_tokens.begin(), topic_tokens.end(), node.name) != topic_tokens.end()) {
            seeds.push_back(node.name);
        }
    }
    if (seeds.empty()) seeds.push_back(topic);
    for (const auto& seed : seeds)
        if (graph.find_node(seed) != UINT32_MAX)
            document.used_concepts.push_back(seed);

    std::vector<CausalPath> intro, causes, consequences, alternatives;
    std::set<std::string> seen_intro, seen_causes, seen_consequences, seen_alternatives;

    // Direct facts give high-precision section material.
    for (const auto& seed : seeds) {
        for (const auto& step : graph.query_direct(seed, 12)) {
            CausalPath path;
            path.steps.push_back(step);
            path.score = step.confidence;
            if (step.dst == seed) add_unique(causes, seen_causes, path);
            else if (step.src == seed) {
                if (step.negated || step.type == RelationType::INHIBITS)
                    add_unique(alternatives, seen_alternatives, path);
                else add_unique(consequences, seen_consequences, path);
            }
            if ((step.type == RelationType::TAXONOMIC || step.type == RelationType::PROPERTY) &&
                step.src == seed)
                add_unique(intro, seen_intro, path);
        }
    }
    // Multi-hop paths add the actual causal chains to the document.
    for (const auto& seed : seeds) {
        for (const auto& path : graph.find_paths({seed}, 3, 8, &codebook)) {
            if (path.steps.empty()) continue;
            if (path.steps.front().src == seed) {
                if (is_inhibitory(path)) add_unique(alternatives, seen_alternatives, path);
                else add_unique(consequences, seen_consequences, path);
            } else if (path.steps.back().dst == seed) {
                add_unique(causes, seen_causes, path);
            }
            if (has_taxonomic(path)) add_unique(intro, seen_intro, path);
        }
    }
    // Do not steal the only consequence/cause to fill an empty introduction:
    // sparse topics are better served by starting with their causal section.
    // Taxonomy/property facts remain the preferred opening when present.
    // Conclusion reuses the strongest consequence (or cause) as a closing route.
    std::vector<CausalPath> conclusion;
    std::set<std::string> seen_conclusion;
    if (!consequences.empty()) add_unique(conclusion, seen_conclusion, consequences.front());
    else if (!causes.empty()) add_unique(conclusion, seen_conclusion, causes.front());
    else if (!intro.empty()) add_unique(conclusion, seen_conclusion, intro.front());

    struct SectionInput { SectionRole role; const std::vector<CausalPath>* paths; const std::vector<std::string>* labels; DialogIntent intent; };
    const SectionInput candidates[] = {
        {SectionRole::INTRODUCTION, &intro, &rules.section_introduction, DialogIntent::ANSWER_FACT},
        {SectionRole::CAUSES, &causes, &rules.section_causes, DialogIntent::EXPLAIN_CAUSE},
        {SectionRole::CONSEQUENCES, &consequences, &rules.section_consequences, DialogIntent::EXPLAIN_CAUSE},
        {SectionRole::ALTERNATIVES, &alternatives, &rules.section_alternatives, DialogIntent::ANSWER_FACT},
        {SectionRole::CONCLUSION, &conclusion, &rules.section_conclusion, DialogIntent::ANSWER_FACT}
    };
    std::set<std::string> used_paths;
    for (const auto& candidate : candidates) {
        if (document.sections.size() >= max_sections) break;
        if (candidate.paths->empty()) continue;
        std::vector<CausalPath> selected;
        for (const auto& path : *candidate.paths) {
            const auto key = sentence_key(path);
            if (used_paths.count(key)) continue;
            used_paths.insert(key);
            selected.push_back(path);
        }
        if (selected.empty()) continue;
        DocumentSection section;
        section.role = candidate.role;
        section.heading = first_or(*candidate.labels, role_name(candidate.role));
        section.paragraphs = split_paragraphs(topic, selected, paragraphs_per_section,
                                              candidate.intent, rules.language);
        if (!section.paragraphs.empty()) document.sections.push_back(std::move(section));
    }
    for (const auto& section : document.sections)
        for (const auto& paragraph : section.paragraphs)
            for (const auto& sentence : paragraph.sentences)
                for (const auto& path : sentence.plan.paths)
                    for (const auto& step : path.steps) {
                        if (std::find(document.introduced_entities.begin(), document.introduced_entities.end(), step.src) == document.introduced_entities.end())
                            document.introduced_entities.push_back(step.src);
                        if (std::find(document.introduced_entities.begin(), document.introduced_entities.end(), step.dst) == document.introduced_entities.end())
                            document.introduced_entities.push_back(step.dst);
                        if (std::find(document.used_concepts.begin(), document.used_concepts.end(), step.rel) == document.used_concepts.end())
                            document.used_concepts.push_back(step.rel);
                    }
    std::ostringstream rationale;
    rationale << "outline=" << document.sections.size() << " seções; intro=" << intro.size()
              << " causas=" << causes.size() << " consequências=" << consequences.size()
              << " alternativas=" << alternatives.size();
    document.rationale = rationale.str();
    return document;
}

std::string DocumentPlanner::realize(const DocumentPlan& plan,
                                     const ISurfaceRealizer& realizer,
                                     const LanguageRules& rules) const {
    std::ostringstream out;
    if (!plan.title.empty()) out << "# " << plan.title << "\n\n";
    size_t connector_index = 0;
    std::unordered_set<std::string> emitted_paragraphs;
    for (const auto& section : plan.sections) {
        if (!section.heading.empty()) out << "## " << section.heading << "\n\n";
        for (size_t p = 0; p < section.paragraphs.size(); ++p) {
            const auto& paragraph = section.paragraphs[p];
            std::string paragraph_text;
            for (const auto& sentence : paragraph.sentences) {
                auto text = realizer.realize(sentence.plan);
                if (text.empty()) continue;
                if (text.back() != '.' && text.back() != '!' && text.back() != '?') text += '.';
                if (!paragraph_text.empty()) paragraph_text += " ";
                paragraph_text += text;
            }
            if (paragraph_text.empty()) continue;
            if (p > 0 && !rules.document_connectors.empty()) {
                auto connector = rules.document_connectors[connector_index++ % rules.document_connectors.size()];
                if (!paragraph_text.empty() && static_cast<unsigned char>(paragraph_text[0]) < 128)
                    paragraph_text[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(paragraph_text[0])));
                paragraph_text = connector + ", " + paragraph_text;
            }
            auto normalized = lower_ascii(paragraph_text);
            for (const auto& connector : rules.document_connectors) {
                const auto prefix = lower_ascii(connector) + ", ";
                if (normalized.starts_with(prefix)) { normalized.erase(0, prefix.size()); break; }
            }
            while (!normalized.empty() && (normalized.back() == '.' || normalized.back() == ' ')) normalized.pop_back();
            if (!emitted_paragraphs.insert(normalized).second) continue;
            out << revise(paragraph_text, rules) << "\n\n";
        }
    }
    return out.str();
}

std::string DocumentPlanner::revise(const std::string& text, const LanguageRules& rules) {
    std::vector<std::string> sentences;
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        current.push_back(text[i]);
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            if (i + 1 >= text.size() || text[i + 1] == ' ' || text[i + 1] == '\n') {
                auto value = current;
                while (!value.empty() && value.front() == ' ') value.erase(value.begin());
                if (!value.empty()) sentences.push_back(std::move(value));
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        while (!current.empty() && current.front() == ' ') current.erase(current.begin());
        if (!current.empty()) sentences.push_back(std::move(current));
    }
    std::vector<std::string> unique;
    std::unordered_set<std::string> seen;
    for (auto& sentence : sentences) {
        std::string normalized = lower_ascii(sentence);
        while (!normalized.empty() && (normalized.back() == ' ' || normalized.back() == '.')) normalized.pop_back();
        if (!seen.insert(normalized).second) continue;
        unique.push_back(std::move(sentence));
    }
    std::string result;
    std::string previous_connector;
    for (auto& sentence : unique) {
        for (const auto& connector : rules.document_connectors) {
            if (sentence.starts_with(connector + ", ") && connector == previous_connector) {
                sentence.erase(0, connector.size() + 2);
                if (!sentence.empty() && static_cast<unsigned char>(sentence[0]) < 128)
                    sentence[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(sentence[0])));
                break;
            }
        }
        for (const auto& connector : rules.document_connectors)
            if (sentence.starts_with(connector + ", ")) { previous_connector = connector; break; }
        if (!result.empty()) result += " ";
        result += sentence;
    }
    return result;
}

} // namespace cognicore
