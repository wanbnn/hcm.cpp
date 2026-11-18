#pragma once

#include "core/language_rules.hpp"
#include "core/realizer.hpp"
#include "core/text_parser.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

struct LexiconEntry {
    std::string lemma;
    std::string pos;
    uint32_t frequency = 0;
    std::array<uint32_t, 4> tone{};
    uint8_t gender = 0, number = 0;
};

/// Corpus-derived lexical knowledge. Surface choices are separate from graph
/// facts: choosing "provocar" instead of "gerar" never changes the stored edge.
class Lexicon {
public:
    void observe(const std::string& text, const TextParser& parser, const std::string& tone);
    void load_synonyms(const LanguageRules& rules);

    std::string choose_verb(const std::string& lemma, const std::string& object,
                            const std::string& tone) const;
    std::string choose_noun(const std::string& noun, const std::string& relation,
                            const std::string& tone) const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);
    void clear();

    size_t entries() const { return entries_.size(); }
    size_t collocations() const { return collocations_.size(); }
    size_t memory_bytes() const;
    bool empty() const { return entries_.empty(); }

private:
    static size_t tone_index(const std::string& tone);
    std::vector<LexiconEntry> entries_;
    std::unordered_map<std::string, uint32_t> index_;
    // Key forms: "<verb>\t<object>" and "<subject>\t<verb>".
    std::unordered_map<std::string, uint32_t> collocations_;
    std::vector<std::vector<std::string>> verb_groups_, noun_groups_;
};

/// Applies surface-only lexical overrides to a plan copy.
class LexicalChooser {
public:
    SemanticPlan apply(const SemanticPlan& plan, const Lexicon& lexicon) const;
};

} // namespace cognicore
