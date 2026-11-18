#pragma once

#include "core/causal_graph.hpp"

#include <string>
#include <unordered_set>
#include <vector>

namespace cognicore {

struct MorphToken {
    int id = 0;
    std::string form;
    std::string lemma;
    std::string upos; // NOUN, VERB, ADJ, ADV, PRON, ADP, DET, etc.
    GramTense tense = GramTense::PRES;
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::SING;
    bool negated = false;
    int head = 0;
    std::string deprel; // nsubj, obj, obl, advcl, mark, etc.
};

enum class ConditionKind : uint8_t {
    CONDITIONAL = 0,
    TEMPORAL = 1,
    CAUSAL = 2,
    CONCESSIVE = 3,
    CONSECUTIVE = 4
};

struct ParsedTriple {
    std::string source_text;
    std::string subject;
    std::string relation;
    std::string object;
    RelationType type = RelationType::CAUSAL;
    float confidence = 0.8f;
    bool negated = false;
    GramTense tense = GramTense::PRES;
    std::string modality;
    GramGender subj_gender = GramGender::MASC;
    GramNumber subj_number = GramNumber::SING;
    GramGender obj_gender = GramGender::MASC;
    GramNumber obj_number = GramNumber::SING;
    std::string quantifier;     // ALL, SOME, NO, MOST, EACH, ANY (empty = unmarked)
    ConditionKind condition_kind = ConditionKind::CONDITIONAL;
    bool has_condition = false;
    std::string relative_antecedent; // resolved antecedent for acl:relcl
};

struct ConstructionFrame {
    std::string pattern; // e.g. "[SUBJ:ENT] gera [OBJ:ENT]"
    RelationType rel_type = RelationType::CAUSAL;
    uint32_t frequency = 1;
};

/// A conditional stated by the text ("se ... então ..." / "if ... then ...").
/// It is kept separate from asserted triples: a conditional premise is not a fact.
struct ParsedConditional {
    std::string antecedent;
    std::string consequent;
    ConditionKind kind = ConditionKind::CONDITIONAL;
    std::string quantifier;
    bool negated = false;
};

struct ParsedDocument {
    std::vector<std::vector<std::string>> sentences_tokens;
    std::vector<std::vector<MorphToken>> morph_sentences;
    std::vector<ParsedTriple> triples;
    std::vector<ConstructionFrame> frames;
    std::vector<ParsedConditional> conditionals;
};

class TextParser {
public:
    TextParser();

    bool load_rules_json(const std::string& path);

    static std::string utf8_tolower(const std::string& s);
    static std::vector<std::string> split_sentences(const std::string& text);
    std::vector<std::string> tokenize(const std::string& sentence) const;
    std::vector<std::string> extract_content_tokens(const std::string& text) const;

    ParsedDocument parse_text(const std::string& text) const;
    /// Embedded rule-based dependency parse (no UDPipe/CoNLL-U required).
    /// Produces heads/deprels, SVO triples, relative clauses, advcl conditions,
    /// quantifiers and negation scope.
    ParsedDocument parse_dependencies(const std::string& text) const;
    ParsedDocument parse_conllu(const std::string& conllu_text) const;
    static const char* condition_kind_name(ConditionKind kind) noexcept;

    static GramGender infer_gender(const std::string& word);
    static GramNumber infer_number(const std::string& word);
    static std::string lemmatize_simple(const std::string& word);

    bool is_stopword(const std::string& tok) const;

private:
    struct MarkerRule {
        std::string phrase;
        std::vector<std::string> words;
        RelationType type;
        bool reverse_order = false; // e.g., "A porque B" => B causes A
    };

    void init_default_rules();
    std::string clean_phrase(const std::vector<std::string>& tokens, size_t start, size_t end) const;

    std::vector<MarkerRule> markers_;
    std::unordered_set<std::string> stopwords_;
};

} // namespace cognicore
