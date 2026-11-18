#include "core/text_parser.hpp"
#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <charconv>
#include <unordered_map>
#include <stdexcept>
#include <functional>

namespace cognicore {

namespace {

// Extracts JSON string array for a given key from a simple JSON object.
std::vector<std::string> extract_json_array(const std::string& json, const std::string& key) {
    std::vector<std::string> out;
    const std::string quoted_key = "\"" + key + "\"";
    const size_t kpos = json.find(quoted_key);
    if (kpos == std::string::npos) return out;
    const size_t lbr = json.find('[', kpos);
    const size_t rbr = json.find(']', lbr);
    if (lbr == std::string::npos || rbr == std::string::npos) return out;

    size_t i = lbr + 1;
    while (i < rbr) {
        const size_t q1 = json.find('"', i);
        if (q1 == std::string::npos || q1 >= rbr) break;
        const size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 >= rbr) break;
        out.push_back(json.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
}

bool is_ascii_punct_or_space(unsigned char c) noexcept {
    return c <= 0x20 || std::ispunct(c) != 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

// General (non-marker) lexical verb recognizer. The lemma/form table is
// loaded from the model-owned parser_pt.json; the suffix rules below are
// morphological code, not a fixed vocabulary.
bool TextParser::general_verb_lemma(const std::string& token, std::string& lemma) const {
    if (token.size() < 2) return false;
    // Closed-class words must never be promoted to verbs ("de" -> "dar").
    if (prepositions_.count(token) || articles_.count(token) ||
        conjunctions_.count(token) || pronouns_.count(token)) return false;
    if (auto it = verb_forms_.find(token); it != verb_forms_.end()) { lemma = it->second; return true; }
    const std::pair<const char*, const char*> endings[] = {
        {"aram", "ar"}, {"eram", "er"}, {"iram", "ir"},
        {"ou", "ar"}, {"eu", "er"}, {"iu", "ir"}
    };
    for (const auto& [suffix, infinitive] : endings) {
        if (token.size() >= std::string_view(suffix).size() + 2 && ends_with(token, suffix)) {
            lemma = token.substr(0, token.size() - std::string_view(suffix).size()) + infinitive;
            return true;
        }
    }
    if (token.size() >= 5 && ends_with(token, "ied")) { lemma = token.substr(0, token.size() - 3) + "y"; return true; }
    if (token.size() >= 5 && ends_with(token, "ed")) { lemma = token.substr(0, token.size() - 2); return true; }
    return false;
}

std::string TextParser::clean_span(const std::vector<std::string>& tokens, size_t start,
                                  size_t end) const {
    while (start < end && (stopwords_.count(tokens[start]) || clean_extra_.count(tokens[start]))) ++start;
    while (end > start && (stopwords_.count(tokens[end - 1]) || clean_extra_.count(tokens[end - 1]))) --end;
    // A new PP introduced by "em/para/com/por/..." usually belongs to the verb,
    // not to the nominal head. Stop the NP before it.
    for (size_t i = start + 1; i < end; ++i) {
        if (np_boundary_prepositions_.count(tokens[i])) { end = i; break; }
    }
    while (end > start && (stopwords_.count(tokens[end - 1]) || clean_extra_.count(tokens[end - 1]))) --end;
    std::string out;
    for (size_t i = start; i < end; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

size_t TextParser::find_main_verb(const std::vector<std::string>& tokens, std::string& lemma,
                                  size_t from) const {
    for (size_t i = from; i < tokens.size(); ++i) {
        if (general_verb_lemma(tokens[i], lemma)) return i;
    }
    return std::string::npos;
}

void TextParser::agree_with_article(const std::vector<std::string>& tokens, size_t begin,
                                   GramGender& gender, GramNumber& number) const {
    if (begin >= tokens.size()) return;
    const auto& article = tokens[begin];
    if (feminine_articles_.count(article)) gender = GramGender::FEM;
    if (masculine_articles_.count(article)) gender = GramGender::MASC;
    if (plural_articles_.count(article)) number = GramNumber::PLUR;
}

TextParser::TextParser() {
    init_default_rules();
}

void TextParser::init_default_rules() {
    markers_.clear();
    stopwords_.clear();
    clean_extra_.clear();
    articles_.clear();
    prepositions_.clear();
    pronouns_.clear();
    relative_pronouns_.clear();
    subordinators_.clear();
    conjunctions_.clear();
    adverbs_.clear();
    negation_words_.clear();
    modal_words_.clear();
    copula_words_.clear();
    adjective_words_.clear();
    pp_modifier_prepositions_.clear();
    feminine_words_.clear();
    masculine_words_.clear();
    singular_exceptions_.clear();
    np_boundary_prepositions_.clear();
    feminine_articles_.clear();
    masculine_articles_.clear();
    plural_articles_.clear();
    verb_forms_.clear();
    quantifier_map_.clear();
    condition_markers_.clear();
    for (const char* path : {"data/parser_pt.json", "../data/parser_pt.json",
                             "../../data/parser_pt.json"}) {
        std::ifstream probe(path);
        if (probe && load_parser_json(path)) return;
    }
}

void TextParser::add_marker_list(const std::vector<std::string>& list, RelationType type) {
    for (const auto& value : list) {
        MarkerRule rule;
        rule.phrase = utf8_tolower(value);
        rule.words = tokenize(rule.phrase);
        rule.type = type;
        rule.reverse_order = reverse_order_markers_.count(rule.phrase) != 0;
        if (!rule.words.empty()) markers_.push_back(std::move(rule));
    }
}

bool TextParser::load_parser_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream buffer;
    buffer << ifs.rdbuf();
    const std::string json = buffer.str();

    auto fill_set = [&](const char* key, std::unordered_set<std::string>& set) {
        for (const auto& value : extract_json_array(json, key)) set.insert(utf8_tolower(value));
    };
    auto fill_map = [&](const char* key, std::unordered_map<std::string, std::string>& map) {
        for (const auto& value : extract_json_array(json, key)) {
            const auto eq = value.find('=');
            if (eq == std::string::npos) continue;
            map[utf8_tolower(value.substr(0, eq))] = value.substr(eq + 1);
        }
    };
    auto fill_conditions = [&](const char* key) {
        for (const auto& value : extract_json_array(json, key)) {
            const auto eq = value.find('=');
            if (eq == std::string::npos) continue;
            const auto label = value.substr(eq + 1);
            ConditionKind kind = ConditionKind::CONDITIONAL;
            if (label == "TEMPORAL") kind = ConditionKind::TEMPORAL;
            else if (label == "CAUSAL") kind = ConditionKind::CAUSAL;
            else if (label == "CONCESSIVE") kind = ConditionKind::CONCESSIVE;
            else if (label == "CONSECUTIVE") kind = ConditionKind::CONSECUTIVE;
            condition_markers_[utf8_tolower(value.substr(0, eq))] = kind;
        }
    };

    fill_set("articles", articles_);
    fill_set("prepositions", prepositions_);
    fill_set("pronouns", pronouns_);
    fill_set("relative_pronouns", relative_pronouns_);
    fill_set("subordinators", subordinators_);
    fill_set("conjunctions", conjunctions_);
    fill_set("adverbs", adverbs_);
    fill_set("negation_words", negation_words_);
    fill_set("modal_words", modal_words_);
    fill_set("copula_words", copula_words_);
    fill_set("adjective_words", adjective_words_);
    fill_set("pp_modifier_prepositions", pp_modifier_prepositions_);
    fill_set("feminine_words", feminine_words_);
    fill_set("masculine_words", masculine_words_);
    fill_set("singular_exceptions", singular_exceptions_);
    fill_set("np_boundary_prepositions", np_boundary_prepositions_);
    fill_set("feminine_articles", feminine_articles_);
    fill_set("masculine_articles", masculine_articles_);
    fill_set("plural_articles", plural_articles_);
    fill_set("stopwords", stopwords_);
    fill_set("clean_extra", clean_extra_);
    fill_map("quantifiers", quantifier_map_);
    fill_conditions("condition_markers");

    verb_forms_.clear();
    auto add_verb = [&](const std::string& lemma, const char* lang) {
        verb_forms_.emplace(lemma, lemma);
        for (auto tense : {GramTense::PRES, GramTense::PAST, GramTense::FUT, GramTense::SUBJ}) {
            for (auto number : {GramNumber::SING, GramNumber::PLUR}) {
                auto form = MorphologicalRealizer::conjugate_verb(lemma, tense, number, lang);
                if (!form.empty() && form != lemma) verb_forms_.emplace(form, lemma);
            }
        }
    };
    for (const auto& lemma : extract_json_array(json, "verb_lemmas_pt")) add_verb(lemma, "pt");
    for (const auto& lemma : extract_json_array(json, "verb_lemmas_en")) add_verb(lemma, "en");
    for (const auto& value : extract_json_array(json, "irregular_verbs")) {
        const auto eq = value.find('=');
        if (eq != std::string::npos) verb_forms_[value.substr(0, eq)] = value.substr(eq + 1);
    }

    markers_.clear();
    add_marker_list(extract_json_array(json, "causal_markers"), RelationType::CAUSAL);
    add_marker_list(extract_json_array(json, "inhibitory_markers"), RelationType::INHIBITS);
    add_marker_list(extract_json_array(json, "taxonomic_markers"), RelationType::TAXONOMIC);
    add_marker_list(extract_json_array(json, "action_markers"), RelationType::ACTION);
    add_marker_list(extract_json_array(json, "has_part_markers"), RelationType::HAS_PART);
    add_marker_list(extract_json_array(json, "part_of_markers"), RelationType::PART_OF);
    add_marker_list(extract_json_array(json, "function_markers"), RelationType::FUNCTION);
    add_marker_list(extract_json_array(json, "enables_markers"), RelationType::ENABLES);
    add_marker_list(extract_json_array(json, "sequence_markers"), RelationType::SEQUENCE);
    std::sort(markers_.begin(), markers_.end(),
              [](const MarkerRule& a, const MarkerRule& b) {
                  return a.words.size() > b.words.size();
              });
    return !stopwords_.empty() || !markers_.empty();
}

bool TextParser::load_rules_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    const std::string json = ss.str();

    auto causal = extract_json_array(json, "causal_markers");
    auto inhib = extract_json_array(json, "inhibitory_markers");
    auto taxon = extract_json_array(json, "taxonomic_markers");
    auto action = extract_json_array(json, "action_markers");
    auto has_part = extract_json_array(json, "has_part_markers");
    auto part_of = extract_json_array(json, "part_of_markers");
    auto function = extract_json_array(json, "function_markers");
    auto enables = extract_json_array(json, "enables_markers");
    auto sequence = extract_json_array(json, "sequence_markers");
    auto stops = extract_json_array(json, "stopwords");

    if (!stops.empty()) {
        for (const auto& s : stops) stopwords_.insert(utf8_tolower(s));
    }

    if (!causal.empty() || !inhib.empty() || !taxon.empty() || !action.empty() ||
        !has_part.empty() || !part_of.empty() || !function.empty() ||
        !enables.empty() || !sequence.empty()) {
        markers_.clear();
        auto add_list = [&](const std::vector<std::string>& list, RelationType t) {
            for (const auto& m : list) {
                MarkerRule r;
                r.phrase = utf8_tolower(m);
                r.words = tokenize(r.phrase);
                r.type = t;
                r.reverse_order = reverse_order_markers_.count(r.phrase) != 0;
                if (!r.words.empty()) markers_.push_back(std::move(r));
            }
        };
        add_list(causal, RelationType::CAUSAL);
        add_list(inhib, RelationType::INHIBITS);
        add_list(taxon, RelationType::TAXONOMIC);
        add_list(action, RelationType::ACTION);
        add_list(has_part, RelationType::HAS_PART);
        add_list(part_of, RelationType::PART_OF);
        add_list(function, RelationType::FUNCTION);
        add_list(enables, RelationType::ENABLES);
        add_list(sequence, RelationType::SEQUENCE);
        std::sort(markers_.begin(), markers_.end(),
                  [](const MarkerRule& a, const MarkerRule& b) {
                      return a.words.size() > b.words.size();
                  });
    }
    return true;
}

std::string TextParser::utf8_tolower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == 0xC3 && i + 1 < s.size()) {
            // Latin-1 Supplement uppercase in UTF-8: 0xC3 0x80..0x9E -> 0xC3 0xA0..0xBE
            unsigned char c2 = static_cast<unsigned char>(s[i + 1]);
            if (c2 >= 0x80 && c2 <= 0x9E && c2 != 0x97) {
                c2 = static_cast<unsigned char>(c2 + 0x20);
            }
            out.push_back(static_cast<char>(c));
            out.push_back(static_cast<char>(c2));
            ++i;
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

std::vector<std::string> TextParser::split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : text) {
        if (ch == '.' || ch == '!' || ch == '?' || ch == ';' || ch == '\n') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::vector<std::string> TextParser::tokenize(const std::string& sentence) const {
    const std::string lower = utf8_tolower(sentence);
    std::vector<std::string> tokens;
    std::string cur;
    for (size_t i = 0; i < lower.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(lower[i]);
        if (c < 0x80 && is_ascii_punct_or_space(c)) {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(static_cast<char>(c));
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

bool TextParser::is_stopword(const std::string& tok) const {
    return stopwords_.find(tok) != stopwords_.end();
}

std::vector<std::string> TextParser::extract_content_tokens(const std::string& text) const {
    std::vector<std::string> out;
    for (const auto& t : tokenize(text)) {
        if (!is_stopword(t) && t.size() > 1) {
            out.push_back(t);
        }
    }
    return out;
}

std::string TextParser::clean_phrase(const std::vector<std::string>& tokens, size_t start, size_t end) const {
    while (start < end && is_stopword(tokens[start])) ++start;
    while (end > start && is_stopword(tokens[end - 1])) --end;
    for (size_t i = start + 1; i < end; ++i) {
        if (np_boundary_prepositions_.count(tokens[i])) { end = i; break; }
    }
    while (end > start && is_stopword(tokens[end - 1])) --end;
    if (start >= end) return "";

    std::string res;
    for (size_t i = start; i < end; ++i) {
        if (!res.empty()) res.push_back(' ');
        res += tokens[i];
    }
    return res;
}

GramGender TextParser::infer_gender(const std::string& word) const {
    std::string w = utf8_tolower(word);
    // Multi-word terms take gender from their head noun, before the first PP.
    for (const char* separator : {" de ", " do ", " da ", " dos ", " das ",
                                  " em ", " no ", " na ", " nos ", " nas ",
                                  " por ", " para ", " com ", " sem "}) {
        const auto at = w.find(separator);
        if (at != std::string::npos) { w = w.substr(0, at); break; }
    }
    if (w.size() >= 2) {
        if (masculine_words_.count(w)) return GramGender::MASC;
        if (feminine_words_.count(w)) return GramGender::FEM;
        if (ends_with(w, "ão") && !masculine_words_.count(w)) return GramGender::FEM;
        auto ends_with = [&](const std::string& suf) {
            return w.size() >= suf.size() && w.compare(w.size() - suf.size(), suf.size(), suf) == 0;
        };
        if (ends_with("ção") || ends_with("são") || ends_with("dade") || ends_with("gem")) {
            return GramGender::FEM;
        }
        if (w.back() == 'a' && !ends_with("ma") && !ends_with("pa")) {
            return GramGender::FEM;
        }
        if (ends_with("as")) {
            return GramGender::FEM;
        }
    }
    return GramGender::MASC;
}

GramNumber TextParser::infer_number(const std::string& word) const {
    std::string w = utf8_tolower(word);
    for (const char* separator : {" de ", " do ", " da ", " dos ", " das ",
                                  " em ", " no ", " na ", " nos ", " nas ",
                                  " por ", " para ", " com ", " sem "}) {
        const auto at = w.find(separator);
        if (at != std::string::npos) { w = w.substr(0, at); break; }
    }
    if (w.size() >= 4 && w.back() == 's') {
        if (singular_exceptions_.count(w)) return GramNumber::SING;
        return GramNumber::PLUR;
    }
    return GramNumber::SING;
}

std::string TextParser::lemmatize_simple(const std::string& word) {
    std::string w = utf8_tolower(word);
    if (w == "são" || w == "foi" || w == "era") return "ser";
    if (w == "gera" || w == "geram" || w == "gerou") return "gerar";
    if (w == "causa" || w == "causam" || w == "causou") return "causar";
    if (w == "provoca" || w == "provocam" || w == "provocou") return "provocar";
    if (w == "derrete" || w == "derretem" || w == "derreteu") return "derreter";
    if (w == "impede" || w == "impedem" || w == "impediu") return "impedir";
    if (w == "apaga" || w == "apagam" || w == "apagou") return "apagar";
    if (w == "aquece" || w == "aquecem" || w == "aqueceu") return "aquecer";
    if (w == "esfria" || w == "esfriam" || w == "esfriou") return "esfriar";
    if (w == "produz" || w == "produzem" || w == "produziu") return "produzir";
    if (w == "forma" || w == "formam" || w == "formou") return "formar";
    if (w == "evapora" || w == "evaporou") return "evaporar";
    if (w == "congela" || w == "congelou") return "congelar";
    static const std::vector<std::string> verbs = {
        "gerar", "causar", "provocar", "derreter", "impedir", "apagar", "aquecer",
        "esfriar", "produzir", "formar", "evaporar", "congelar", "bloquear", "reduzir",
        "aumentar", "evitar", "ser", "ter", "poder", "dever", "possuir"
    };
    static const auto inflections = [] {
        std::unordered_map<std::string, std::string> forms;
        for (const auto& lemma : verbs)
            for (auto tense : {GramTense::PRES, GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
                for (auto number : {GramNumber::SING, GramNumber::PLUR})
                    forms.emplace(MorphologicalRealizer::conjugate_verb(lemma, tense, number), lemma);
        return forms;
    }();
    if (auto it = inflections.find(w); it != inflections.end()) return it->second;
    return w;
}

const char* TextParser::condition_kind_name(ConditionKind kind) noexcept {
    switch (kind) {
        case ConditionKind::CONDITIONAL: return "CONDITIONAL";
        case ConditionKind::TEMPORAL: return "TEMPORAL";
        case ConditionKind::CAUSAL: return "CAUSAL";
        case ConditionKind::CONCESSIVE: return "CONCESSIVE";
        case ConditionKind::CONSECUTIVE: return "CONSECUTIVE";
    }
    return "CONDITIONAL";
}

std::string TextParser::quantifier_for(const std::string& token) const {
    const auto it = quantifier_map_.find(token);
    return it == quantifier_map_.end() ? std::string{} : it->second;
}

bool TextParser::negation_token(const std::string& token) const {
    return negation_words_.count(token) != 0;
}

bool TextParser::modal_token(const std::string& token) const {
    return modal_words_.count(token) != 0;
}

bool TextParser::copula_token(const std::string& lemma) const {
    return copula_words_.count(lemma) != 0;
}

bool TextParser::adjective_token(const std::string& token) const {
    if (adjective_words_.count(token)) return true;
    return token.size() > 4 && (ends_with(token, "oso") || ends_with(token, "osa") ||
        ends_with(token, "ivo") || ends_with(token, "iva") || ends_with(token, "vel") ||
        ends_with(token, "al") || ends_with(token, "il") || ends_with(token, "nte"));
}

ConditionKind TextParser::condition_kind_for(const std::string& marker) const {
    const auto it = condition_markers_.find(marker);
    return it == condition_markers_.end() ? ConditionKind::CONDITIONAL : it->second;
}

struct DepToken {
    std::string form, lemma, upos;
    GramTense tense = GramTense::PRES;
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::SING;
    bool negated = false;
    std::string quantifier;
    int head = 0;
    std::string deprel;
};

ParsedDocument TextParser::parse_dependencies(const std::string& text) const {
    ParsedDocument doc;
    auto relation_type = [&](const std::string& lemma) {
        const auto canonical = lemmatize_simple(lemma);
        for (const auto& rule : markers_)
            if (rule.phrase == lemma || lemmatize_simple(rule.phrase) == canonical) return rule.type;
        return RelationType::ACTION;
    };

    for (const auto& sentence : split_sentences(text)) {
        const auto forms = tokenize(sentence);
        if (forms.empty()) continue;
        const int n = static_cast<int>(forms.size());
        std::vector<DepToken> t(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            auto& token = t[static_cast<size_t>(i)];
            token.form = forms[static_cast<size_t>(i)];
            token.lemma = lemmatize_simple(token.form);
            token.gender = infer_gender(token.form);
            token.number = infer_number(token.form);
            const auto quantity = quantifier_for(token.form);
            if (!quantity.empty()) { token.upos = "DET"; token.quantifier = quantity; continue; }
            if (articles_.count(token.form)) { token.upos = "DET"; continue; }
            if (prepositions_.count(token.form)) { token.upos = "ADP"; continue; }
            if (conjunctions_.count(token.form)) { token.upos = "CCONJ"; continue; }
            if (subordinators_.count(token.form)) { token.upos = "SCONJ"; continue; }
            if (relative_pronouns_.count(token.form) || pronouns_.count(token.form)) { token.upos = "PRON"; continue; }
            if (adverbs_.count(token.form)) {
                token.upos = "ADV";
                token.negated = negation_token(token.form);
                continue;
            }
            std::string lemma;
            if (general_verb_lemma(token.form, lemma)) {
                token.lemma = lemma;
                token.upos = modal_token(token.form) ? "AUX" : "VERB";
                continue;
            }
            if (adjective_token(token.form)) { token.upos = "ADJ"; continue; }
            if (token.form.find_first_of("0123456789") != std::string::npos) { token.upos = "NUM"; continue; }
            token.upos = "NOUN";
        }

        std::vector<int> verbs;
        for (int i = 0; i < n; ++i)
            if (t[static_cast<size_t>(i)].upos == "VERB") verbs.push_back(i);
        std::vector<MorphToken> morph;
        for (int i = 0; i < n; ++i)
            morph.push_back(MorphToken{i + 1, t[static_cast<size_t>(i)].form, t[static_cast<size_t>(i)].lemma,
                                       t[static_cast<size_t>(i)].upos, t[static_cast<size_t>(i)].tense,
                                       t[static_cast<size_t>(i)].gender, t[static_cast<size_t>(i)].number,
                                       t[static_cast<size_t>(i)].negated, 0, ""});
        doc.sentences_tokens.push_back(forms);
        if (verbs.empty()) {
            doc.morph_sentences.push_back(std::move(morph));
            continue;
        }

        std::vector<bool> relative_verb(static_cast<size_t>(n), false);
        std::vector<int> relative_antecedent(static_cast<size_t>(n), -1);
        std::vector<bool> relative_pronoun(static_cast<size_t>(n), false);
        std::vector<int> relative_pronoun_verb(static_cast<size_t>(n), -1);
        std::vector<bool> has_condition(static_cast<size_t>(n), false);
        std::vector<ConditionKind> condition_kind(static_cast<size_t>(n), ConditionKind::CONDITIONAL);
        std::vector<std::string> condition_marker(static_cast<size_t>(n));
        std::vector<std::string> modality_of_verb(static_cast<size_t>(n));
        std::vector<bool> future_verb(static_cast<size_t>(n), false);

        for (int i = 1; i < n; ++i) {
            if (t[static_cast<size_t>(i)].upos != "PRON" || !relative_pronouns_.count(forms[static_cast<size_t>(i)])) continue;
            const int previous = i - 1;
            const auto previous_upos = t[static_cast<size_t>(previous)].upos;
            if (previous_upos != "NOUN" && previous_upos != "PRON" && previous_upos != "ADJ") continue;
            int verb = -1;
            for (int j = i + 1; j < n && j <= i + 4; ++j)
                if (t[static_cast<size_t>(j)].upos == "VERB") { verb = j; break; }
            if (verb < 0) continue;
            relative_pronoun[static_cast<size_t>(i)] = true;
            relative_pronoun_verb[static_cast<size_t>(i)] = verb;
            relative_verb[static_cast<size_t>(verb)] = true;
            relative_antecedent[static_cast<size_t>(verb)] = previous;
        }
        for (int i = 0; i < n; ++i) {
            if (t[static_cast<size_t>(i)].upos != "SCONJ") continue;
            int verb = -1;
            for (int j = i + 1; j < n && j <= i + 5; ++j)
                if (t[static_cast<size_t>(j)].upos == "VERB") { verb = j; break; }
            if (verb < 0) continue;
            has_condition[static_cast<size_t>(verb)] = true;
            condition_kind[static_cast<size_t>(verb)] = condition_kind_for(forms[static_cast<size_t>(i)]);
            condition_marker[static_cast<size_t>(verb)] = forms[static_cast<size_t>(i)];
        }

        const bool leading_conditional = t[0].upos == "SCONJ";
        int main_verb = -1;
        for (int v : verbs) {
            if (!relative_verb[static_cast<size_t>(v)] && !has_condition[static_cast<size_t>(v)]) {
                main_verb = v;
                break;
            }
        }
        if (main_verb < 0) main_verb = verbs.front();

        for (int v : verbs) {
            auto& token = t[static_cast<size_t>(v)];
            if (v == main_verb) { token.head = 0; token.deprel = "root"; }
            else if (relative_verb[static_cast<size_t>(v)]) { token.head = main_verb + 1; token.deprel = "acl:relcl"; }
            else if (has_condition[static_cast<size_t>(v)]) { token.head = main_verb + 1; token.deprel = "advcl"; }
            else { token.head = main_verb + 1; token.deprel = "conj"; }
        }
        for (int i = 0; i < n; ++i) {
            if (t[static_cast<size_t>(i)].upos != "AUX") continue;
            int target = -1;
            for (int j = i + 1; j < n && j <= i + 3; ++j)
                if (t[static_cast<size_t>(j)].upos == "VERB") { target = j; break; }
            if (target < 0) continue;
            t[static_cast<size_t>(i)].head = target + 1;
            t[static_cast<size_t>(i)].deprel = "aux";
            if (modal_token(forms[static_cast<size_t>(i)]))
                modality_of_verb[static_cast<size_t>(target)] = forms[static_cast<size_t>(i)];
            if (future_auxiliaries_.count(forms[static_cast<size_t>(i)]))
                future_verb[static_cast<size_t>(target)] = true;
        }
        auto next_nominal = [&](int from) {
            for (int j = from + 1; j < n && j <= from + 5; ++j)
                if (t[static_cast<size_t>(j)].upos == "NOUN" || t[static_cast<size_t>(j)].upos == "PRON" ||
                    t[static_cast<size_t>(j)].upos == "ADJ") return j;
            return -1;
        };
        auto nearest_verb = [&](int at) {
            int best = -1, distance = 99;
            for (int v : verbs) {
                const int d = std::abs(v - at);
                if (d < distance) { distance = d; best = v; }
            }
            return best;
        };
        for (int i = 0; i < n; ++i) {
            auto& token = t[static_cast<size_t>(i)];
            if (token.upos == "DET" || token.upos == "NUM" || token.upos == "ADJ") {
                const int target = next_nominal(i);
                if (target >= 0) {
                    token.head = target + 1;
                    token.deprel = token.upos == "DET" ? "det" : token.upos == "NUM" ? "nummod" : "amod";
                }
            } else if (token.upos == "ADP") {
                const int target = next_nominal(i);
                if (target >= 0) { token.head = target + 1; token.deprel = "case"; }
            } else if (token.upos == "ADV") {
                const int verb = nearest_verb(i);
                if (verb >= 0) { token.head = verb + 1; token.deprel = "advmod"; }
            } else if (token.upos == "CCONJ") {
                const int verb = nearest_verb(i);
                if (verb >= 0) { token.head = verb + 1; token.deprel = "cc"; }
            } else if (relative_pronoun[static_cast<size_t>(i)]) {
                const int verb = relative_pronoun_verb[static_cast<size_t>(i)];
                if (verb >= 0) { token.head = verb + 1; token.deprel = "nsubj"; }
            } else if (token.upos == "SCONJ") {
                const int verb = nearest_verb(i);
                if (verb >= 0) { token.head = verb + 1; token.deprel = "mark"; }
            }
        }
        for (int i = 0; i < n; ++i) {
            auto& token = t[static_cast<size_t>(i)];
            if ((token.upos != "NOUN" && token.upos != "PRON") || token.head != 0) continue;
            int verb = nearest_verb(i);
            if (verb < 0) continue;
            bool already_subject = false, already_object = false;
            for (int j = 0; j < n; ++j) if (t[static_cast<size_t>(j)].head == verb + 1) {
                if (t[static_cast<size_t>(j)].deprel == "nsubj") already_subject = true;
                if (t[static_cast<size_t>(j)].deprel == "obj") already_object = true;
            }
            const bool preceded_by_preposition = i > 0 && t[static_cast<size_t>(i - 1)].upos == "ADP";
            if (preceded_by_preposition) { token.head = verb + 1; token.deprel = "obl"; }
            else if (i < verb) { token.head = verb + 1; token.deprel = already_subject ? "obl" : "nsubj"; }
            else { token.head = verb + 1; token.deprel = already_object ? "iobj" : "obj"; }
        }

        // Shared noun-phrase chunker. It expands the seed over determiners,
        // adjectives and (only) de/do/da/dos/das PPs. This keeps "motor de
        // combustão interna" together without swallowing verbal obliques such
        // as "conecta o êmbolo à árvore".
        auto is_de_pp = [&](int at) {
            if (at < 0 || at >= n || t[static_cast<size_t>(at)].upos != "ADP") return false;
            const auto& word = t[static_cast<size_t>(at)].form;
            return pp_modifier_prepositions_.count(word) != 0;
        };
        auto phrase = [&](int seed) {
            if (seed < 0 || seed >= n) return std::string{};
            int start = seed, end = seed;
            auto extend_left_np = [&](int& at) {
                while (at > 0 && (t[static_cast<size_t>(at - 1)].upos == "DET" ||
                                  t[static_cast<size_t>(at - 1)].upos == "ADJ" ||
                                  t[static_cast<size_t>(at - 1)].upos == "NUM")) --at;
            };
            extend_left_np(start);
            // If the seed is the object of a nominal "de" PP, move to its head.
            while (start >= 2 && is_de_pp(start - 1)) {
                int head = start - 2;
                extend_left_np(head);
                if (head < 0 || (t[static_cast<size_t>(head)].upos != "NOUN" &&
                                 t[static_cast<size_t>(head)].upos != "PROPN" &&
                                 t[static_cast<size_t>(head)].upos != "PRON")) break;
                start = head;
                extend_left_np(start);
            }
            int guard = 0;
            while (end + 1 < n && guard++ < 16) {
                const auto next = t[static_cast<size_t>(end + 1)].upos;
                if (next == "ADJ" || next == "NUM") { ++end; continue; }
                // A postposed bare noun often acts as a compound/adjectival
                // modifier in Portuguese ("combustão interna" when "interna"
                // is not tagged as ADJ).
                if ((next == "NOUN" || next == "PROPN") &&
                    (t[static_cast<size_t>(end)].upos == "NOUN" ||
                     t[static_cast<size_t>(end)].upos == "PROPN")) { ++end; continue; }
                if (!is_de_pp(end + 1)) break;
                int at = end + 2;
                if (at < n && t[static_cast<size_t>(at)].upos == "DET") ++at;
                while (at < n && (t[static_cast<size_t>(at)].upos == "ADJ" ||
                                  t[static_cast<size_t>(at)].upos == "NUM")) ++at;
                if (at >= n || (t[static_cast<size_t>(at)].upos != "NOUN" &&
                                t[static_cast<size_t>(at)].upos != "PROPN")) break;
                end = at;
                while (end + 1 < n && t[static_cast<size_t>(end + 1)].upos == "ADJ") ++end;
            }
            std::string out;
            for (int i = start; i <= end; ++i) {
                if (t[static_cast<size_t>(i)].upos == "DET") continue;
                if (!out.empty()) out += " ";
                out += t[static_cast<size_t>(i)].form;
            }
            if (out.empty()) out = t[static_cast<size_t>(seed)].form;
            return out;
        };

        for (int i = 0; i < n; ++i) {
            morph[static_cast<size_t>(i)].head = t[static_cast<size_t>(i)].head;
            morph[static_cast<size_t>(i)].deprel = t[static_cast<size_t>(i)].deprel;
            if (future_verb[static_cast<size_t>(i)]) morph[static_cast<size_t>(i)].tense = GramTense::FUT;
        }
        doc.morph_sentences.push_back(std::move(morph));

        for (int v : verbs) {
            int subject = -1, object = -1, fallback_object = -1;
            for (int j = 0; j < n; ++j) {
                if (t[static_cast<size_t>(j)].head != v + 1) continue;
                if (subject < 0 && t[static_cast<size_t>(j)].deprel == "nsubj") subject = j;
                if (object < 0 && t[static_cast<size_t>(j)].deprel == "obj") object = j;
                if (fallback_object < 0 && t[static_cast<size_t>(j)].deprel == "obl") fallback_object = j;
            }
            if (subject < 0 && relative_verb[static_cast<size_t>(v)] && relative_antecedent[static_cast<size_t>(v)] >= 0)
                subject = relative_antecedent[static_cast<size_t>(v)];
            if (object < 0) object = fallback_object;
            const bool copula = copula_token(t[static_cast<size_t>(v)].lemma);
            if (object < 0 && copula) {
                for (int j = v + 1; j < n && j <= v + 5; ++j) {
                    if (t[static_cast<size_t>(j)].upos == "ADJ" || t[static_cast<size_t>(j)].upos == "NOUN" ||
                        t[static_cast<size_t>(j)].upos == "PRON") { object = j; break; }
                }
            }
            if (subject < 0 && v == main_verb) {
                for (int j = v - 1; j >= 0 && j >= v - 5; --j) {
                    if ((t[static_cast<size_t>(j)].upos == "NOUN" || t[static_cast<size_t>(j)].upos == "PRON") &&
                        t[static_cast<size_t>(j)].deprel != "obj") { subject = j; break; }
                }
            }
            if (relative_verb[static_cast<size_t>(v)] && relative_antecedent[static_cast<size_t>(v)] >= 0)
                subject = relative_antecedent[static_cast<size_t>(v)];
            else if (subject >= 0 && t[static_cast<size_t>(subject)].upos == "PRON" &&
                     relative_pronouns_.count(t[static_cast<size_t>(subject)].form)) {
                for (int j = subject - 1; j >= 0; --j) {
                    if ((t[static_cast<size_t>(j)].upos == "NOUN" || t[static_cast<size_t>(j)].upos == "PRON") &&
                        t[static_cast<size_t>(j)].deprel != "obj") { subject = j; break; }
                }
            }
            if (subject < 0 || object < 0 || subject == object) continue;
            ParsedTriple triple;
            triple.source_text = sentence;
            triple.subject = phrase(subject);
            triple.object = phrase(object);
            if (triple.subject.empty() || triple.object.empty()) continue;
            triple.relation = t[static_cast<size_t>(v)].lemma;
            triple.negated = t[static_cast<size_t>(v)].negated;
            for (int j = 0; j < n; ++j)
                if (t[static_cast<size_t>(j)].head == v + 1 &&
                    (t[static_cast<size_t>(j)].deprel == "advmod" || t[static_cast<size_t>(j)].deprel == "aux") &&
                    t[static_cast<size_t>(j)].negated) triple.negated = true;
            triple.modality = modality_of_verb[static_cast<size_t>(v)];
            triple.tense = future_verb[static_cast<size_t>(v)] ? GramTense::FUT : t[static_cast<size_t>(v)].tense;
            for (auto tense : {GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
                for (auto number : {GramNumber::SING, GramNumber::PLUR})
                    if (MorphologicalRealizer::conjugate_verb(triple.relation, tense, number) == t[static_cast<size_t>(v)].form)
                        triple.tense = tense;
            if (copula) {
                if (t[static_cast<size_t>(object)].upos == "ADJ") triple.type = RelationType::PROPERTY;
                else if (t[static_cast<size_t>(object)].upos == "NOUN") triple.type = RelationType::TAXONOMIC;
                else triple.type = relation_type(triple.relation);
            } else triple.type = relation_type(triple.relation);
            triple.subj_gender = t[static_cast<size_t>(subject)].gender;
            triple.subj_number = t[static_cast<size_t>(subject)].number;
            triple.obj_gender = t[static_cast<size_t>(object)].gender;
            triple.obj_number = t[static_cast<size_t>(object)].number;
            triple.quantifier = t[static_cast<size_t>(subject)].quantifier;
            if (triple.quantifier.empty()) {
                for (int j = subject - 1; j >= 0 && j >= subject - 2; --j)
                    if (!t[static_cast<size_t>(j)].quantifier.empty()) { triple.quantifier = t[static_cast<size_t>(j)].quantifier; break; }
            }
            if (triple.quantifier.empty()) triple.quantifier = t[static_cast<size_t>(object)].quantifier;
            if (relative_verb[static_cast<size_t>(v)] && relative_antecedent[static_cast<size_t>(v)] >= 0)
                triple.relative_antecedent = phrase(relative_antecedent[static_cast<size_t>(v)]);
            if (has_condition[static_cast<size_t>(v)]) {
                triple.has_condition = true;
                triple.condition_kind = condition_kind[static_cast<size_t>(v)];
            }
            if (leading_conditional) continue;
            doc.triples.push_back(std::move(triple));
            doc.frames.push_back({"[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]", doc.triples.back().type, 1});
        }

        int marker = -1;
        for (int i = 0; i < n; ++i)
            if (t[static_cast<size_t>(i)].upos == "SCONJ") { marker = i; break; }
        if (marker >= 0) {
            int separator = -1;
            for (int i = 0; i < n; ++i)
                if (forms[static_cast<size_t>(i)] == "então" || forms[static_cast<size_t>(i)] == "then") { separator = i; break; }
            std::string antecedent, consequent;
            if (separator > marker) {
                for (int i = marker + 1; i < separator; ++i) {
                    if (!antecedent.empty()) antecedent += " ";
                    antecedent += forms[static_cast<size_t>(i)];
                }
                for (int i = separator + 1; i < n; ++i) {
                    if (!consequent.empty()) consequent += " ";
                    consequent += forms[static_cast<size_t>(i)];
                }
            } else if (marker > 0 && condition_kind_for(forms[static_cast<size_t>(marker)]) != ConditionKind::CONDITIONAL) {
                for (int i = 0; i < marker; ++i) {
                    if (!consequent.empty()) consequent += " ";
                    consequent += forms[static_cast<size_t>(i)];
                }
                for (int i = marker + 1; i < n; ++i) {
                    if (!antecedent.empty()) antecedent += " ";
                    antecedent += forms[static_cast<size_t>(i)];
                }
            }
            if (!antecedent.empty() && !consequent.empty()) {
                ParsedConditional conditional;
                conditional.antecedent = antecedent;
                conditional.consequent = consequent;
                conditional.kind = condition_kind_for(forms[static_cast<size_t>(marker)]);
                for (const auto& form : forms) {
                    if (conditional.quantifier.empty()) conditional.quantifier = quantifier_for(form);
                    if (negation_token(form)) conditional.negated = true;
                }
                doc.conditionals.push_back(std::move(conditional));
            }
        }
    }
    return doc;
}

ParsedDocument TextParser::parse_text(const std::string& text) const {
    ParsedDocument doc;
    auto sents = split_sentences(text);
    for (const auto& s : sents) {
        auto toks = tokenize(s);
        if (toks.empty()) continue;
        doc.sentences_tokens.push_back(toks);

        // Detect negation & modality in the sentence
        bool sent_negated = false;
        std::string sent_modality;
        GramTense sent_tense = GramTense::PRES;
        for (const auto& t : toks) {
            if (negation_token(t)) sent_negated = true;
            if (modal_token(t)) sent_modality = t;
            if (future_auxiliaries_.count(t)) {
                sent_tense = GramTense::FUT;
            } else if (past_tense_markers_.count(t)) {
                sent_tense = GramTense::PAST;
            }
        }

        const auto lower_sentence = utf8_tolower(s);
        const bool hypothetical = lower_sentence.starts_with("se ") || lower_sentence.starts_with("if ");
        struct LeadingMarker { const char* prefix; const char* kind; };
        static const LeadingMarker leading_markers[] = {
            {"se ", "CONDITIONAL"}, {"if ", "CONDITIONAL"},
            {"quando ", "TEMPORAL"}, {"when ", "TEMPORAL"},
            {"enquanto ", "TEMPORAL"}, {"while ", "TEMPORAL"},
            {"embora ", "CONCESSIVE"}, {"although ", "CONCESSIVE"}, {"though ", "CONCESSIVE"},
            {"caso ", "CONDITIONAL"}
        };
        const LeadingMarker* leading = nullptr;
        for (const auto& marker : leading_markers)
            if (lower_sentence.starts_with(marker.prefix)) { leading = &marker; break; }
        // Conditional, temporal and concessive clauses are rules, not asserted
        // facts. Both "se X então Y" and the comma form are recognized.
        if (leading) {
            const size_t start = std::char_traits<char>::length(leading->prefix);
            std::string antecedent, consequent;
            auto marker = lower_sentence.find(" então ");
            size_t width = std::string(" então ").size();
            if (marker == std::string::npos) { marker = lower_sentence.find(" then "); width = 6; }
            if (marker == std::string::npos) { marker = lower_sentence.find(", "); width = 2; }
            if (marker != std::string::npos && marker > start) {
                antecedent = lower_sentence.substr(start, marker - start);
                consequent = lower_sentence.substr(marker + width);
                while (!antecedent.empty() && antecedent.back() == ' ') antecedent.pop_back();
                while (!consequent.empty() && (consequent.back() == '.' || consequent.back() == ' ')) consequent.pop_back();
                if (!antecedent.empty() && !consequent.empty()) {
                    ParsedConditional conditional;
                    conditional.antecedent = antecedent;
                    conditional.consequent = consequent;
                    const std::string kind_text = leading->kind;
                    conditional.kind = kind_text == "TEMPORAL" ? ConditionKind::TEMPORAL :
                        kind_text == "CONCESSIVE" ? ConditionKind::CONCESSIVE : ConditionKind::CONDITIONAL;
                    for (const auto& token : tokenize(antecedent)) {
                        if (conditional.quantifier.empty()) conditional.quantifier = quantifier_for(token);
                        if (negation_token(token)) conditional.negated = true;
                    }
                    doc.conditionals.push_back(std::move(conditional));
                }
            }
            continue;
        }

        // Try matching a linguistic marker
        bool matched = false;
        for (const auto& rule : markers_) {
            const size_t mlen = rule.words.size();
            if (toks.size() < mlen + 2) continue;

            for (size_t pos = 1; pos + mlen < toks.size(); ++pos) {
                // Do not treat a verb inside a relative clause as the main
                // predicate of the sentence ("O calor, que derrete gelo, ...").
                bool inside_relative = false;
                for (size_t k = 1; k < pos; ++k) {
                    if (!relative_pronouns_.count(toks[k]))
                        continue;
                    std::string ignored;
                    if (k > 0 && !general_verb_lemma(toks[k - 1], ignored)) { inside_relative = true; break; }
                }
                if (inside_relative) continue;
                bool eq = true;
                for (size_t j = 0; j < mlen; ++j) {
                    if (toks[pos + j] != rule.words[j] &&
                        !(mlen == 1 && lemmatize_simple(toks[pos + j]) == lemmatize_simple(rule.words[j]))) {
                        eq = false;
                        break;
                    }
                }
                if (eq) {
                    // Exclude negation/modality tokens from subject/object spans
                    size_t lhs_end = pos;
                    while (lhs_end > 0 && (negation_token(toks[lhs_end - 1]) ||
                                           modal_token(toks[lhs_end - 1]))) {
                        --lhs_end;
                    }
                    std::string lhs = clean_phrase(toks, 0, lhs_end);
                    std::string rhs = clean_phrase(toks, pos + mlen, toks.size());
                    if (!lhs.empty() && !rhs.empty()) {
                        ParsedTriple tr;
                        tr.source_text = s;
                        tr.type = rule.type;
                        tr.confidence = 0.85f;
                        tr.negated = sent_negated;
                        tr.tense = sent_tense;
                        tr.modality = sent_modality;
                        if (rule.reverse_order) {
                            tr.subject = rhs;
                            tr.relation = "causa";
                            tr.object = lhs;
                        } else {
                            tr.subject = lhs;
                            tr.relation = mlen == 1 ? lemmatize_simple(toks[pos]) : rule.phrase;
                            if (mlen == 1) {
                                for (auto tense : {GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
                                    for (auto number : {GramNumber::SING, GramNumber::PLUR})
                                        if (MorphologicalRealizer::conjugate_verb(tr.relation, tense, number) == toks[pos])
                                            tr.tense = tense;
                            }
                            tr.object = rhs;
                        }
                        tr.subj_gender = infer_gender(tr.subject);
                        tr.subj_number = infer_number(tr.subject);
                        tr.obj_gender = infer_gender(tr.object);
                        tr.obj_number = infer_number(tr.object);
                        if (!rule.reverse_order) {
                            agree_with_article(toks, 0, tr.subj_gender, tr.subj_number);
                            agree_with_article(toks, pos + mlen, tr.obj_gender, tr.obj_number);
                        }
                        doc.triples.push_back(tr);

                        // Extract abstract construction frame
                        ConstructionFrame cf;
                        cf.rel_type = rule.type;
                        cf.pattern = "[SUBJ:ENT] " + rule.phrase + " [OBJ:ENT]";
                        cf.frequency = 1;
                        doc.frames.push_back(std::move(cf));

                        matched = true;
                        break;
                    }
                }
            }
            if (matched) break;
        }
        if (matched || hypothetical) continue;

        // Interrogatives and imperatives do not assert facts.
        bool question = s.find('?') != std::string::npos;
        for (const auto& start : question_starts_)
            if (lower_sentence.starts_with(start)) { question = true; break; }
        if (question) continue;

        auto classify_relation = [&](const std::string& lemma) {
            const auto canonical = lemmatize_simple(lemma);
            for (const auto& rule : markers_)
                if (rule.phrase == lemma || lemmatize_simple(rule.phrase) == canonical) return rule.type;
            return RelationType::ACTION;
        };

        // General dependency-free SVO fallback: finds the first known verb and
        // uses the material before/after it as subject and complement. This
        // extends the parser beyond fixed causal markers without pretending to
        // be a full dependency parser.
        std::function<void(const std::vector<std::string>&, int)> parse_general;
        parse_general = [&](const std::vector<std::string>& parts, int depth) {
            if (parts.empty()) return;
            std::string verb;
            size_t verb_at = find_main_verb(parts, verb);
            if (verb_at == std::string::npos) return;
            // Coordination: "A faz X, e B faz Y" -> two independent clauses.
            if (depth < 2) {
                for (size_t cut = 1; cut + 1 < parts.size(); ++cut) {
                    if (parts[cut] != "e" && parts[cut] != "and" && parts[cut] != "mas" &&
                        parts[cut] != "but" && parts[cut] != "porém" && parts[cut] != "however") continue;
                    std::string left_verb, right_verb;
                    if (find_main_verb(parts, left_verb) >= cut) continue;
                    std::vector<std::string> right(parts.begin() + cut + 1, parts.end());
                    if (find_main_verb(right, right_verb) == std::string::npos) continue;
                    std::vector<std::string> left(parts.begin(), parts.begin() + cut);
                    parse_general(left, depth + 1);
                    parse_general(right, depth + 1);
                    return;
                }
            }
            const auto subject = clean_span(parts, 0, verb_at);
            if (subject.empty()) return;
            // Multiple coordinated complements: "gera calor e luz".
            std::vector<std::vector<std::string>> complement_groups;
            std::vector<std::string> current;
            for (size_t i = verb_at + 1; i < parts.size(); ++i) {
                if (parts[i] == "e" || parts[i] == "and" || parts[i] == "ou" || parts[i] == "or") {
                    if (!current.empty()) { complement_groups.push_back(current); current.clear(); }
                    continue;
                }
                current.push_back(parts[i]);
            }
            if (!current.empty()) complement_groups.push_back(std::move(current));
            for (const auto& group : complement_groups) {
                const auto object = clean_span(group, 0, group.size());
                if (object.empty() || object == subject) continue;
                ParsedTriple tr;
                tr.source_text = s;
                tr.subject = subject;
                tr.relation = verb;
                tr.object = object;
                tr.type = classify_relation(verb);
                tr.confidence = 0.65f;
                tr.negated = sent_negated;
                tr.tense = sent_tense;
                tr.modality = sent_modality;
                for (auto tense : {GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
                    for (auto number : {GramNumber::SING, GramNumber::PLUR})
                        if (MorphologicalRealizer::conjugate_verb(verb, tense, number) == parts[verb_at])
                            tr.tense = tense;
                tr.subj_gender = infer_gender(subject);
                tr.subj_number = infer_number(subject);
                tr.obj_gender = infer_gender(object);
                tr.obj_number = infer_number(object);
                agree_with_article(parts, 0, tr.subj_gender, tr.subj_number);
                agree_with_article(parts, verb_at + 1, tr.obj_gender, tr.obj_number);
                doc.triples.push_back(tr);
                doc.frames.push_back({"[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]", tr.type, 1});
            }
        };
        bool relative_sentence = false;
        for (size_t k = 1; k < toks.size(); ++k) {
            if (!relative_pronouns_.count(toks[k])) continue;
            std::string ignored;
            if (!general_verb_lemma(toks[k - 1], ignored)) { relative_sentence = true; break; }
        }
        if (!relative_sentence) parse_general(toks, 0);
    }

    // Merge the embedded dependency parse. Marker rules remain authoritative for
    // their exact triples (higher precision), while dependency triples add
    // relative clauses, quantifiers, advcl metadata and general SVO coverage.
    auto dependencies = parse_dependencies(text);
    if (!dependencies.morph_sentences.empty()) doc.morph_sentences = std::move(dependencies.morph_sentences);
    auto same_conditional = [&](const ParsedConditional& candidate) {
        for (const auto& existing : doc.conditionals)
            if (existing.antecedent == candidate.antecedent && existing.consequent == candidate.consequent &&
                existing.kind == candidate.kind) return true;
        return false;
    };
    for (const auto& conditional : dependencies.conditionals)
        if (!same_conditional(conditional)) doc.conditionals.push_back(conditional);
    auto relation_lemma = [&](const std::string& relation) {
        std::string lemma;
        if (general_verb_lemma(relation, lemma)) return lemma;
        return TextParser::lemmatize_simple(relation);
    };
    for (const auto& triple : dependencies.triples) {
        // "motor de combustão interna" must not yield the object
        // "combustão interna"; the PP belongs to the subject.
        if (!triple.object.empty() && triple.object != triple.subject &&
            triple.subject.find(triple.object) != std::string::npos) continue;
        auto existing = std::find_if(doc.triples.begin(), doc.triples.end(), [&](const ParsedTriple& value) {
            return value.subject == triple.subject &&
                   relation_lemma(value.relation) == relation_lemma(triple.relation) &&
                   value.object == triple.object;
        });
        if (existing != doc.triples.end()) {
            if (existing->quantifier.empty()) existing->quantifier = triple.quantifier;
            if (!existing->has_condition && triple.has_condition) {
                existing->has_condition = true;
                existing->condition_kind = triple.condition_kind;
            }
            if (existing->relative_antecedent.empty()) existing->relative_antecedent = triple.relative_antecedent;
            continue;
        }
        // The rule fallback often keeps the whole clause as the object
        // ("trabalho pela expansão..."). The dependency parse has a cleaner
        // NP. When they share subject+relation, prefer the dependency triple.
        auto weaker = std::find_if(doc.triples.begin(), doc.triples.end(), [&](const ParsedTriple& value) {
            if (value.subject != triple.subject) return false;
            const bool same_type = value.type == triple.type &&
                (triple.type == RelationType::TAXONOMIC || triple.type == RelationType::PROPERTY);
            if (!same_type && relation_lemma(value.relation) != relation_lemma(triple.relation)) return false;
            if (value.object == triple.object) return false;
            return value.object.find(triple.object) != std::string::npos ||
                   triple.object.find(value.object) != std::string::npos;
        });
        if (weaker != doc.triples.end()) {
            *weaker = triple;
            continue;
        }
        doc.triples.push_back(triple);
    }
    for (const auto& frame : dependencies.frames) {
        const bool exists = std::any_of(doc.frames.begin(), doc.frames.end(), [&](const ConstructionFrame& current) {
            return current.pattern == frame.pattern && current.rel_type == frame.rel_type;
        });
        if (!exists) doc.frames.push_back(frame);
    }
    return doc;

}


ParsedDocument TextParser::parse_conllu(const std::string& conllu_text) const {
    ParsedDocument doc;
    std::istringstream input(conllu_text);
    std::string line;
    std::vector<MorphToken> sentence;
    auto base = [](const std::string& rel) { return rel.substr(0, rel.find(':')); };
    auto negative = [&](const MorphToken& t) {
        return t.negated || negation_token(TextParser::utf8_tolower(t.lemma));
    };
    auto modal = [&](const MorphToken& t) {
        const auto w = TextParser::utf8_tolower(t.lemma);
        return modal_token(w) || w == "poder" || w == "dever";
    };
    auto flush = [&]() {
        if (sentence.empty()) return;
        std::unordered_set<int> ids;
        for (const auto& t : sentence) ids.insert(t.id);
        for (const auto& t : sentence) {
            if (t.head && !ids.count(t.head)) throw std::invalid_argument("CoNLL-U: dangling dependency head");
            std::unordered_set<int> visited;
            const MorphToken* current = &t;
            while (current && current->head) {
                if (!visited.insert(current->id).second) throw std::invalid_argument("CoNLL-U: dependency cycle");
                const int parent = current->head;
                current = nullptr;
                for (const auto& candidate : sentence) if (candidate.id == parent) { current = &candidate; break; }
            }
        }
        std::vector<std::string> words;
        for (const auto& t : sentence) words.push_back(utf8_tolower(t.form));
        doc.sentences_tokens.push_back(std::move(words));
        auto token = [&](int id) -> const MorphToken* {
            for (const auto& t : sentence) if (t.id == id) return &t;
            return nullptr;
        };
        // Keep nominal modifiers and surface inflection, omitting determiners.
        auto phrase = [&](const MorphToken& head) {
            std::unordered_set<int> ids{head.id};
            for (size_t pass = 0; pass < sentence.size(); ++pass) {
                bool changed = false;
                for (const auto& t : sentence) {
                    const auto r = base(t.deprel);
                    if (ids.count(t.head) && !ids.count(t.id) &&
                        (r == "amod" || r == "compound" || r == "flat" || r == "nmod" ||
                         r == "case" || r == "nummod")) {
                        ids.insert(t.id); changed = true;
                    }
                }
                if (!changed) break;
            }
            std::string text;
            for (const auto& t : sentence) if (ids.count(t.id)) {
                if (!text.empty()) text += " ";
                text += utf8_tolower(t.form);
            }
            return text;
        };
        for (const auto& v : sentence) {
            const MorphToken *subj = nullptr, *obj = nullptr, *cop = nullptr;
            const MorphToken* governor = token(v.head);
            bool neg = negative(v);
            std::string modality;
            auto tense = v.tense;
            for (const auto& child : sentence) {
                if (child.head != v.id) continue;
                auto r = base(child.deprel);
                if (r == "nsubj") subj = &child;
                if (r == "obj" || (!obj && r == "obl")) obj = &child;
                if (r == "cop") cop = &child;
                if (negative(child)) neg = true;
                if (r == "aux") {
                    if (modal(child)) {
                        modality = utf8_tolower(child.lemma);
                        if (tense == GramTense::INF) tense = child.tense;
                    }
                    if (child.tense != GramTense::PRES) tense = child.tense;
                    if (future_auxiliaries_.count(child.lemma)) tense = GramTense::FUT;
                }
            }
            if (v.upos != "VERB" && !cop) continue;
            // UD represents modal constructions either as aux or as VERB+xcomp.
            if (modal(v)) continue;
            if (!subj && governor && (base(v.deprel) == "xcomp" || base(v.deprel) == "conj")) {
                for (const auto& child : sentence) {
                    if (child.head == governor->id && base(child.deprel) == "nsubj") subj = &child;
                    if (child.head == governor->id && negative(child)) neg = true;
                }
                if (modal(*governor)) {
                    modality = utf8_tolower(governor->lemma);
                    tense = governor->tense;
                }
            }
            if (cop) { obj = &v; tense = cop->tense; }
            if (!subj || !obj) continue;
            ParsedTriple tr;
            for (const auto& word : sentence) {
                if (!tr.source_text.empty() && word.upos != "PUNCT") tr.source_text += " ";
                tr.source_text += word.form;
            }
            tr.subject = phrase(*subj);
            tr.object = phrase(*obj);
            tr.relation = utf8_tolower(cop ? cop->lemma : v.lemma);
            tr.negated = neg;
            tr.tense = tense;
            tr.modality = modality;
            tr.subj_gender = subj->gender;
            tr.subj_number = subj->number;
            tr.obj_gender = obj->gender;
            tr.obj_number = obj->number;
            tr.confidence = 0.92f;
            tr.type = RelationType::ACTION;
            for (const auto& rule : markers_) {
                if (rule.phrase == tr.relation || lemmatize_simple(rule.phrase) == tr.relation) {
                    tr.type = rule.type; break;
                }
            }
            if (cop) tr.type = v.upos == "ADJ" ? RelationType::PROPERTY : RelationType::TAXONOMIC;
            doc.frames.push_back({"[SUBJ:ENT] " + tr.relation + " [OBJ:ENT]", tr.type, 1});
            doc.triples.push_back(std::move(tr));
        }
        doc.morph_sentences.push_back(std::move(sentence));
        sentence.clear();
    };
    size_t line_number = 0;
    auto integer = [&](const std::string& field) {
        int value = 0;
        const auto result = std::from_chars(field.data(), field.data() + field.size(), value);
        if (result.ec != std::errc{} || result.ptr != field.data() + field.size() || value < 0)
            throw std::invalid_argument("CoNLL-U: invalid integer at line " + std::to_string(line_number));
        return value;
    };
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) { flush(); continue; }
        if (line[0] == '#') continue;
        std::vector<std::string> cols;
        std::istringstream row(line);
        std::string col;
        while (std::getline(row, col, '\t')) cols.push_back(col);
        if (cols.size() != 10) throw std::invalid_argument("CoNLL-U: expected 10 columns at line " + std::to_string(line_number));
        if (cols[0].find_first_of("-.") != std::string::npos) continue;
        MorphToken t;
        t.id = integer(cols[0]);
        if (t.id == 0 || (!sentence.empty() && t.id <= sentence.back().id))
            throw std::invalid_argument("CoNLL-U: unordered token IDs at line " + std::to_string(line_number));
        t.form = cols[1]; t.lemma = cols[2] == "_" ? cols[1] : cols[2]; t.upos = cols[3];
        const auto& feats = cols[5];
        if (feats.find("Mood=Sub") != std::string::npos) t.tense = GramTense::SUBJ;
        else if (feats.find("Tense=Past") != std::string::npos) t.tense = GramTense::PAST;
        else if (feats.find("Tense=Fut") != std::string::npos) t.tense = GramTense::FUT;
        else if (feats.find("VerbForm=Inf") != std::string::npos) t.tense = GramTense::INF;
        t.gender = feats.find("Gender=Fem") != std::string::npos ? GramGender::FEM :
                   feats.find("Gender=Neut") != std::string::npos ? GramGender::NEUT : GramGender::MASC;
        if (feats.find("Number=Plur") != std::string::npos) t.number = GramNumber::PLUR;
        t.negated = feats.find("Polarity=Neg") != std::string::npos;
        t.head = integer(cols[6]); t.deprel = cols[7];
        sentence.push_back(std::move(t));
    }
    flush();
    return doc;
}

} // namespace cognicore
