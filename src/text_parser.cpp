#include "core/text_parser.hpp"
#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <charconv>
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

} // namespace

TextParser::TextParser() {
    init_default_rules();
}

void TextParser::init_default_rules() {
    markers_.clear();
    stopwords_ = {
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "por", "para", "com", "sem", "sob", "sobre", "que", "se", "e", "ou",
        "the", "an", "of", "in", "on", "at", "to", "for", "with", "by", "and", "or", "that", "which",
        "é", "são", "foi", "ser", "como", "quando", "qual", "quais", "what", "why", "how", "is", "does"
    };

    auto add_list = [&](const std::vector<std::string>& list, RelationType t) {
        for (const auto& m : list) {
            MarkerRule r;
            r.phrase = utf8_tolower(m);
            r.words = tokenize(r.phrase);
            r.type = t;
            r.reverse_order = (r.phrase == "porque" || r.phrase == "pois" || r.phrase == "because");
            if (!r.words.empty()) markers_.push_back(std::move(r));
        }
    };

    add_list({"porque", "pois", "causa", "causar", "provoca", "provocar", "gera", "gerar",
              "produz", "produzir", "leva a", "resulta em", "faz com que", "acarreta",
              "origina", "logo", "portanto",
              "causes", "cause", "leads to", "results in", "produces", "produce",
              "generates", "generate", "triggers", "because"},
             RelationType::CAUSAL);
    add_list({"impede", "impedir", "bloqueia", "bloquear", "evita", "evitar",
              "reduz", "reduzir", "diminui", "neutraliza", "apaga", "apagar",
              "prevents", "prevent", "blocks", "block", "inhibits", "reduces", "stops"},
             RelationType::INHIBITS);
    add_list({"é um", "é uma", "são", "possui", "tem", "contém", "faz parte de",
              "is a", "is an", "are", "has", "contains", "part of"},
             RelationType::TAXONOMIC);
    add_list({"derrete", "derreter", "aquece", "aquecer", "esfria", "esfriar",
              "queima", "queimar", "evapora", "evaporar", "congela", "congelar", "transforma",
              "melts", "melt", "heats", "heat", "cools", "cool", "burns", "burn",
              "evaporates", "freezes", "transforms"},
             RelationType::ACTION);

    std::sort(markers_.begin(), markers_.end(),
              [](const MarkerRule& a, const MarkerRule& b) {
                  return a.words.size() > b.words.size();
              });
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
    auto stops = extract_json_array(json, "stopwords");

    if (!stops.empty()) {
        for (const auto& s : stops) stopwords_.insert(utf8_tolower(s));
    }

    if (!causal.empty() || !inhib.empty() || !taxon.empty() || !action.empty()) {
        markers_.clear();
        auto add_list = [&](const std::vector<std::string>& list, RelationType t) {
            for (const auto& m : list) {
                MarkerRule r;
                r.phrase = utf8_tolower(m);
                r.words = tokenize(r.phrase);
                r.type = t;
                r.reverse_order = (r.phrase == "porque" || r.phrase == "pois" || r.phrase == "because");
                if (!r.words.empty()) markers_.push_back(std::move(r));
            }
        };
        add_list(causal, RelationType::CAUSAL);
        add_list(inhib, RelationType::INHIBITS);
        add_list(taxon, RelationType::TAXONOMIC);
        add_list(action, RelationType::ACTION);
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
    if (start >= end) return "";

    std::string res;
    for (size_t i = start; i < end; ++i) {
        if (!res.empty()) res.push_back(' ');
        res += tokens[i];
    }
    return res;
}

GramGender TextParser::infer_gender(const std::string& word) {
    const std::string w = utf8_tolower(word);
    if (w.size() >= 2) {
        if (w == "água" || w == "agua" || w == "energia" || w == "luz" || w == "chama" ||
            w == "chuva" || w == "nuvem" || w == "nuvens" || w == "gravidade" ||
            w == "evaporação" || w == "evaporacao" || w == "queda" || w == "força" || w == "forca") {
            return GramGender::FEM;
        }
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

GramNumber TextParser::infer_number(const std::string& word) {
    const std::string w = utf8_tolower(word);
    if (w.size() >= 4 && w.back() == 's') {
        if (w == "gás" || w == "gas" || w == "luz" || w == "país" || w == "mais" || w == "menos") {
            return GramNumber::SING;
        }
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
    for (const auto& lemma : verbs)
        for (auto tense : {GramTense::PRES, GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
            for (auto number : {GramNumber::SING, GramNumber::PLUR})
                if (MorphologicalRealizer::conjugate_verb(lemma, tense, number) == w) return lemma;
    return w;
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
            if (t == "não" || t == "nao" || t == "nunca" || t == "jamais" ||
                t == "not" || t == "never" || t == "don't" || t == "doesn't") {
                sent_negated = true;
            }
            if (t == "pode" || t == "podem" || t == "costuma" || t == "costumam" ||
                t == "deve" || t == "devem" || t == "can" || t == "may" || t == "must") {
                sent_modality = t;
            }
            if (t == "vai" || t == "irá" || t == "ira" || t == "will") {
                sent_tense = GramTense::FUT;
            } else if (t == "foi" || t == "gerou" || t == "causou" || t == "derreteu" || t == "apagou") {
                sent_tense = GramTense::PAST;
            }
        }

        // Try matching a linguistic marker
        bool matched = false;
        for (const auto& rule : markers_) {
            const size_t mlen = rule.words.size();
            if (toks.size() < mlen + 2) continue;

            for (size_t pos = 1; pos + mlen < toks.size(); ++pos) {
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
                    while (lhs_end > 0 && (toks[lhs_end - 1] == "não" || toks[lhs_end - 1] == "nao" ||
                                           toks[lhs_end - 1] == "nunca" || toks[lhs_end - 1] == "pode" ||
                                           toks[lhs_end - 1] == "costuma" || toks[lhs_end - 1] == "deve" ||
                                           toks[lhs_end - 1] == "can" || toks[lhs_end - 1] == "not" ||
                                           toks[lhs_end - 1] == "never" || toks[lhs_end - 1] == "jamais" ||
                                           toks[lhs_end - 1] == "podem" || toks[lhs_end - 1] == "devem" ||
                                           toks[lhs_end - 1] == "costumam" || toks[lhs_end - 1] == "will" ||
                                           toks[lhs_end - 1] == "vai" || toks[lhs_end - 1] == "irá")) {
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
                        auto agree_article = [&](size_t begin, GramGender& gender, GramNumber& number) {
                            if (begin >= toks.size()) return;
                            const auto& article = toks[begin];
                            if (article == "a" || article == "as" || article == "uma" || article == "umas") gender = GramGender::FEM;
                            if (article == "o" || article == "os" || article == "um" || article == "uns") gender = GramGender::MASC;
                            if (article == "as" || article == "os" || article == "umas" || article == "uns") number = GramNumber::PLUR;
                        };
                        if (!rule.reverse_order) {
                            agree_article(0, tr.subj_gender, tr.subj_number);
                            agree_article(pos + mlen, tr.obj_gender, tr.obj_number);
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
    }
    return doc;
}

ParsedDocument TextParser::parse_conllu(const std::string& conllu_text) const {
    ParsedDocument doc;
    std::istringstream input(conllu_text);
    std::string line;
    std::vector<MorphToken> sentence;
    auto base = [](const std::string& rel) { return rel.substr(0, rel.find(':')); };
    auto negative = [](const MorphToken& t) {
        const auto w = TextParser::utf8_tolower(t.lemma);
        return t.negated || w == "não" || w == "nunca" || w == "jamais" ||
               w == "not" || w == "never";
    };
    auto modal = [](const MorphToken& t) {
        const auto w = TextParser::utf8_tolower(t.lemma);
        return w == "poder" || w == "dever" || w == "can" || w == "may" ||
               w == "must" || w == "could" || w == "should";
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
                    if (child.lemma == "will") tense = GramTense::FUT;
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
