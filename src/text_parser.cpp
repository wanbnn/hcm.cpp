#include "core/text_parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

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

    add_list({"porque", "pois", "causa", "provoca", "gera", "produz", "leva a",
              "resulta em", "faz com que", "acarreta", "origina", "logo", "portanto",
              "causes", "leads to", "results in", "produces", "generates", "triggers", "because"},
             RelationType::CAUSAL);
    add_list({"impede", "bloqueia", "evita", "reduz", "diminui", "neutraliza", "apaga",
              "prevents", "blocks", "inhibits", "reduces", "stops"},
             RelationType::INHIBITS);
    add_list({"é um", "é uma", "são", "possui", "tem", "contém", "faz parte de",
              "is a", "is an", "are", "has", "contains", "part of"},
             RelationType::TAXONOMIC);
    add_list({"derrete", "aquece", "esfria", "queima", "evapora", "congela", "transforma",
              "melts", "heats", "cools", "burns", "evaporates", "freezes", "transforms"},
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

ParsedDocument TextParser::parse_text(const std::string& text) const {
    ParsedDocument doc;
    auto sents = split_sentences(text);
    for (const auto& s : sents) {
        auto toks = tokenize(s);
        if (toks.empty()) continue;
        doc.sentences_tokens.push_back(toks);

        // Try matching a linguistic marker
        bool matched = false;
        for (const auto& rule : markers_) {
            const size_t mlen = rule.words.size();
            if (toks.size() < mlen + 2) continue;

            for (size_t pos = 1; pos + mlen < toks.size(); ++pos) {
                bool eq = true;
                for (size_t j = 0; j < mlen; ++j) {
                    if (toks[pos + j] != rule.words[j]) {
                        eq = false;
                        break;
                    }
                }
                if (eq) {
                    std::string lhs = clean_phrase(toks, 0, pos);
                    std::string rhs = clean_phrase(toks, pos + mlen, toks.size());
                    if (!lhs.empty() && !rhs.empty()) {
                        ParsedTriple tr;
                        tr.type = rule.type;
                        tr.confidence = 0.85f;
                        if (rule.reverse_order) {
                            tr.subject = rhs;
                            tr.relation = "causa";
                            tr.object = lhs;
                        } else {
                            tr.subject = lhs;
                            tr.relation = rule.phrase;
                            tr.object = rhs;
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

} // namespace cognicore
