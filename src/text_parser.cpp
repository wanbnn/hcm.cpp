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

// General (non-marker) lexical verb recognizer. It is deliberately
// conservative: a token is a verb only if it appears in a known conjugation
// table or has an unambiguous past-tense ending. This keeps the rule fallback
// from turning ordinary nouns into predicates.
const std::unordered_map<std::string, std::string>& known_verb_forms() {
    static const auto forms = [] {
        static const std::vector<std::string> pt_lemmas = {
            "analisar", "observar", "estudar", "pesquisar", "medir", "calcular", "mover",
            "empurrar", "puxar", "criar", "construir", "destruir", "quebrar", "consertar",
            "usar", "aplicar", "incluir", "excluir", "abrir", "fechar", "ler", "escrever",
            "falar", "dizer", "ver", "ouvir", "pensar", "entender", "aprender", "ensinar",
            "comprar", "vender", "pagar", "receber", "enviar", "levar", "trazer", "comer",
            "beber", "dormir", "acordar", "andar", "correr", "nadar", "voar", "cair",
            "subir", "descer", "começar", "terminar", "continuar", "parar", "mudar",
            "aumentar", "diminuir", "melhorar", "piorar", "permitir", "proibir", "exigir",
            "precisar", "depender", "conter", "manter", "obter", "perder", "ganhar",
            "escolher", "decidir", "provar", "mostrar", "indicar", "representar",
            "significar", "acontecer", "ocorrer", "existir", "viver", "morrer", "nascer",
            "crescer", "ferver", "congelar", "queimar", "derreter", "evaporar",
            "condensar", "aquecer", "esfriar", "iluminar", "escurecer", "acelerar",
            "frear", "girar", "vibrar", "refletir", "absorver", "emitir", "conduzir",
            "isolar", "carregar", "descarregar", "armazenar", "processar", "transformar",
            "combinar", "separar", "dividir", "juntar", "somar", "subtrair", "multiplicar",
            "falhar", "caber", "superaquecer", "repartir", "sobrar", "faltar", "ocorrer",
            "gerar", "causar", "provocar", "produzir", "formar", "apagar", "bloquear",
            "reduzir", "evitar", "possuir", "ser", "ter", "poder", "dever", "fazer", "ir",
            "vir", "estar", "haver", "dar", "saber", "querer", "trazer", "seguir"
        };
        static const std::vector<std::string> en_lemmas = {
            "analyze", "observe", "study", "measure", "calculate", "move", "push", "pull",
            "create", "build", "destroy", "break", "fix", "use", "apply", "include",
            "exclude", "open", "close", "read", "write", "speak", "say", "see", "hear",
            "think", "understand", "learn", "teach", "buy", "sell", "pay", "receive",
            "send", "carry", "bring", "eat", "drink", "sleep", "wake", "walk", "run",
            "swim", "fly", "fall", "rise", "descend", "begin", "finish", "continue",
            "stop", "change", "increase", "decrease", "improve", "worsen", "allow",
            "forbid", "require", "need", "depend", "contain", "keep", "obtain", "lose",
            "win", "choose", "decide", "prove", "show", "indicate", "represent", "mean",
            "happen", "occur", "exist", "live", "die", "grow", "boil", "freeze", "burn",
            "melt", "evaporate", "condense", "heat", "cool", "illuminate", "accelerate",
            "brake", "spin", "vibrate", "reflect", "absorb", "emit", "conduct",
            "isolate", "load", "unload", "store", "process", "transform", "combine",
            "separate", "divide", "join", "add", "subtract", "multiply", "generate",
            "cause", "produce", "prevent", "block", "reduce", "avoid", "have", "be", "do",
            "go", "come", "make", "take", "get", "give", "know", "want"
        };
        std::unordered_map<std::string, std::string> out;
        for (const auto& lemma : pt_lemmas) {
            out.emplace(lemma, lemma);
            for (auto tense : {GramTense::PRES, GramTense::PAST, GramTense::FUT, GramTense::SUBJ})
                for (auto number : {GramNumber::SING, GramNumber::PLUR}) {
                    auto form = MorphologicalRealizer::conjugate_verb(lemma, tense, number, "pt");
                    if (!form.empty() && form != lemma) out.emplace(form, lemma);
                }
        }
        for (const auto& lemma : en_lemmas) {
            out.emplace(lemma, lemma);
            for (auto tense : {GramTense::PRES, GramTense::PAST, GramTense::FUT})
                for (auto number : {GramNumber::SING, GramNumber::PLUR}) {
                    auto form = MorphologicalRealizer::conjugate_verb(lemma, tense, number, "en");
                    if (!form.empty() && form != lemma) out.emplace(form, lemma);
                }
        }
        // Frequent irregulars that the generic conjugator cannot derive.
        const std::pair<const char*, const char*> irregulars[] = {
            {"é", "ser"}, {"são", "ser"}, {"foi", "ser"}, {"foram", "ser"}, {"era", "ser"},
            {"tem", "ter"}, {"têm", "ter"}, {"teve", "ter"}, {"tiveram", "ter"},
            {"faz", "fazer"}, {"fazem", "fazer"}, {"fez", "fazer"}, {"fizeram", "fazer"},
            {"vai", "ir"}, {"vão", "ir"}, {"foi", "ir"}, {"vem", "vir"}, {"vêm", "vir"},
            {"está", "estar"}, {"estão", "estar"}, {"estava", "estar"}, {"há", "haver"},
            {"dá", "dar"}, {"dão", "dar"}, {"deu", "dar"}, {"sabe", "saber"}, {"sabem", "saber"},
            {"quer", "querer"}, {"querem", "querer"}, {"põe", "pôr"}, {"põem", "pôr"},
            {"is", "be"}, {"are", "be"}, {"was", "be"}, {"were", "be"}, {"has", "have"},
            {"had", "have"}, {"does", "do"}, {"did", "do"}, {"goes", "go"}, {"went", "go"},
            {"makes", "make"}, {"made", "make"}, {"takes", "take"}, {"took", "take"},
            {"gets", "get"}, {"got", "get"}, {"gives", "give"}, {"gave", "give"}
        };
        for (const auto& [form, lemma] : irregulars) out.emplace(form, lemma);
        return out;
    }();
    return forms;
}

bool general_verb_lemma(const std::string& token, std::string& lemma) {
    if (token.size() < 2) return false;
    const auto& forms = known_verb_forms();
    if (auto it = forms.find(token); it != forms.end()) { lemma = it->second; return true; }
    // Unambiguous regular past endings (analytic languages rarely use these
    // endings for nouns).
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

std::string clean_span(const std::vector<std::string>& tokens, size_t start, size_t end,
                       const std::unordered_set<std::string>& stopwords) {
    static const std::unordered_set<std::string> extra = {
        "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
        "este", "esta", "esse", "essa", "isso", "isto", "aquele", "aquela",
        "only", "just", "muito", "muita", "muitos", "muitas", "só", "so"
    };
    while (start < end && (stopwords.count(tokens[start]) || extra.count(tokens[start]))) ++start;
    while (end > start && (stopwords.count(tokens[end - 1]) || extra.count(tokens[end - 1]))) --end;
    std::string out;
    for (size_t i = start; i < end; ++i) {
        if (!out.empty()) out.push_back(' ');
        out += tokens[i];
    }
    return out;
}

size_t find_main_verb(const std::vector<std::string>& tokens, std::string& lemma, size_t from = 1) {
    for (size_t i = from; i < tokens.size(); ++i) {
        if (general_verb_lemma(tokens[i], lemma)) return i;
    }
    return std::string::npos;
}

void agree_with_article(const std::vector<std::string>& tokens, size_t begin,
                        GramGender& gender, GramNumber& number) {
    if (begin >= tokens.size()) return;
    const auto& article = tokens[begin];
    if (article == "a" || article == "as" || article == "uma" || article == "umas") gender = GramGender::FEM;
    if (article == "o" || article == "os" || article == "um" || article == "uns") gender = GramGender::MASC;
    if (article == "as" || article == "os" || article == "umas" || article == "uns") number = GramNumber::PLUR;
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

namespace {

std::string quantifier_for(const std::string& token) {
    static const std::unordered_map<std::string, std::string> table = {
        {"todo", "ALL"}, {"todos", "ALL"}, {"toda", "ALL"}, {"todas", "ALL"},
        {"qualquer", "ANY"}, {"quaisquer", "ANY"}, {"all", "ALL"}, {"every", "ALL"},
        {"any", "ANY"}, {"algum", "SOME"}, {"alguma", "SOME"}, {"alguns", "SOME"},
        {"algumas", "SOME"}, {"some", "SOME"}, {"nenhum", "NO"}, {"nenhuma", "NO"},
        {"none", "NO"}, {"no", "NO"}, {"maioria", "MOST"}, {"most", "MOST"},
        {"cada", "EACH"}, {"each", "EACH"}
    };
    if (auto it = table.find(token); it != table.end()) return it->second;
    return {};
}

bool negation_token(const std::string& token) {
    return token == "não" || token == "nao" || token == "nunca" || token == "jamais" ||
           token == "not" || token == "never" || token == "no";
}

bool modal_token(const std::string& token) {
    return token == "pode" || token == "podem" || token == "podia" || token == "poderia" ||
           token == "deve" || token == "devem" || token == "costuma" || token == "costumam" ||
           token == "can" || token == "could" || token == "may" || token == "might" ||
           token == "must" || token == "should" || token == "will" || token == "would";
}

bool copula_token(const std::string& lemma) {
    return lemma == "ser" || lemma == "estar" || lemma == "be" || lemma == "parecer" || lemma == "seem";
}

bool adjective_token(const std::string& token) {
    static const char* known[] = {"quente", "frio", "fria", "grande", "pequeno", "pequena",
        "rápido", "lenta", "lento", "lenta", "novo", "nova", "antigo", "antiga", "sólido",
        "sólida", "liquido", "líquido", "gasoso", "corrompido", "seguro", "vazio", "cheio",
        "hot", "cold", "large", "small", "fast", "slow", "new", "old", "solid", "liquid"};
    for (const auto* word : known) if (token == word) return true;
    return token.size() > 4 && (ends_with(token, "oso") || ends_with(token, "osa") ||
        ends_with(token, "ivo") || ends_with(token, "iva") || ends_with(token, "vel") ||
        ends_with(token, "al") || ends_with(token, "il") || ends_with(token, "nte"));
}

ConditionKind condition_kind_for(const std::string& marker) {
    if (marker == "quando" || marker == "enquanto" || marker == "when" || marker == "while" ||
        marker == "antes" || marker == "depois" || marker == "before" || marker == "after")
        return ConditionKind::TEMPORAL;
    if (marker == "porque" || marker == "pois" || marker == "because" || marker == "since" ||
        marker == "já que" || marker == "como")
        return ConditionKind::CAUSAL;
    if (marker == "embora" || marker == "apesar" || marker == "although" || marker == "though" ||
        marker == "even though")
        return ConditionKind::CONCESSIVE;
    if (marker == "então" || marker == "then" || marker == "portanto" || marker == "therefore")
        return ConditionKind::CONSECUTIVE;
    return ConditionKind::CONDITIONAL;
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

} // namespace

ParsedDocument TextParser::parse_dependencies(const std::string& text) const {
    ParsedDocument doc;
    static const std::unordered_set<std::string> articles = {
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "the", "an"
    };
    static const std::unordered_set<std::string> prepositions = {
        "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "por", "pelo",
        "pela", "para", "com", "sem", "sob", "sobre", "entre", "até", "of", "in", "on",
        "at", "to", "for", "with", "by", "without", "from", "into"
    };
    static const std::unordered_set<std::string> pronouns = {
        "ele", "ela", "eles", "elas", "isso", "isto", "esse", "essa", "este", "esta",
        "aquele", "aquela", "eu", "tu", "nós", "você", "vocês",
        "he", "she", "it", "they", "this", "that", "these", "those", "we", "you", "i"
    };
    static const std::unordered_set<std::string> relative_pronouns = {
        "que", "quem", "qual", "quais", "which", "who", "whom", "that", "cujo", "cuja"
    };
    static const std::unordered_set<std::string> subordinators = {
        "se", "if", "quando", "when", "enquanto", "while", "porque", "because", "pois",
        "embora", "although", "though", "caso", "since", "unless", "sempre"
    };
    static const std::unordered_set<std::string> conjunctions = {
        "e", "ou", "mas", "porém", "contudo", "entretanto", "and", "or", "but", "however"
    };
    static const std::unordered_set<std::string> adverbs = {
        "não", "nao", "nunca", "jamais", "sempre", "muito", "muita", "muitos", "muitas",
        "só", "so", "também", "tambem", "já", "ja", "ainda", "aqui", "lá", "la", "ali",
        "ontem", "hoje", "amanhã", "amanha", "rapidamente", "lentamente", "bem", "mal",
        "not", "never", "always", "very", "only", "also", "already", "still", "here",
        "there", "yesterday", "today", "tomorrow", "quickly", "slowly", "well", "badly",
        "mais", "menos", "more", "less", "quase", "almost"
    };
    static const std::unordered_set<std::string> phrase_deprels = {
        "det", "nummod", "amod", "compound", "flat", "nmod"
    };

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
            if (articles.count(token.form)) { token.upos = "DET"; continue; }
            if (prepositions.count(token.form)) { token.upos = "ADP"; continue; }
            if (conjunctions.count(token.form)) { token.upos = "CCONJ"; continue; }
            if (subordinators.count(token.form)) { token.upos = "SCONJ"; continue; }
            if (relative_pronouns.count(token.form) || pronouns.count(token.form)) { token.upos = "PRON"; continue; }
            if (adverbs.count(token.form)) {
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
            if (t[static_cast<size_t>(i)].upos != "PRON" || !relative_pronouns.count(forms[static_cast<size_t>(i)])) continue;
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
            if (forms[static_cast<size_t>(i)] == "will" || forms[static_cast<size_t>(i)] == "vai" ||
                forms[static_cast<size_t>(i)] == "irá" || forms[static_cast<size_t>(i)] == "ira")
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

        auto phrase = [&](int seed) {
            std::vector<int> selected;
            std::vector<bool> visited(static_cast<size_t>(n), false);
            std::function<void(int)> collect = [&](int idx) {
                if (idx < 0 || idx >= n || visited[static_cast<size_t>(idx)]) return;
                visited[static_cast<size_t>(idx)] = true;
                const auto& token = t[static_cast<size_t>(idx)];
                if (token.upos == "DET" || token.upos == "ADP" || token.upos == "ADV" ||
                    token.upos == "SCONJ" || token.upos == "CCONJ" || token.upos == "AUX" ||
                    token.upos == "VERB") return;
                selected.push_back(idx);
                for (int j = 0; j < n; ++j) {
                    if (t[static_cast<size_t>(j)].head != idx + 1) continue;
                    if (!phrase_deprels.count(t[static_cast<size_t>(j)].deprel)) continue;
                    collect(j);
                }
            };
            collect(seed);
            std::sort(selected.begin(), selected.end());
            std::string out;
            for (int idx : selected) {
                if (!out.empty()) out += " ";
                out += t[static_cast<size_t>(idx)].form;
            }
            if (out.empty() && seed >= 0 && seed < n) out = t[static_cast<size_t>(seed)].form;
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
                     relative_pronouns.count(t[static_cast<size_t>(subject)].form)) {
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
                    if (toks[k] != "que" && toks[k] != "quem" && toks[k] != "which" && toks[k] != "who")
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
        if (matched || hypothetical) continue;

        // Interrogatives and imperatives do not assert facts.
        static const char* question_starts[] = {
            "por que", "porque", "why", "o que", "what", "qual", "which", "quem", "who",
            "como", "how", "quando", "when", "onde", "where", "quanto", "quantos", "how many"
        };
        bool question = s.find('?') != std::string::npos;
        for (const auto* start : question_starts)
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
            const auto subject = clean_span(parts, 0, verb_at, stopwords_);
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
                const auto object = clean_span(group, 0, group.size(), stopwords_);
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
            if (toks[k] != "que" && toks[k] != "quem" && toks[k] != "which" && toks[k] != "who") continue;
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
    for (const auto& triple : dependencies.triples) {
        auto existing = std::find_if(doc.triples.begin(), doc.triples.end(), [&](const ParsedTriple& value) {
            return value.subject == triple.subject && value.relation == triple.relation && value.object == triple.object;
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
