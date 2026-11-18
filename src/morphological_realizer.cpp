#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace cognicore {

namespace {

std::string capitalize_sentence(std::string s) {
    if (!s.empty()) {
        unsigned char c = static_cast<unsigned char>(s[0]);
        if (c < 0x80) s[0] = static_cast<char>(std::toupper(c));
    }
    return s;
}

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.size(), to);
        pos += to.size();
    }
    return str;
}

std::string to_infinitive_pt(const std::string& verb_or_lemma) {
    // Normalize common 3sg present forms to infinitive if needed
    static const std::unordered_map<std::string, std::string> rev = {
        {"gera", "gerar"}, {"causa", "causar"}, {"provoca", "provocar"},
        {"derrete", "derreter"}, {"impede", "impedir"}, {"apaga", "apagar"},
        {"aquece", "aquecer"}, {"esfria", "esfriar"}, {"produz", "produzir"},
        {"forma", "formar"}, {"evapora", "evaporar"}, {"congela", "congelar"},
        {"é", "ser"}, {"são", "ser"}, {"tem", "ter"}, {"possui", "possuir"},
        {"pode", "poder"}, {"podem", "poder"}, {"deve", "dever"}, {"devem", "dever"},
        {"é um", "ser"}, {"é uma", "ser"}, {"is", "be"}, {"bloqueia", "bloquear"}, {"reduz", "reduzir"}, {"aumenta", "aumentar"}
    };
    auto it = rev.find(verb_or_lemma);
    return (it != rev.end()) ? it->second : verb_or_lemma;
}

} // namespace

std::string MorphologicalRealizer::definite_article(GramGender gender,
                                                    GramNumber number,
                                                    const std::string& lang) {
    if (lang == "en") return "the";
    if (gender == GramGender::FEM) {
        return (number == GramNumber::PLUR) ? "as" : "a";
    }
    return (number == GramNumber::PLUR) ? "os" : "o";
}

std::string MorphologicalRealizer::contract_prepositions(const std::string& text) {
    std::string out = " " + text + " ";
    out = replace_all(out, " de os ", " dos ");
    out = replace_all(out, " de as ", " das ");
    out = replace_all(out, " de o ", " do ");
    out = replace_all(out, " de a ", " da ");
    out = replace_all(out, " em os ", " nos ");
    out = replace_all(out, " em as ", " nas ");
    out = replace_all(out, " em o ", " no ");
    out = replace_all(out, " em a ", " na ");
    out = replace_all(out, " por o ", " pelo ");
    out = replace_all(out, " por a ", " pela ");
    out = replace_all(out, " por os ", " pelos ");
    out = replace_all(out, " por as ", " pelas ");
    return out.substr(1, out.size() - 2);
}

std::string MorphologicalRealizer::conjugate_verb(const std::string& raw_lemma,
                                                  GramTense tense,
                                                  GramNumber number,
                                                  const std::string& lang) {
    std::string lemma = (lang == "pt") ? to_infinitive_pt(raw_lemma) : raw_lemma;
    if (lang == "en") {
        static const std::unordered_map<std::string, std::string> forms = {
            {"is", "be"}, {"are", "be"}, {"was", "be"}, {"were", "be"},
            {"has", "have"}, {"causes", "cause"}, {"produces", "produce"},
            {"generates", "generate"}, {"prevents", "prevent"}, {"blocks", "block"}
        };
        if (auto it = forms.find(lemma); it != forms.end()) lemma = it->second;
    }
    if (tense == GramTense::INF) return lemma;

    if (lang == "en") {
        if (tense == GramTense::SUBJ) return lemma;
        if (lemma == "be") {
            if (tense == GramTense::PAST) return (number == GramNumber::PLUR) ? "were" : "was";
            if (tense == GramTense::FUT) return "will be";
            return (number == GramNumber::PLUR) ? "are" : "is";
        }
        if (lemma == "have") {
            if (tense == GramTense::PAST) return "had";
            if (tense == GramTense::FUT) return "will have";
            return (number == GramNumber::PLUR) ? "have" : "has";
        }
        if (tense == GramTense::FUT) return "will " + lemma;
        if (tense == GramTense::PAST) {
            static const std::unordered_map<std::string, std::string> past = {
                {"go", "went"}, {"make", "made"}, {"read", "read"}, {"write", "wrote"},
                {"run", "ran"}, {"do", "did"}, {"take", "took"}, {"get", "got"}
            };
            if (auto it = past.find(lemma); it != past.end()) return it->second;
            if (lemma.size() > 1 && lemma.back() == 'y' && std::string("aeiou").find(lemma[lemma.size()-2]) == std::string::npos)
                return lemma.substr(0, lemma.size()-1) + "ied";
            if (!lemma.empty() && lemma.back() == 'e') return lemma + "d";
            return lemma + "ed";
        }
        if (number == GramNumber::SING && !lemma.empty()) {
            if (lemma.size() > 1 && lemma.back() == 'y' && std::string("aeiou").find(lemma[lemma.size()-2]) == std::string::npos)
                return lemma.substr(0, lemma.size()-1) + "ies";
            if (lemma.ends_with("ch") || lemma.ends_with("sh") || lemma.back() == 'o') return lemma + "es";
            if (lemma.back() == 's' || lemma.back() == 'x' || lemma.back() == 'z') return lemma + "es";
            return lemma + "s";
        }
        return lemma;
    }

    if (lemma == "poder") {
        if (tense == GramTense::PAST) return number == GramNumber::PLUR ? "puderam" : "pôde";
        if (tense == GramTense::FUT) return number == GramNumber::PLUR ? "poderão" : "poderá";
        if (tense == GramTense::SUBJ) return number == GramNumber::PLUR ? "possam" : "possa";
        return number == GramNumber::PLUR ? "podem" : "pode";
    }
    // Portuguese irregulars
    if (lemma == "ser") {
        if (tense == GramTense::PAST) return (number == GramNumber::PLUR) ? "foram" : "foi";
        if (tense == GramTense::FUT)  return (number == GramNumber::PLUR) ? "serão" : "será";
        if (tense == GramTense::SUBJ) return (number == GramNumber::PLUR) ? "sejam" : "seja";
        return (number == GramNumber::PLUR) ? "são" : "é";
    }
    if (lemma == "ter") {
        if (tense == GramTense::PAST) return (number == GramNumber::PLUR) ? "tiveram" : "teve";
        if (tense == GramTense::FUT)  return (number == GramNumber::PLUR) ? "terão" : "terá";
        if (tense == GramTense::SUBJ) return (number == GramNumber::PLUR) ? "tenham" : "tenha";
        return (number == GramNumber::PLUR) ? "têm" : "tem";
    }
    if (lemma == "impedir" && tense == GramTense::SUBJ) return number == GramNumber::PLUR ? "impeçam" : "impeça";
    if (lemma == "aquecer" && tense == GramTense::SUBJ) return number == GramNumber::PLUR ? "aqueçam" : "aqueça";
    if (lemma == "possuir" && tense == GramTense::PRES) return number == GramNumber::PLUR ? "possuem" : "possui";
    if (lemma == "produzir" || lemma == "reduzir") {
        std::string stem = lemma.substr(0, lemma.size() - 2);
        if (tense == GramTense::PRES && number == GramNumber::SING) return stem.substr(0, stem.size() - 1) + "z";
        if (tense == GramTense::PRES && number == GramNumber::PLUR) return stem + "em";
    }

    // Regular PT-BR conjugation (-ar, -er, -ir)
    if (lemma.size() >= 3) {
        const std::string ending = lemma.substr(lemma.size() - 2);
        const std::string stem = lemma.substr(0, lemma.size() - 2);
        if (ending == "ar") {
            if (tense == GramTense::PAST) return stem + ((number == GramNumber::PLUR) ? "aram" : "ou");
            if (tense == GramTense::FUT)  return lemma + ((number == GramNumber::PLUR) ? "ão" : "á");
            if (tense == GramTense::SUBJ) {
                std::string base = stem;
                if (base.ends_with("g")) base += "u";
                else if (base.ends_with("c")) base = base.substr(0, base.size()-1) + "qu";
                else if (base.ends_with("ç")) base = base.substr(0, base.size()-2) + "c";
                return base + ((number == GramNumber::PLUR) ? "em" : "e");
            }
            return stem + ((number == GramNumber::PLUR) ? "am" : "a");
        }
        if (ending == "er") {
            if (tense == GramTense::PAST) return stem + ((number == GramNumber::PLUR) ? "eram" : "eu");
            if (tense == GramTense::FUT)  return lemma + ((number == GramNumber::PLUR) ? "ão" : "á");
            if (tense == GramTense::SUBJ) return stem + ((number == GramNumber::PLUR) ? "am" : "a");
            return stem + ((number == GramNumber::PLUR) ? "em" : "e");
        }
        if (ending == "ir") {
            if (tense == GramTense::PAST) return stem + ((number == GramNumber::PLUR) ? "iram" : "iu");
            if (tense == GramTense::FUT)  return lemma + ((number == GramNumber::PLUR) ? "ão" : "á");
            if (tense == GramTense::SUBJ) return stem + ((number == GramNumber::PLUR) ? "am" : "a");
            return stem + ((number == GramNumber::PLUR) ? "em" : "e");
        }
    }
    return raw_lemma;
}

std::string MorphologicalRealizer::realize_clause(const CausalStep& step,
                                                  const std::string& subject_surface,
                                                  const std::string& lang,
                                                  const std::string& pattern) const {
    std::string verb_phrase;
    if (lang == "en") {
        if (!step.modality.empty()) {
            verb_phrase = step.modality + (step.negated ? " not " : " ") +
                          conjugate_verb(step.rel, GramTense::INF, step.src_number, "en");
        } else if (step.negated && step.tense == GramTense::FUT) {
            verb_phrase = "will not " + conjugate_verb(step.rel, GramTense::INF, step.src_number, "en");
        } else if (step.negated && conjugate_verb(step.rel, GramTense::INF, step.src_number, "en") == "be") {
            verb_phrase = conjugate_verb("be", step.tense, step.src_number, "en") + " not";
        } else if (step.negated) {
            std::string aux = (step.src_number == GramNumber::PLUR) ? "do not " : "does not ";
            if (step.tense == GramTense::PAST) aux = "did not ";
            verb_phrase = aux + conjugate_verb(step.rel, GramTense::INF, step.src_number, "en");
        } else {
            verb_phrase = conjugate_verb(step.rel, step.tense, step.src_number, "en");
        }
    } else {
        // PT-BR
        std::string neg_prefix = step.negated ? "não " : "";
        if (!step.modality.empty()) {
            std::string mod_conj = conjugate_verb(step.modality, step.tense, step.src_number, "pt");
            std::string inf_v = conjugate_verb(step.rel, GramTense::INF, step.src_number, "pt");
            verb_phrase = neg_prefix + mod_conj + " " + inf_v;
        } else {
            verb_phrase = neg_prefix + conjugate_verb(step.rel, step.tense, step.src_number, "pt");
        }
    }

    auto noun_phrase = [&](const std::string& noun, GramGender gender, GramNumber number) {
        const auto lower = TextParser::utf8_tolower(noun);
        for (const auto* article : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
            if (lower.starts_with(article)) return noun;
        return definite_article(gender, number, lang) + " " + noun;
    };
    std::string obj_phrase = noun_phrase(step.dst, step.dst_gender, step.dst_number);
    if (step.type == RelationType::TAXONOMIC && lang == "pt") {
        obj_phrase = (step.dst_number == GramNumber::PLUR
            ? (step.dst_gender == GramGender::FEM ? "umas " : "uns ")
            : (step.dst_gender == GramGender::FEM ? "uma " : "um ")) + step.dst;
    }

    if (step.type == RelationType::PROPERTY) {
        obj_phrase = step.dst;
        if (lang == "pt") {
            if (step.src_gender == GramGender::FEM && obj_phrase.ends_with("o")) obj_phrase.back() = 'a';
            if (step.src_number == GramNumber::PLUR && !obj_phrase.ends_with("s")) obj_phrase += "s";
        }
    }
    auto clause = replace_all(pattern, "[SUBJ:ENT]", subject_surface);
    clause = replace_all(clause, "[PRED:VERB]", verb_phrase);
    clause = replace_all(clause, "[OBJ:ENT]", obj_phrase);
    return contract_prepositions(clause);
}

std::string MorphologicalRealizer::realize(const SemanticPlan& plan) const {
    if (plan.quoted_statement) return *plan.quoted_statement;
    if (plan.resource) return realize_resource_assessment(plan);
    if (!plan.conditional_conclusions.empty()) return realize_conditional_conclusions(plan);
    const std::string lang = plan.language.empty() ? "pt" : plan.language;

    const auto social = social_constructions(plan);
    if (!social.empty()) return social.front();
    if (plan.intent == DialogIntent::GREET) {
        return (lang == "en")
            ? "Hello! I am ready to reason and learn new concepts with you."
            : "Olá! Estou pronto para raciocinar e aprender novos conceitos com você.";
    }
    if (plan.intent == DialogIntent::AGREE) {
        return (lang == "en")
            ? "Yes, that is consistent with the causal relations stored in memory."
            : "Sim, isso está coerente com as relações causais registradas na memória.";
    }
    if (plan.intent == DialogIntent::ASK_CLARIFY || plan.paths.empty()) {
        if (plan.query_concepts.empty()) {
            return (lang == "en")
                ? "Could you clarify the concepts involved so I can infer the relation?"
                : "Poderia detalhar os conceitos envolvidos para que eu possa inferir a relação?";
        }
        return (lang == "en")
            ? ("I do not have enough connections about " + plan.query_concepts[0] + " yet.")
            : ("Ainda não tenho conexões suficientes sobre " + plan.query_concepts[0] + ".");
    }

    const auto& steps = plan.paths[0].steps;
    if (steps.empty()) return "";

    // Discourse planner + Anaphora substitution across multi-hop sentences
    static const char* pt_connectors[] = {"", "Como resultado, ", "Por conseguinte, ", "Além disso, "};
    static const char* en_connectors[] = {"", "As a result, ", "Consequently, ", "Furthermore, "};

    std::string result;
    std::string prev_subj;
    std::string prev_obj;

    for (size_t i = 0; i < steps.size(); ++i) {
        const auto& st = steps[i];

        // Choose subject surface (full NP vs pronoun/anaphora)
        std::string subj_surface;
        if (i > 0 && st.src == prev_subj) {
            if (lang == "en") {
                subj_surface = (st.src_number == GramNumber::PLUR) ? "they" : "it";
            } else {
                if (st.src_gender == GramGender::NEUT) {
                    subj_surface = "isso";
                } else if (st.src_gender == GramGender::FEM) {
                    subj_surface = (st.src_number == GramNumber::PLUR) ? "elas" : "ela";
                } else {
                    subj_surface = (st.src_number == GramNumber::PLUR) ? "eles" : "ele";
                }
            }
        } else if (i > 0 && st.src == prev_obj) {
            if (lang == "en") {
                subj_surface = (st.src_number == GramNumber::PLUR) ? "they" : "it";
            } else {
                subj_surface = definite_article(st.src_gender, st.src_number, lang) + " " + st.src;
            }
        } else {
            subj_surface = definite_article(st.src_gender, st.src_number, lang) + " " + st.src;
        }

        const auto lower = TextParser::utf8_tolower(st.src);
        for (const auto* article : {"o ", "a ", "os ", "as ", "the "})
            if (lower.starts_with(article) && subj_surface.ends_with(st.src)) { subj_surface = st.src; break; }
        std::string clause = realize_clause(st, subj_surface, lang);

        std::string conn;
        if (i > 0) {
            if (st.type == RelationType::INHIBITS || st.negated) {
                conn = (lang == "en") ? "However, " : "Entretanto, ";
            } else if (st.src == prev_subj) {
                conn = lang == "en" ? "Furthermore, " : "Além disso, ";
            } else if (st.src != prev_obj) {
                conn = lang == "en" ? "Also, " : "Também, ";
            } else {
                conn = (lang == "en") ? en_connectors[std::min<size_t>(i, 3)]
                                      : pt_connectors[std::min<size_t>(i, 3)];
            }
        }

        std::string sentence = conn.empty() ? capitalize_sentence(clause) : (conn + clause);
        if (!result.empty()) result += " ";
        result += sentence + ".";

        prev_subj = st.src;
        prev_obj = st.dst;
    }

    // Planned secondary route (if any). The planner only supplies a distinct,
    // graph-grounded route; the connector marks it as an alternative rather
    // than presenting it as another link in the causal chain.
    for (size_t a = 0; a < plan.alternatives.size(); ++a) {
        SemanticPlan secondary;
        secondary.intent = plan.intent;
        secondary.language = lang;
        secondary.paths.push_back(plan.alternatives[a]);
        auto text = MorphologicalRealizer{}.realize(secondary);
        if (text.empty()) continue;
        if (static_cast<unsigned char>(text[0]) < 128)
            text[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(text[0])));
        const bool contrast = a < plan.rhetorical_roles.size() &&
            plan.rhetorical_roles[a + 1] == static_cast<uint8_t>(4);
        if (!result.empty()) result += " ";
        result += (lang == "en" ? (contrast ? "However, " : "Alternatively, ")
                                : (contrast ? "Porém, " : "Alternativamente, ")) + text;
    }

    return result;
}

} // namespace cognicore
