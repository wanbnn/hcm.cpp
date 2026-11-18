#include "core/natural_prose.hpp"

#include "core/morphological_realizer.hpp"
#include "core/text_parser.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <vector>

namespace cognicore {
namespace {

std::string lower(std::string value) {
    return TextParser::utf8_tolower(value);
}

std::string capitalize(std::string value) {
    if (!value.empty() && static_cast<unsigned char>(value[0]) < 128)
        value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
    return value;
}

std::string lang_choice(const std::vector<std::string>& values, const std::string& lang,
                        size_t index, const std::string& fallback) {
    std::vector<std::string> selected;
    const auto prefix = lang + "|";
    for (const auto& value : values)
        if (value.starts_with(prefix)) selected.push_back(value.substr(prefix.size()));
    if (selected.empty()) {
        for (const auto& value : values) {
            const auto bar = value.find('|');
            selected.push_back(bar == std::string::npos ? value : value.substr(bar + 1));
        }
    }
    if (selected.empty()) return fallback;
    return selected[index % selected.size()];
}

std::string article_np(const std::string& name, GramGender gender, GramNumber number,
                       const std::string& lang) {
    const auto lowered = lower(name);
    for (const auto* article : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
        if (lowered.starts_with(article)) return name;
    return MorphologicalRealizer::definite_article(gender, number, lang) + " " + name;
}

std::string prep_article(const std::string& preposition, GramGender gender,
                         GramNumber number, const std::string& lang) {
    if (lang != "pt") return preposition + " " + MorphologicalRealizer::definite_article(gender, number, lang);
    const bool fem = gender == GramGender::FEM;
    const bool plur = number == GramNumber::PLUR;
    if (preposition == "por") return fem ? (plur ? "pelas" : "pela") : (plur ? "pelos" : "pelo");
    if (preposition == "de") return fem ? (plur ? "das" : "da") : (plur ? "dos" : "do");
    return preposition;
}

std::string prep_np(const std::string& preposition, const std::string& name,
                    GramGender gender, GramNumber number, const std::string& lang) {
    return prep_article(preposition, gender, number, lang) + " " + name;
}

std::string join_objects(const std::vector<std::string>& objects) {
    if (objects.empty()) return {};
    if (objects.size() == 1) return objects.front();
    std::string out;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (i) out += (i + 1 == objects.size()) ? " e " : ", ";
        out += objects[i];
    }
    return out;
}

std::string pronoun(const CausalStep& step, const std::string& lang) {
    if (lang == "en") return step.src_number == GramNumber::PLUR ? "they" : "it";
    if (step.src_gender == GramGender::FEM)
        return step.src_number == GramNumber::PLUR ? "elas" : "ela";
    return step.src_number == GramNumber::PLUR ? "eles" : "ele";
}

} // namespace

std::string NaturalProseRealizer::realize(const DocumentParagraph& paragraph,
                                          const LanguageRules& rules) const {
    std::vector<CausalStep> steps;
    std::set<std::string> seen;
    for (const auto& sentence : paragraph.sentences)
        for (const auto& path : sentence.plan.paths)
            for (const auto& step : path.steps) {
                const auto key = lower(step.src) + "|" + lower(step.rel) + "|" + lower(step.dst);
                if (seen.insert(key).second) steps.push_back(step);
            }
    if (steps.empty()) return {};

    const std::string lang = rules.language.empty() ? "pt" : rules.language;
    MorphologicalRealizer morphology(rules);
    std::ostringstream out;
    bool first_sentence = true;
    std::string current_subject;

    auto append_sentence = [&](const std::string& sentence) {
        if (sentence.empty()) return;
        if (!first_sentence) out << ' ';
        auto text = MorphologicalRealizer::contract_prepositions(sentence);
        out << capitalize(text);
        if (text.back() != '.' && text.back() != '!' && text.back() != '?') out << '.';
        first_sentence = false;
    };

    // Definition/property first: it is the natural topic sentence.
    for (const auto& step : steps) {
        if (step.type != RelationType::TAXONOMIC && step.type != RelationType::PROPERTY) continue;
        const auto subject = article_np(step.surface_src.empty() ? step.src : step.surface_src,
                                        step.src_gender, step.src_number, lang);
        append_sentence(morphology.realize_clause(step, subject, lang));
        current_subject = step.src;
    }

    // Parts: coordinate all part-of objects under one subject.
    std::vector<CausalStep> parts;
    for (const auto& step : steps)
        if (step.type == RelationType::HAS_PART || step.type == RelationType::PART_OF)
            parts.push_back(step);
    if (!parts.empty()) {
        std::vector<std::string> objects;
        for (const auto& step : parts) {
            if (step.type != RelationType::HAS_PART) continue;
            objects.push_back(prep_np("por", step.surface_dst.empty() ? step.dst : step.surface_dst,
                                      step.dst_gender, step.dst_number, lang));
        }
        if (!objects.empty()) {
            const auto& first = parts.front();
            const auto subject = article_np(first.surface_src.empty() ? first.src : first.surface_src,
                                            first.src_gender, first.src_number, lang);
            auto text = lang_choice(rules.part_of_templates, lang, 0,
                                    "{SUBJ} é formado {OBJS}.");
            const auto subj_at = text.find("{SUBJ}");
            if (subj_at != std::string::npos) text.replace(subj_at, 6, subject);
            const auto objs_at = text.find("{OBJS}");
            if (objs_at != std::string::npos) text.replace(objs_at, 6, join_objects(objects));
            append_sentence(text);
            current_subject = first.src;
        }
        for (const auto& step : parts) {
            if (step.type != RelationType::PART_OF) continue;
            const auto subject = article_np(step.surface_src.empty() ? step.src : step.surface_src,
                                            step.src_gender, step.src_number, lang);
            const auto object = prep_np("de", step.surface_dst.empty() ? step.dst : step.surface_dst,
                                        step.dst_gender, step.dst_number, lang);
            append_sentence(subject + " faz parte " + object);
            current_subject = step.src;
        }
    }

    // Functions/actions/causes: pronominalize the repeated topic.
    for (const auto& step : steps) {
        if (step.type == RelationType::TAXONOMIC || step.type == RelationType::PROPERTY ||
            step.type == RelationType::HAS_PART || step.type == RelationType::PART_OF ||
            step.type == RelationType::SEQUENCE || step.type == RelationType::PRECEDES) continue;
        std::string subject;
        if (first_sentence || step.src != current_subject)
            subject = article_np(step.surface_src.empty() ? step.src : step.surface_src,
                                 step.src_gender, step.src_number, lang);
        else
            subject = pronoun(step, lang);
        append_sentence(morphology.realize_clause(step, subject, lang));
        current_subject = step.src;
    }

    // Sequence: explicit ordinals, one clause per ordered fact.
    std::vector<CausalStep> sequence;
    for (const auto& step : steps)
        if (step.type == RelationType::SEQUENCE || step.type == RelationType::PRECEDES)
            sequence.push_back(step);
    if (!sequence.empty()) {
        std::string text;
        for (size_t i = 0; i < sequence.size(); ++i) {
            const auto& step = sequence[i];
            const auto subject = article_np(step.surface_src.empty() ? step.src : step.surface_src,
                                            step.src_gender, step.src_number, lang);
            auto clause = morphology.realize_clause(step, subject, lang);
            if (sequence.size() == 1) { text = clause; break; }
            if (static_cast<unsigned char>(clause[0]) < 128)
                clause[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(clause[0])));
            auto ordinal = (i + 1 == sequence.size())
                ? lang_choice(rules.sequence_ordinals, lang, 3, "Por fim")
                : lang_choice(rules.sequence_ordinals, lang, i,
                              i == 0 ? "Primeiro" : "Em seguida");
            if (i == 0) text = capitalize(ordinal) + ", " + clause;
            else text += "; " + lower(ordinal) + ", " + clause;
        }
        if (!text.empty()) append_sentence(text);
    }

    return out.str();
}

} // namespace cognicore
