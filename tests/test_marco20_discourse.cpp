#include "core/discourse_generation.hpp"
#include "core/document_planner.hpp"
#include "core/language_rules.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

namespace {
CausalStep step(const char* src, const char* rel, const char* dst, RelationType type,
                GramGender sg, GramNumber sn, GramGender og, GramNumber on) {
    CausalStep value;
    value.src = src;
    value.rel = rel;
    value.dst = dst;
    value.type = type;
    value.src_gender = sg;
    value.src_number = sn;
    value.dst_gender = og;
    value.dst_number = on;
    value.confidence = 0.9f;
    return value;
}

void add(DocumentParagraph& paragraph, const CausalStep& value) {
    DocumentSentence sentence;
    CausalPath path;
    path.steps.push_back(value);
    sentence.plan.paths.push_back(path);
    paragraph.sentences.push_back(std::move(sentence));
}
} // namespace

int main() {
    // Entity identity is canonical, not string-case sensitive.
    {
        EntityRegistry registry;
        const auto a = registry.get_or_add("motor de combustão interna", GramGender::MASC, GramNumber::SING);
        const auto b = registry.get_or_add("Motor de Combustão Interna", GramGender::MASC, GramNumber::SING);
        assert(a == b);
        assert(registry.entity(a).canonical == "motor de combustão interna");
        std::cout << "[PASS] EntityRegistry: identidade canônica\n";
    }

    // Centering: two entities with the same features make pronouns unsafe.
    {
        EntityRegistry registry;
        const auto motor = registry.get_or_add("motor", GramGender::MASC, GramNumber::SING);
        const auto bloco = registry.get_or_add("bloco", GramGender::MASC, GramNumber::SING);
        DiscourseState state;
        std::vector<Proposition> propositions(2);
        propositions[0].subject = motor;
        propositions[0].object = bloco;
        propositions[1].subject = bloco;
        propositions[1].object = motor;
        state.observe(propositions, registry);
        assert(state.ambiguous(GramGender::MASC, GramNumber::SING));
        ReferenceGenerator generator;
        const auto reference = generator.decide(motor, GrammaticalRole::SUBJECT, 1, state, registry);
        assert(reference.form != ReferenceForm::PRONOUN);
        std::cout << "[PASS] DiscourseState/ReferenceGenerator: ambiguidade bloqueia pronome\n";
    }

    // Aggregation: part-of facts become one plural coordination with 'por'.
    {
        EntityRegistry registry;
        std::vector<Proposition> parts;
        for (const auto& [name, gender] : {std::pair<const char*, GramGender>{"bloco", GramGender::MASC},
                                           std::pair<const char*, GramGender>{"árvore de manivelas", GramGender::FEM}}) {
            Proposition proposition;
            proposition.step = step("motor de combustão interna", "conter", name,
                                     RelationType::HAS_PART, GramGender::MASC, GramNumber::SING,
                                     gender, GramNumber::SING);
            proposition.subject = registry.get_or_add(proposition.step.src, GramGender::MASC, GramNumber::SING);
            proposition.object = registry.get_or_add(name, gender, GramNumber::SING);
            parts.push_back(proposition);
        }
        Aggregator aggregator;
        const auto plans = aggregator.group(parts, registry);
        assert(plans.size() == 1);
        assert(plans.front().conjuncts.size() == 2);
        assert(plans.front().number == GramNumber::PLUR);
        assert(plans.front().gender == GramGender::MASC);
        assert(plans.front().preposition == "por");
        std::cout << "[PASS] Aggregator: coordenação e agreement\n";
    }

    // End-to-end: cohesive prose with coordination and anaphora.
    {
        DocumentParagraph paragraph;
        add(paragraph, step("motor de combustão interna", "conter", "bloco", RelationType::HAS_PART,
                            GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING));
        add(paragraph, step("motor de combustão interna", "conter", "cabeçote", RelationType::HAS_PART,
                            GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING));
        add(paragraph, step("motor de combustão interna", "produzir", "trabalho", RelationType::CAUSAL,
                            GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING));
        add(paragraph, step("motor de combustão interna", "usar", "ciclo otto", RelationType::ACTION,
                            GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING));
        LanguageRules rules;
        const auto text = DiscourseGenerator{}.realize(paragraph, rules);
        assert(text.find("é formado pelo") != std::string::npos);
        assert(text.find("Ele ") != std::string::npos);
        assert(text.find("bloco") != std::string::npos);
        assert(text.find("cabeçote") != std::string::npos);
        std::cout << "[PASS] DiscourseGenerator: coordenação + anáfora\n";
    }

    // Compound coordination keeps the preposition on each conjunct.
    {
        DocumentParagraph paragraph;
        add(paragraph, step("motor de combustão interna", "conter", "as bielas e os êmbolos",
                            RelationType::HAS_PART, GramGender::MASC, GramNumber::SING,
                            GramGender::FEM, GramNumber::PLUR));
        LanguageRules rules;
        const auto text = DiscourseGenerator{}.realize(paragraph, rules);
        assert(text.find("pelas bielas e pelos êmbolos") != std::string::npos);
        std::cout << "[PASS] coordenação composta: preposição por conjunto\n";
    }

    // Search penalizes repetition.
    {
        GenerationSearch search;
        const auto unique = search.score("O motor contém o bloco e o cabeçote.");
        const auto repeated = search.score("O motor contém o bloco. O motor contém o bloco.");
        assert(unique < repeated);
        std::cout << "[PASS] GenerationSearch: repetição custa mais\n";
    }

    return 0;
}
