#include "core/causal_graph.hpp"
#include "core/content_planner.hpp"
#include "core/document_planner.hpp"
#include "core/morphological_realizer.hpp"
#include "core/term_resolver.hpp"
#include "core/text_parser.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

int main() {
    // Multi-word noun phrases and head-noun gender.
    {
        TextParser parser;
        auto doc = parser.parse_dependencies("O motor de combustão interna produz trabalho.");
        assert(!doc.triples.empty());
        assert(doc.triples.front().subject == "motor de combustão interna");
        assert(doc.triples.front().object == "trabalho");
        assert(doc.triples.front().subj_gender == GramGender::MASC);

        auto vela = parser.parse_dependencies("A vela de ignição inflama a mistura comprimida.");
        assert(!vela.triples.empty());
        assert(vela.triples.front().subject == "vela de ignição");
        assert(vela.triples.front().subj_gender == GramGender::FEM);
        std::cout << "[PASS] parser: NPs multi-palavra e gênero pelo núcleo\n";
    }

    // Alias resolution.
    {
        TermResolver terms;
        terms.add_term("motor de combustão interna");
        assert(terms.resolve("motor") == "motor de combustão interna");
        assert(terms.resolve("Motor de Combustão Interna") == "motor de combustão interna");
        std::cout << "[PASS] TermResolver: alias de núcleo\n";
    }

    // Content-first planning: mechanism with parts and functions.
    {
        CausalGraph graph;
        graph.add_fact("motor de combustão interna", "conter", "bloco",
                       RelationType::HAS_PART, 0.9f, false, false, GramTense::PRES, "",
                       GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING);
        graph.add_fact("motor de combustão interna", "conter", "cabeçote",
                       RelationType::HAS_PART, 0.9f, false, false, GramTense::PRES, "",
                       GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING);
        graph.add_fact("motor de combustão interna", "produzir", "trabalho",
                       RelationType::CAUSAL, 0.9f, false, false, GramTense::PRES, "",
                       GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING);
        graph.add_fact("motor de combustão interna", "transformar", "movimento",
                       RelationType::ACTION, 0.9f, false, false, GramTense::PRES, "",
                       GramGender::MASC, GramNumber::SING, GramGender::MASC, GramNumber::SING);
        graph.add_fact("admissão", "precede", "compressão",
                       RelationType::SEQUENCE, 0.9f, false, false, GramTense::PRES, "",
                       GramGender::FEM, GramNumber::SING, GramGender::FEM, GramNumber::SING);

        Codebook codebook;
        LanguageRules rules;
        ContentPlanner planner;
        auto plan = planner.plan("Como funciona o motor de combustão interna?", graph,
                                 codebook, rules);
        assert(plan.goal == CommunicativeGoal::MECHANISM);
        assert(plan.answer.parts.size() == 2);
        assert(plan.answer.functions.size() >= 2);
        assert(plan.document.sections.size() >= 2);
        assert(plan.shape == ResponseShape::DOCUMENT);

        MorphologicalRealizer mouth(rules);
        DocumentPlanner renderer;
        RenderOptions options;
        options.title = true;
        options.headings = true;
        const auto text = renderer.realize(plan.document, mouth, rules, options);
        assert(text.find("## Partes") != std::string::npos);
        assert(text.find("bloco") != std::string::npos);
        assert(text.find("cabeçote") != std::string::npos);

        RenderOptions natural_options;
        natural_options.title = true;
        natural_options.headings = true;
        natural_options.natural = true;
        const auto natural = renderer.realize(plan.document, mouth, rules, natural_options);
        assert(natural.find("é formado pelo") != std::string::npos);
        assert(natural.find("Ele ") != std::string::npos);
        std::cout << "[PASS] ContentPlanner: MECHANISM -> DOCUMENT (prosa natural)\n";
    }

    // A single definition should stay brief.
    {
        CausalGraph graph;
        graph.add_fact("gelo", "ser", "sólido", RelationType::TAXONOMIC, 0.9f, false, false,
                       GramTense::PRES, "", GramGender::MASC, GramNumber::SING,
                       GramGender::MASC, GramNumber::SING);
        Codebook codebook;
        LanguageRules rules;
        ContentPlanner planner;
        auto plan = planner.plan("O que é o gelo?", graph, codebook, rules);
        assert(plan.goal == CommunicativeGoal::DEFINITION);
        assert(plan.shape == ResponseShape::BRIEF);
        std::cout << "[PASS] ContentPlanner: definição curta -> BRIEF\n";
    }

    return 0;
}
