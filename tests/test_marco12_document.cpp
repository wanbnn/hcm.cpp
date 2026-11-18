#include "core/document_planner.hpp"
#include "core/morphological_realizer.hpp"
#include "core/text_parser.hpp"

#include <cassert>
#include <iostream>
#include <set>
#include <sstream>

using namespace cognicore;

int main() {
    CausalGraph graph;
    graph.add_fact("atrito", "gerar", "calor");
    graph.add_fact("calor", "derreter", "gelo");
    graph.add_fact("gelo", "ser", "sólido", RelationType::TAXONOMIC);
    graph.add_fact("gelo", "impedir", "movimento", RelationType::INHIBITS);
    graph.add_fact("calor", "causar", "expansão");
    Codebook codebook;
    LanguageRules rules;
    assert(rules.load_default());
    DocumentPlanner planner;
    const auto plan = planner.plan("calor", graph, codebook, rules, 5, 2);
    assert(plan.sections.size() >= 3);
    bool has_causes = false, has_consequences = false;
    for (const auto& section : plan.sections) {
        if (section.role == SectionRole::CAUSES) has_causes = true;
        if (section.role == SectionRole::CONSEQUENCES) has_consequences = true;
        assert(!section.heading.empty());
        for (const auto& paragraph : section.paragraphs) assert(!paragraph.sentences.empty());
    }
    assert(has_causes && has_consequences);

    MorphologicalRealizer realizer;
    const auto text = planner.realize(plan, realizer, rules);
    assert(text.find("# ") != std::string::npos);
    assert(text.find("## ") != std::string::npos);
    assert(text.find("calor") != std::string::npos);
    assert(text.find("gelo") != std::string::npos);
    assert(text.size() > 120);

    const auto revised = DocumentPlanner::revise("O calor gera calor. O calor gera calor. O gelo derrete.", rules);
    const auto first = revised.find("O calor gera calor.");
    assert(first != std::string::npos);
    assert(revised.find("O calor gera calor.", first + 1) == std::string::npos);

    std::cout << text << "\n[PASS] marco12 document planner\n";
    return 0;
}
