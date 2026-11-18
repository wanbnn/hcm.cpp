#include "core/code_ast.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

void test_syntax_validation() {
    assert(TreeSitterCodeParser::validate_syntax("int add(int a, int b) { return a + b; }", CodeLanguage::CPP));
    assert(!TreeSitterCodeParser::validate_syntax("int add(int a, { return ;;;", CodeLanguage::CPP));

    assert(TreeSitterCodeParser::validate_syntax("def greet(name: str) -> str:\n    return 'hi ' + name\n", CodeLanguage::PYTHON));
    assert(!TreeSitterCodeParser::validate_syntax("def greet(name:\n    return\n", CodeLanguage::PYTHON));

    assert(TreeSitterCodeParser::validate_syntax("function sum(a, b) { return a + b; }", CodeLanguage::JAVASCRIPT));
    assert(!TreeSitterCodeParser::validate_syntax("function sum(a, b { return ;", CodeLanguage::JAVASCRIPT));

    assert(TreeSitterCodeParser::validate_syntax("function add(x: number): number { return x + 1; }", CodeLanguage::TYPESCRIPT));
    assert(!TreeSitterCodeParser::validate_syntax("function add(x: ): number { return;", CodeLanguage::TYPESCRIPT));
    CodeGraph typescript;
    assert(TreeSitterCodeParser::ingest_source("function add(x: number): number { return x + 1; }", "a.ts", CodeLanguage::TYPESCRIPT, typescript) == 1);
    assert(typescript.get_return_type("add") == "number");
    assert(typescript.get_param_types("add") == std::vector<std::string>{"number"});
    std::cout << "[PASS] test_syntax_validation (CPP, Python, JS)\n";
}

void test_ingest_self_repo() {
    CodeGraph graph;
    KanervaSDM sdm;

    const std::string src_dir = std::string(COGNICORE_SOURCE_DIR);
    size_t count = TreeSitterCodeParser::ingest_directory(src_dir, graph, &sdm);
    std::cout << "[INFO] Ingested " << count << " functions from " << src_dir
              << " (symbols=" << graph.num_symbols() << ", edges=" << graph.num_edges() << ")\n";
    assert(count > 10);
    assert(graph.num_symbols() > 10);
    assert(graph.num_edges() > 10);

    // Query callers of "read" (e.g., ModelBundle::infer_and_respond calls sdm.read)
    auto callers = graph.find_callers("read");
    std::cout << "Callers of 'read':";
    for (const auto& c : callers) std::cout << " " << c.name;
    std::cout << "\n";
    assert(!callers.empty());

    // Check KanervaSDM::read signature extracted from src/sdm.cpp
    auto ptypes = graph.get_param_types("KanervaSDM::read");
    std::string rtype = graph.get_return_type("KanervaSDM::read");
    std::cout << "KanervaSDM::read -> return_type='" << rtype << "', params=" << ptypes.size() << "\n";
    assert(rtype == "HyperVector");
    assert(!ptypes.empty());

    // VSA query for HyperVector + read
    HyperVector q = CodeGraph::encode_code_vsa(CodeLanguage::CPP, "read", "HyperVector", {});
    auto vsa_hits = graph.query_vsa(q, 5);
    assert(!vsa_hits.empty());
    bool found = false;
    for (const auto& hit : vsa_hits) if (hit.name == "KanervaSDM::read") found = true;
    assert(found);
    assert(!graph.find_callers("KanervaSDM::read").empty());
    std::cout << "[PASS] test_ingest_self_repo\n";
}

int main() {
    test_syntax_validation();
    test_ingest_self_repo();
    return 0;
}
