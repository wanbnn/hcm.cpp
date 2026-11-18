#include "core/code_synthesizer.hpp"

#include <cassert>
#include <iostream>

using namespace cognicore;

void test_cpp_and_python_synthesis() {
    CodeGraph graph;
    KanervaSDM sdm;
    Codebook codebook;

    const std::string mini_cpp_lib =
        "#include <fstream>\n"
        "#include <string>\n"
        "#include <vector>\n"
        "std::vector<int> read_numbers(const std::string& path) {\n"
        "    return {1, 2, 3};\n"
        "}\n"
        "int sum_vector(const std::vector<int>& vals) {\n"
        "    int total = 0;\n"
        "    for (int v : vals) total += v;\n"
        "    return total;\n"
        "}\n";

    const std::string mini_py_lib =
        "import os\n"
        "def load_items(path: str):\n"
        "    return [1, 2, 3]\n"
        "def compute_total(items):\n"
        "    return sum(items)\n";

    assert(TreeSitterCodeParser::ingest_source(mini_cpp_lib, "mini_lib.cpp", CodeLanguage::CPP, graph, &sdm) == 2);
    assert(TreeSitterCodeParser::ingest_source(mini_py_lib, "mini_lib.py", CodeLanguage::PYTHON, graph, &sdm) == 2);

    CodeSynthesizer synth;

    auto cpp_res = synth.synthesize("read_numbers e sum_vector de arquivo", CodeLanguage::CPP, graph, codebook);
    std::cout << "=== Synthesized C++ ===\n" << cpp_res.code << "\n";
    assert(cpp_res.syntax_valid);
    assert(cpp_res.code.find("read_numbers") != std::string::npos);
    assert(cpp_res.code.find("sum_vector") != std::string::npos);

    auto py_res = synth.synthesize("load_items e compute_total", CodeLanguage::PYTHON, graph, codebook);
    std::cout << "=== Synthesized Python ===\n" << py_res.code << "\n";
    assert(py_res.syntax_valid);
    assert(py_res.code.find("load_items") != std::string::npos);
    assert(py_res.code.find("compute_total") != std::string::npos);

    CodeGraph incompatible;
    TreeSitterCodeParser::ingest_source(
        "int first(const std::string& input) { return 1; }\nfloat second(const std::vector<int>& input) { return 1; }",
        "bad.cpp", CodeLanguage::CPP, incompatible);
    auto rejected = synth.synthesize("first e second", CodeLanguage::CPP, incompatible, codebook);
    assert(!rejected.syntax_valid && rejected.code.empty() && !rejected.error.empty());
    CodeGraph chain;
    TreeSitterCodeParser::ingest_source(
        "int first(int x) {return x+1;} int second(int x) {return x*2;} int third(int x) {return x-1;}",
        "chain.cpp", CodeLanguage::CPP, chain);
    auto three = synth.synthesize("first e second e third", CodeLanguage::CPP, chain, codebook);
    assert(three.syntax_valid && three.used_symbols.size() == 3);
    assert(three.code.find("third(step2)") != std::string::npos);
    assert(three.symbol_ids.size() == 3 && !three.pattern_votes.empty());
    auto unknown = synth.synthesize("abra o arquivo de cláusulas e valide se está corrompido", CodeLanguage::CPP, chain, codebook);
    assert(!unknown.syntax_valid && unknown.code.empty());
    auto unrelated = synth.synthesize("calcule uma matriz", CodeLanguage::CPP, chain, codebook);
    assert(!unrelated.syntax_valid && unrelated.code.empty());
    std::cout << "[PASS] test_cpp_and_python_synthesis\n";
}

int main() {
    test_cpp_and_python_synthesis();
    return 0;
}
