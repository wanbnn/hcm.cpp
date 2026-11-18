#include "core/code_synthesizer.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace cognicore;

int main() {
    const auto workspace = std::filesystem::temp_directory_path() /
        ("cognicore_code_tests_" + std::to_string(getpid()));
    std::filesystem::create_directories(workspace);
    std::ofstream(workspace / "mathlib.py")
        << "def identity(value: str) -> str:\n    return value\n\n"
           "def count(value: str) -> int:\n    return len(value)\n";

    CodeGraph graph;
    Codebook codebook;
    TreeSitterCodeParser::ingest_source(
        "def identity(value: str) -> str:\n    return value\n\n"
        "def count(value: str) -> int:\n    return len(value)\n",
        (std::filesystem::path("mathlib.py")).string(), CodeLanguage::PYTHON, graph);
    CodeSynthesizer synthesizer;
    const auto pipeline = synthesizer.synthesize_tested(
        "identity e count", CodeLanguage::PYTHON, graph, codebook,
        {{"\"hello\"", "5"}}, workspace.string());
    assert(pipeline.syntax_valid && pipeline.iterations >= 1);
    assert(pipeline.code.find("identity") != std::string::npos);
    assert(pipeline.generated_tests.size() == 1);

    const auto loop = synthesizer.synthesize_tested(
        "somar uma lista de valores", CodeLanguage::PYTHON, CodeGraph{}, codebook,
        {{"[1, 2, 3]", "6"}}, workspace.string());
    assert(loop.syntax_valid);
    assert(loop.code.find("for value in values") != std::string::npos);
    assert(loop.iterations >= 1);

    const auto impossible = synthesizer.synthesize_tested(
        "somar uma lista de valores", CodeLanguage::PYTHON, CodeGraph{}, codebook,
        {{"[1, 2, 3]", "7"}}, workspace.string());
    assert(!impossible.syntax_valid);
    assert(!impossible.error.empty());
    assert(!impossible.repair_notes.empty());

    std::filesystem::remove_all(workspace);
    std::cout << "[PASS] marco17 tested code synthesis\n";
    return 0;
}
