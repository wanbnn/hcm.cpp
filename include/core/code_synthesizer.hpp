#pragma once

#include "core/code_ast.hpp"
#include "core/language_rules.hpp"
#include "core/tsetlin.hpp"
#include "core/vsa.hpp"

#include <string>
#include <vector>

namespace cognicore {

enum class CodePatternTag : uint8_t {
    READ_AND_PROCESS = 0,
    TRANSFORM_VECTOR = 1,
    CALL_CHAIN       = 2,
    SAFE_TRY_CATCH   = 3,
    COUNT            = 4
};

struct CodeTestCase {
    std::string input;
    std::string expected;
};

struct CodeSynthesisResult {
    std::string code;
    std::string error;
    std::vector<std::string> generated_tests;
    size_t iterations = 0;
    std::string repair_notes;
    CodePatternTag selected_pattern = CodePatternTag::CALL_CHAIN;
    std::vector<std::string> used_symbols;
    std::vector<uint32_t> symbol_ids;
    bool is_patch = false;
    std::string patch_target;
    std::vector<int32_t> pattern_votes;
    std::vector<size_t> fired_clauses;
    bool syntax_valid = false;
};

/// Structural AST Code Synthesizer driven by Tsetlin Machine + CodeGraph + VSA,
/// with automatic Tree-sitter syntax verification.
class CodeSynthesizer {
public:
    CodeSynthesizer();
    void set_rules(const LanguageRules& rules) { rules_ = rules; }

    /// Trains the internal Tsetlin pattern selector on (task_description, pattern_tag) pairs.
    void train_patterns(const std::vector<std::pair<std::string, CodePatternTag>>& examples,
                        size_t epochs = 20);

    /// Synthesizes a new function combining compatible symbols/APIs from `graph`
    /// in the requested `lang`, pretty-prints it, and validates syntax via tree-sitter.
    CodeSynthesisResult synthesize(const std::string& task_description,
                                   CodeLanguage lang,
                                   const CodeGraph& graph,
                                   const Codebook& codebook) const;

    /// CEGIS-style loop: generates candidate programs, executes generated tests
    /// (Python) and returns the first passing candidate. Python tests run in
    /// `workspace`; C++/JS candidates are syntax-validated only.
    CodeSynthesisResult synthesize_tested(const std::string& task_description,
                                          CodeLanguage lang,
                                          const CodeGraph& graph,
                                          const Codebook& codebook,
                                          const std::vector<CodeTestCase>& tests,
                                          const std::string& workspace = "") const;

private:
    std::vector<uint64_t> encode_task_features(const std::string& task, CodeLanguage lang) const;

    TsetlinMachine pattern_tm_;
    LanguageRules rules_;
};

} // namespace cognicore
