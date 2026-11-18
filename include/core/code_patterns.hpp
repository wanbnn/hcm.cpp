#pragma once

#include <string>
#include <vector>

namespace cognicore {

struct CodeConstruction {
    std::string id;
    std::string name;
    std::string language; // cpp, py, js, ts
    std::vector<std::string> tags;
    std::vector<std::string> required_tags;
    std::string template_code; // {name}, {expression}, {param} slots
};

/// Code constructions are model data, not hardcoded strings in the inference
/// path. The synthesizer selects a construction by task/tag evidence and binds
/// typed slots, then validates the resulting AST with Tree-sitter.
class CodePatternLibrary {
public:
    bool load(const std::string& path);
    bool save(const std::string& path) const;
    bool load_default();
    bool empty() const { return patterns_.empty(); }
    size_t size() const { return patterns_.size(); }
    const std::vector<CodeConstruction>& patterns() const { return patterns_; }
    /// Highest tag-overlap construction for the requested language.
    const CodeConstruction* match(const std::string& lower_task, const std::string& language) const;

private:
    std::vector<CodeConstruction> patterns_;
};

} // namespace cognicore
