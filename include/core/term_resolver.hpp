#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

/// Maps surface variants and head nouns to canonical multi-word terms.
/// Example: "motor" -> "motor de combustão interna". Ambiguous aliases are
/// ignored; exact canonical names always resolve to themselves.
class TermResolver {
public:
    void clear();
    std::string resolve(const std::string& name) const;
    void add_term(const std::string& canonical);
    void add_term(const std::string& canonical, const std::string& surface);
    void add_alias(const std::string& alias, const std::string& canonical);
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    size_t size() const { return aliases_.size(); }
    size_t memory_bytes() const;

private:
    static std::string norm(const std::string& value);
    static std::string head_of(const std::string& canonical);
    std::unordered_map<std::string, std::string> alias_to_canonical_;
    std::unordered_map<std::string, std::vector<std::string>> aliases_;
};

} // namespace cognicore
