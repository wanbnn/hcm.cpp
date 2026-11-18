#include "core/term_resolver.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace cognicore {
namespace {
const char* const kPrepositions[] = {" de ", " do ", " da ", " dos ", " das ",
                                     " em ", " no ", " na ", " nos ", " nas ",
                                     " por ", " para ", " com ", " sem "};
}

std::string TermResolver::norm(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    bool space = false;
    for (unsigned char ch : value) {
        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            space = !out.empty();
            continue;
        }
        if (space) { out.push_back(' '); space = false; }
        out.push_back(static_cast<char>(ch < 128 ? std::tolower(ch) : ch));
    }
    return out;
}

std::string TermResolver::head_of(const std::string& canonical) {
    std::string head = canonical;
    for (const char* prep : kPrepositions) {
        const auto at = head.find(prep);
        if (at != std::string::npos) { head = head.substr(0, at); break; }
    }
    return head;
}

void TermResolver::clear() {
    alias_to_canonical_.clear();
    aliases_.clear();
}

void TermResolver::add_alias(const std::string& alias, const std::string& canonical) {
    const auto a = norm(alias), c = norm(canonical);
    if (a.empty() || c.empty() || a == c) return;
    const auto existing = alias_to_canonical_.find(a);
    if (existing != alias_to_canonical_.end() && existing->second != c) return; // ambiguous
    alias_to_canonical_[a] = c;
    auto& list = aliases_[c];
    if (std::find(list.begin(), list.end(), a) == list.end()) list.push_back(a);
}

void TermResolver::add_term(const std::string& canonical) {
    const auto c = norm(canonical);
    if (c.empty()) return;
    aliases_.try_emplace(c);
    alias_to_canonical_[c] = c;
    const auto head = head_of(c);
    if (head != c) add_alias(head, c);
}

void TermResolver::add_term(const std::string& canonical, const std::string& surface) {
    add_term(canonical);
    add_alias(surface, canonical);
}

std::string TermResolver::resolve(const std::string& name) const {
    const auto key = norm(name);
    const auto it = alias_to_canonical_.find(key);
    return it == alias_to_canonical_.end() ? key : it->second;
}

bool TermResolver::save(const std::string& path) const {
    std::vector<std::string> canonicals;
    canonicals.reserve(aliases_.size());
    for (const auto& [canonical, _] : aliases_) canonicals.push_back(canonical);
    std::sort(canonicals.begin(), canonicals.end());
    std::ofstream out(path);
    if (!out) return false;
    for (const auto& canonical : canonicals) {
        out << canonical;
        const auto it = aliases_.find(canonical);
        if (it != aliases_.end())
            for (const auto& alias : it->second) out << '\t' << alias;
        out << '\n';
    }
    return static_cast<bool>(out);
}

bool TermResolver::load(const std::string& path) {
    clear();
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string canonical;
        if (!std::getline(fields, canonical, '\t') || canonical.empty()) continue;
        add_term(canonical);
        std::string alias;
        while (std::getline(fields, alias, '\t')) add_alias(alias, canonical);
    }
    return true;
}

size_t TermResolver::memory_bytes() const {
    size_t total = 0;
    for (const auto& [canonical, list] : aliases_) {
        total += canonical.capacity() + 32;
        for (const auto& alias : list) total += alias.capacity() + 16;
    }
    return total;
}

} // namespace cognicore
