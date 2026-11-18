#pragma once

#include "core/causal_graph.hpp"
#include "core/language_rules.hpp"
#include "core/realizer.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

struct CoherenceScore {
    double language = 0.0;   // negative log-probability per token (lower is better)
    double centering = 0.0;  // higher is better
    double grounding = 0.0;  // higher is better
    double total = 0.0;      // higher is better
};

/// Count-based coherence model: interpolated absolute-discount n-grams,
/// topic cache, centering heuristics and graph grounding. No neural training.
class CoherenceModel {
public:
    void observe(const std::string& text, const TextParser& parser);
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    void clear();

    double sentence_cost(const std::vector<std::string>& tokens) const;
    CoherenceScore score(const std::string& candidate,
                         const SemanticPlan& plan,
                         const std::string& previous_sentence,
                         const CausalGraph& graph,
                         const LanguageRules& rules) const;
    std::string choose(const std::vector<std::string>& candidates,
                       const SemanticPlan& plan,
                       const std::string& previous_sentence,
                       const CausalGraph& graph,
                       const LanguageRules& rules) const;

    size_t unigrams() const { return unigrams_.size(); }
    size_t bigrams() const { return bigrams_.size(); }
    size_t trigrams() const { return trigrams_.size(); }
    size_t memory_bytes() const;
    bool empty() const { return unigrams_.empty(); }

private:
    std::unordered_map<std::string, uint32_t> unigrams_, bigrams_, trigrams_;
    uint64_t total_ = 0;
};

} // namespace cognicore
