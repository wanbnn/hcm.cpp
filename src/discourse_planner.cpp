#include "core/discourse_planner.hpp"

#include <algorithm>
#include <set>
#include <sstream>

namespace cognicore {
namespace {

std::string signature(const CausalPath& path) {
    std::ostringstream out;
    for (const auto& step : path.steps)
        out << step.src << '\x1f' << step.rel << '\x1f' << step.dst << '\x1f'
            << unsigned(step.type) << '\x1f' << step.negated << '\x1f' << unsigned(step.tense) << '\x1f'
            << step.modality << '\x1e';
    return out.str();
}

bool is_prefix(const CausalPath& prefix, const CausalPath& whole) {
    if (prefix.steps.size() >= whole.steps.size()) return false;
    for (size_t i = 0; i < prefix.steps.size(); ++i) {
        const auto& a = prefix.steps[i];
        const auto& b = whole.steps[i];
        if (a.src != b.src || a.rel != b.rel || a.dst != b.dst || a.type != b.type ||
            a.negated != b.negated) return false;
    }
    return true;
}

bool shares_node(const CausalPath& a, const CausalPath& b) {
    for (const auto& x : a.steps) for (const auto& y : b.steps)
        if (x.src == y.src || x.src == y.dst || x.dst == y.src || x.dst == y.dst) return true;
    return false;
}

bool ends_at(const CausalPath& path, const std::string& node) {
    return !path.steps.empty() && path.steps.back().dst == node;
}

bool ends_in_seed(const CausalPath& path, const std::vector<std::string>& seeds) {
    if (path.steps.empty()) return false;
    return std::find(seeds.begin(), seeds.end(), path.steps.back().dst) != seeds.end();
}

} // namespace

const char* rhetorical_role_name(RhetoricalRole role) noexcept {
    switch (role) {
        case RhetoricalRole::THESIS: return "THESIS";
        case RhetoricalRole::CAUSAL_CHAIN: return "CAUSAL_CHAIN";
        case RhetoricalRole::CONSEQUENCE: return "CONSEQUENCE";
        case RhetoricalRole::ALTERNATIVE: return "ALTERNATIVE";
        case RhetoricalRole::CONTRAST: return "CONTRAST";
    }
    return "UNKNOWN";
}

DiscoursePlan DiscoursePlanner::plan(const std::vector<CausalPath>& candidates,
                                     DialogIntent intent, bool hypothetical,
                                     const std::vector<std::string>& seeds) const {
    DiscoursePlan result;
    std::vector<CausalPath> valid;
    for (const auto& path : candidates) {
        if (path.steps.empty()) continue;
        valid.push_back(path);
    }
    if (valid.empty()) {
        result.rationale = "nenhum caminho no grafo";
        return result;
    }

    // Remove exact duplicates, then drop strict prefixes when a longer path
    // starts from the same premise. A one-hop fact is not discarded if it is the
    // only connection; it is only hidden when a longer derivation subsumes it.
    std::sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    std::vector<CausalPath> unique;
    std::set<std::string> seen;
    for (auto& path : valid) {
        auto key = signature(path);
        if (!seen.insert(key).second) continue;
        unique.push_back(std::move(path));
    }
    std::vector<bool> covered(unique.size(), false);
    for (size_t i = 0; i < unique.size(); ++i) {
        for (size_t j = 0; j < unique.size(); ++j) {
            if (i == j || unique[i].steps.empty() || unique[j].steps.empty()) continue;
            // Keep a shorter route when its endpoint is a query seed: the longer
            // route is still useful, but it must not replace the direct answer.
            // Explicit relation queries receive a high score from the inference
            // pipeline; they are intentional answers and must not be pruned by
            // a longer generic path.
            if (unique[i].score < 10.0f &&
                unique[i].steps.front().src == unique[j].steps.front().src &&
                is_prefix(unique[i], unique[j]) && !ends_in_seed(unique[i], seeds)) {
                covered[i] = true; break;
            }
        }
    }
    std::vector<CausalPath> pruned;
    for (size_t i = 0; i < unique.size(); ++i)
        if (!covered[i]) pruned.push_back(std::move(unique[i]));
    if (pruned.empty()) pruned = std::move(unique);

    // The main thesis is the highest-confidence path that ends at a query seed.
    // Multi-hop derivations receive the graph's existing inference bonus.
    size_t main_index = 0;
    for (size_t i = 0; i < pruned.size(); ++i) {
        if (!ends_in_seed(pruned[i], seeds)) continue;
        main_index = i; break;
    }
    result.paths.push_back(pruned[main_index]);
    result.roles.push_back(intent == DialogIntent::EXPLAIN_CAUSE ||
                           intent == DialogIntent::HYPOTHESIZE
                           ? RhetoricalRole::CAUSAL_CHAIN : RhetoricalRole::THESIS);

    const auto& main = pruned[main_index];
    const std::string endpoint = main.steps.empty() ? "" : main.steps.back().dst;
    // One alternative path: a distinct route sharing a node with the thesis.
    // It may support the conclusion (same endpoint and different premise) or
    // present an opposite/inhibitory outcome.
    for (size_t i = 0; i < pruned.size(); ++i) {
        if (i == main_index) continue;
        const auto& candidate = pruned[i];
        if (signature(candidate) == signature(main)) continue;
        if (!shares_node(main, candidate)) continue;
        bool supports = ends_at(candidate, endpoint) || ends_at(main, candidate.steps.front().src);
        bool contrast = candidate.steps.front().negated ||
                        candidate.steps.front().type == RelationType::INHIBITS;
        if (!supports && !contrast) continue;
        result.alternatives.push_back(candidate);
        result.roles.push_back(contrast ? RhetoricalRole::CONTRAST : RhetoricalRole::ALTERNATIVE);
        break;
    }

    std::ostringstream rationale;
    rationale << "tese=" << main.steps.size() << " hop(s), score=" << main.score;
    if (!result.alternatives.empty())
        rationale << "; alternativa=" << result.alternatives.front().steps.size()
                  << " hop(s), score=" << result.alternatives.front().score;
    if (hypothetical) rationale << "; modo hipotético (consequência não afirmada)";
    result.rationale = rationale.str();
    return result;
}

} // namespace cognicore
