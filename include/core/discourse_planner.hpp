#pragma once

#include "core/causal_graph.hpp"
#include "core/tsetlin.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace cognicore {

enum class RhetoricalRole : uint8_t {
    THESIS = 0,
    CAUSAL_CHAIN = 1,
    CONSEQUENCE = 2,
    ALTERNATIVE = 3,
    CONTRAST = 4
};

struct DiscoursePlan {
    std::vector<CausalPath> paths;
    std::vector<CausalPath> alternatives;
    std::vector<RhetoricalRole> roles;
    std::string rationale;
};

/// Symbolic discourse planner. It selects and orders graph paths before any
/// surface text is produced. No language model is involved: ordering is based on
/// seed connectivity, graph confidence and rhetorical compatibility.
class DiscoursePlanner {
public:
    DiscoursePlan plan(const std::vector<CausalPath>& candidates,
                       DialogIntent intent,
                       bool hypothetical,
                       const std::vector<std::string>& seeds = {}) const;
};

const char* rhetorical_role_name(RhetoricalRole role) noexcept;

} // namespace cognicore
