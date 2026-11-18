#pragma once

#include "core/causal_graph.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

struct Construction {
    std::string id;
    std::string pattern; // {SUBJ}, {OBJS}, {VERB}
    RelationType relation = RelationType::HAS_PART;
    double cost = 0.0;
};

struct DiscourseConfig {
    int max_pronoun_distance = 2;
    bool allow_zero_anaphora = true;
    double repetition_penalty = 0.35;
    double pronoun_bonus = 0.15;
    double length_penalty = 0.01;
    double connector_penalty = 0.05;
    int beam_width = 4;
    std::vector<Construction> constructions;
    std::unordered_map<std::string, std::string> contractions; // "por|o" -> "pelo"
    std::unordered_map<std::string, std::string> entity_types; // canonical -> type

    bool load(const std::string& model_dir);
    std::vector<const Construction*> constructions_for(RelationType relation) const;
    std::string entity_type(const std::string& canonical) const;
    std::string contraction(const std::string& preposition, GramGender gender,
                            GramNumber number) const;
};

} // namespace cognicore
