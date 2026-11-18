#pragma once

#include "core/causal_graph.hpp"
#include "core/text_parser.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cognicore {

/// A discourse entity tracked across turns. Salience is symbolic (mentions,
/// recency and grammatical agreement), never a neural embedding.
struct ContextEntity {
    std::string name;
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::SING;
    uint32_t mentions = 0;
    uint32_t last_turn = 0;
    int32_t salience = 0;
};

struct ReferenceResolution {
    std::string surface;
    std::string referent;
    std::string reason;
    double score = 0.0;
    bool ambiguous = false;
    std::vector<std::string> candidates;
};

/// Short-term contextual memory: discourse entities, pronominal resolution and
/// topic carry-over. It is independent from the long-term episodic SDM; the two
/// are combined by the inference pipeline.
class ContextMemory {
public:
    void observe_entity(const std::string& name, GramGender gender, GramNumber number);
    void observe_user_turn(const std::string& text, const TextParser& parser, const CausalGraph& graph);
    void observe_concepts(const std::vector<std::string>& concepts);

    /// Resolves explicit mentions, pronouns and deictics ("lá", "nesse servidor").
    std::optional<ReferenceResolution> resolve(const std::string& text, const TextParser& parser) const;

    /// Returns the previous topic when the current turn is a follow-up that does
    /// not introduce any known concept of its own.
    std::vector<std::string> carryover_concepts(const std::string& text, const TextParser& parser,
                                                const CausalGraph& graph) const;

    const std::vector<ContextEntity>& entities() const { return entities_; }
    const std::vector<std::string>& last_concepts() const { return last_concepts_; }
    uint64_t turn() const { return turn_; }
    const std::string& last_user() const { return last_user_; }

    void restore(std::vector<ContextEntity> entities, std::vector<std::string> last_concepts, uint64_t turn);
    void clear();
    size_t memory_bytes() const;

private:
    ContextEntity* find(const std::string& name);
    const ContextEntity* find(const std::string& name) const;
    std::vector<const ContextEntity*> salient() const;

    std::vector<ContextEntity> entities_;
    std::vector<std::string> last_concepts_;
    std::string last_user_;
    uint64_t turn_ = 0;
};

} // namespace cognicore
