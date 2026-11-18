#pragma once

#include "core/causal_graph.hpp"
#include "core/document_planner.hpp"
#include "core/language_rules.hpp"
#include "core/discourse_config.hpp"
#include "core/tsetlin.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

using EntityId = uint32_t;
inline constexpr EntityId INVALID_ENTITY = UINT32_MAX;

enum class GrammaticalRole : uint8_t { SUBJECT = 0, OBJECT = 1, OBLIQUE = 2 };
enum class ReferenceForm : uint8_t { FULL_NP = 0, DEFINITE_NP = 1, PRONOUN = 2, ZERO = 3 };
enum class SentenceKind : uint8_t { DEFINITION = 0, PART_OF = 1, FUNCTION = 2, SEQUENCE = 3, GENERAL = 4 };

struct Entity {
    EntityId id = INVALID_ENTITY;
    std::string canonical;
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::SING;
    std::string type;
    int salience = 0;
};

struct Mention {
    EntityId entity = INVALID_ENTITY;
    std::string surface;
    GrammaticalRole role = GrammaticalRole::SUBJECT;
    size_t position = 0;
};

struct Proposition {
    EntityId subject = INVALID_ENTITY;
    EntityId object = INVALID_ENTITY;
    CausalStep step;
};

struct DiscourseState {
    std::vector<EntityId> forward_centers;
    EntityId backward_center = INVALID_ENTITY;
    std::unordered_map<EntityId, int> salience;
    std::unordered_map<EntityId, Mention> last_mention;
    // Gender/number pairs that currently have more than one candidate entity.
    std::vector<std::pair<GramGender, GramNumber>> ambiguous_features;
    bool ambiguous(GramGender gender, GramNumber number) const;
    void observe(const std::vector<Proposition>& propositions, const class EntityRegistry& registry);
};

struct CoordinationPlan {
    EntityId subject = INVALID_ENTITY;
    std::vector<EntityId> conjuncts;
    std::string preposition; // "por", "de", "" (direct object)
    GramGender gender = GramGender::MASC;
    GramNumber number = GramNumber::PLUR;
    RelationType relation = RelationType::HAS_PART;
};

struct ReferencePlan {
    EntityId entity = INVALID_ENTITY;
    ReferenceForm form = ReferenceForm::FULL_NP;
};

struct SentencePlan {
    SentenceKind kind = SentenceKind::GENERAL;
    std::vector<Proposition> propositions;
    std::vector<ReferencePlan> references;
    std::vector<CoordinationPlan> coordinations;
    EntityId topic = INVALID_ENTITY;
    EntityId focus = INVALID_ENTITY;
    bool given_topic = false;
};

struct ParagraphPlan {
    std::string heading;
    std::vector<SentencePlan> sentences;
    DiscourseState state;
};

struct TextPlan {
    std::vector<Entity> entities;
    std::vector<ParagraphPlan> paragraphs;
};

class EntityRegistry {
public:
    EntityId get_or_add(const std::string& canonical, GramGender gender, GramNumber number,
                        const std::string& type = {});
    EntityId find(const std::string& canonical) const;
    const Entity& entity(EntityId id) const;
    const std::vector<Entity>& entities() const { return entities_; }
private:
    std::unordered_map<std::string, EntityId> index_;
    std::vector<Entity> entities_;
};

class ReferenceGenerator {
public:
    ReferencePlan decide(EntityId entity, GrammaticalRole role, size_t position,
                         const DiscourseState& state, const EntityRegistry& registry,
                         size_t max_distance = 2) const;
};

class Aggregator {
public:
    std::vector<CoordinationPlan> group(const std::vector<Proposition>& propositions,
                                        const EntityRegistry& registry) const;
};

class TextPlanner {
public:
    TextPlan plan(const DocumentParagraph& paragraph, const LanguageRules& rules,
                  const DiscourseConfig* config = nullptr) const;
};

class CohesionRanker {
public:
    CohesionRanker();
    double score(const std::string& text) const;
private:
    static uint64_t features(const std::string& text);
    SparseTsetlinModel model_;
};

class GenerationSearch {
public:
    struct Candidate {
        std::string text;
        double cost = 0.0;
    };
    std::string select(std::vector<Candidate> candidates) const;
    double score(const std::string& text, const DiscourseConfig* config = nullptr) const;
};

class DiscourseGenerator {
public:
    std::string realize(const DocumentParagraph& paragraph, const LanguageRules& rules,
                        const DiscourseConfig* config = nullptr) const;
};

} // namespace cognicore
