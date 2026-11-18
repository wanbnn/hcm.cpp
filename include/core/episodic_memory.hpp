#pragma once
#include "core/sdm.hpp"
#include <optional>
#include <string>
#include <vector>
namespace cognicore {
struct EpisodicFact {
    uint64_t id = 0;
    std::string entity, property, value, source;
};
struct EpisodicRecall {
    EpisodicFact fact;
    double similarity = 0;
    std::vector<uint32_t> active_slots;
};
// The journal is provenance; role-bound SDM payloads are the associative index.
class EpisodicMemory {
public:
    EpisodicMemory();
    void remember(std::string entity, std::string property, std::string value, std::string source);
    std::optional<EpisodicRecall> recall(const std::string& entity, const std::string& property) const;
    const std::vector<EpisodicFact>& facts() const { return facts_; }
    void clear();
    size_t memory_bytes() const;
private:
    KanervaSDM memory_;
    std::vector<EpisodicFact> facts_;
};
struct MemoryQuantity { uint64_t bytes; std::string surface; };
std::optional<MemoryQuantity> parse_memory_quantity(const std::string& text);
} // namespace cognicore
