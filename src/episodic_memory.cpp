#include "core/episodic_memory.hpp"
#include "core/text_parser.hpp"
#include <regex>
#include <limits>
namespace cognicore {
namespace {
SDMConfig config() { SDMConfig c; c.num_locations = 256; c.top_k = 8; return c; }
HyperVector address(const EpisodicFact& f) {
    return HyperVector::from_token("EPISODE:" + std::to_string(f.id)).bind(
        HyperVector::from_token("ENTITY:" + f.entity));
}
HyperVector payload(const EpisodicFact& f) {
    return HyperVector::from_token("PROPERTY:" + f.property).bind(HyperVector::from_token(f.value));
}
}
EpisodicMemory::EpisodicMemory() : memory_(config()) {}
size_t EpisodicMemory::memory_bytes() const {
    size_t n = memory_.memory_bytes() + facts_.capacity() * sizeof(EpisodicFact);
    for (const auto& f : facts_) n += f.entity.capacity() + f.property.capacity() + f.value.capacity() + f.source.capacity();
    return n;
}
void EpisodicMemory::clear() { facts_.clear(); memory_ = KanervaSDM(config()); }
void EpisodicMemory::remember(std::string entity, std::string property, std::string value, std::string source) {
    if (entity.starts_with("meu ")) entity.erase(0, 4);
    if (entity.starts_with("minha ")) entity.erase(0, 6);
    if (entity.ends_with(" só")) entity.resize(entity.size() - std::string(" só").size());
    if (entity.empty() || property.empty() || value.empty()) return;
    if (!facts_.empty()) {
        const auto& last = facts_.back();
        if (last.entity == entity && last.property == property && last.value == value && last.source == source) return;
    }
    EpisodicFact f{facts_.size() + 1, std::move(entity), std::move(property), std::move(value), std::move(source)};
    memory_.write(address(f), payload(f));
    facts_.push_back(std::move(f));
}
std::optional<EpisodicRecall> EpisodicMemory::recall(const std::string& entity, const std::string& property) const {
    for (auto it = facts_.rbegin(); it != facts_.rend(); ++it) {
        if (it->entity != entity || it->property != property) continue;
        const auto a = address(*it);
        const auto value = memory_.read(a).unbind(HyperVector::from_token("PROPERTY:" + property));
        const double similarity = value.similarity(HyperVector::from_token(it->value));
        // A full top-K read alone is never evidence. Reject corrupted recalls.
        if (similarity < 0.15) return std::nullopt;
        return EpisodicRecall{*it, similarity, memory_.activate(a)};
    }
    return std::nullopt;
}
std::optional<MemoryQuantity> parse_memory_quantity(const std::string& text) {
    static const std::regex pattern(R"((\d+(?:[.,]\d+)?)\s*(gib|mib|kib|gb|mb|kb|bytes?)\b)", std::regex::icase);
    std::smatch m;
    if (!std::regex_search(text, m, pattern)) return std::nullopt;
    if (m.position() > 0 && text[static_cast<size_t>(m.position()) - 1] == '-') return std::nullopt;
    const auto remainder = m.suffix().str();
    if (std::regex_search(remainder, pattern)) return std::nullopt; // ambiguous demand/capacity
    auto number = m[1].str(); for (auto& ch : number) if (ch == ',') ch = '.';
    const auto unit = TextParser::utf8_tolower(m[2].str());
    // GB/MB are decimal; GiB/MiB are binary. Never silently equate them.
    const uint64_t base = unit.find('i') != std::string::npos ? 1024 : 1000;
    const unsigned power = unit[0] == 'g' ? 3 : unit[0] == 'm' ? 2 : unit[0] == 'k' ? 1 : 0;
    uint64_t factor = 1; for (unsigned i = 0; i < power; ++i) factor *= base;
    long double bytes;
    try { bytes = std::stold(number) * factor; }
    catch (const std::exception&) { return std::nullopt; }
    if (bytes > std::numeric_limits<uint64_t>::max()) return std::nullopt;
    return MemoryQuantity{static_cast<uint64_t>(bytes), m[0].str()};
}
} // namespace cognicore
