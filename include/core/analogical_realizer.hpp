#pragma once
#include "core/realizer.hpp"
#include "core/sdm.hpp"
#include <array>
#include <map>
#include <memory>
#include <span>
namespace cognicore {
enum class DiscourseTone : uint8_t { TECHNICAL, DIDACTIC, WARNING, CASUAL };
enum class PhraseShape : uint8_t { CLAUSE, SEQUENCE, CAUSE_SEQUENCE, CONTRAST };
// DOP production: ROOT -> CLAUSE | CLAUSE connective CLAUSE. Leaves are typed
// slots; no source entities or corpus replies are stored in these productions.
// `relation` is either a concrete RelationType (0..4) or 255 for a generic
// clause template. `predicate` further restricts a production to one verb.
struct PhraseFragment {
    uint32_t id = 0, frequency = 1;
    PhraseShape shape = PhraseShape::CLAUSE;
    DiscourseTone tone = DiscourseTone::TECHNICAL;
    uint8_t english = 0, relation = 255;
    char connective[96]{};
    char clause_pattern[160]{};
    char predicate[64]{};
};
struct PPMEvaluation {
    double bits_per_byte = 0.0;
    double perplexity = 1.0;
    size_t bytes = 0;
};
struct PPMRecord {
    std::array<uint8_t, 5> context{};
    uint8_t order = 0, symbol = 0, reserved = 0;
    uint32_t count = 0;
};
class PPMModel {
public:
    void observe(const std::string& text);
    // PPM-C escape/backoff, byte alphabet, orders 0..5. Q16 bits/byte.
    int64_t cost_q16(const std::string& text) const;
    /// Calibrated cross-entropy for reporting only (double log2; never used in
    /// generation). Perplexity = 2^(bits_per_byte).
    PPMEvaluation evaluate(const std::string& text) const;
    std::vector<PPMRecord> records() const;
    void attach(std::span<const PPMRecord> records);
    size_t contexts() const;
    size_t memory_bytes() const;
private:
    void materialize();
    std::map<std::string, std::map<uint8_t, uint32_t>> counts_;
    std::span<const PPMRecord> mapped_;
    size_t entries_ = 0;
};
struct FluencyCandidate {
    std::string text;
    int64_t ppm_cost_q16 = 0, rank_q16 = 0;
    int32_t style_vote = 0;
    std::vector<uint32_t> fragment_ids;
    /// Clause template and connective used for each hop, parallel to
    /// realized_steps. They let structural verification reconstruct the exact
    /// morphological yield even after the archive is mapped from disk.
    std::vector<uint32_t> clause_ids;
    std::vector<std::string> clause_patterns, joiners;
    std::vector<CausalStep> realized_steps;
};
struct FluencyDerivation {
    std::vector<FluencyCandidate> candidates;
    std::vector<uint32_t> retrieved_fragments, active_slots;
    size_t selected = 0;
};
class AnalogicalFluencyRealizer : public ISurfaceRealizer {
public:
    AnalogicalFluencyRealizer();
    void observe(const std::string& source, const TextParser& parser, const ParsedDocument* parsed = nullptr);
    void train(bool use_gpu = false);
    std::string realize(const SemanticPlan& plan) const override;
    FluencyDerivation derive(const SemanticPlan& plan) const;
    static bool preserves_plan(const FluencyCandidate& candidate, const SemanticPlan& plan);
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    bool ready() const { return trained_; }
    bool mapped() const { return bool(mapping_); }
    size_t num_fragments() const;
    size_t memory_bytes() const;
    size_t ppm_contexts() const { return ppm_.contexts(); }
    PPMEvaluation evaluate_text(const std::string& text) const { return ppm_.evaluate(text); }
    DiscourseTone target_tone(const SemanticPlan& plan) const;
private:
    void add_fragment(PhraseFragment fragment);
    void rebuild_index(bool use_gpu = false);
    std::span<const PhraseFragment> fragments() const;
    std::vector<PhraseFragment> owned_;
    std::span<const PhraseFragment> mapped_fragments_;
    struct Mapping;
    std::shared_ptr<Mapping> mapping_;
    PPMModel ppm_;
    KanervaSDM index_;
    SparseTsetlinModel style_;
    bool trained_ = false;
};
} // namespace cognicore
