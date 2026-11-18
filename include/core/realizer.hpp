#pragma once

#include "core/causal_graph.hpp"
#include "core/text_parser.hpp"
#include "core/tsetlin.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

struct SemanticPlan {
    DialogIntent intent = DialogIntent::ANSWER_FACT;
    std::vector<CausalPath> paths;
    std::vector<std::string> query_concepts;
    std::string language = "pt";
};

class ISurfaceRealizer {
public:
    virtual ~ISurfaceRealizer() = default;
    virtual std::string realize(const SemanticPlan& plan) const = 0;
};

class ConstructionGrammarRealizer : public ISurfaceRealizer {
public:
    ConstructionGrammarRealizer();

    void add_frame(const ConstructionFrame& frame);
    void observe_sentence_tokens(const std::vector<std::string>& tokens);
    void consolidate_frames();

    /// Scores fluency of a token sequence using smoothed bigram/trigram (CTW-style) log-probability.
    double score_fluency(const std::vector<std::string>& tokens) const;

    std::string realize(const SemanticPlan& plan) const override;

    size_t num_frames() const noexcept { return frames_.size(); }
    size_t num_ngrams() const noexcept { return bigrams_.size() + trigrams_.size(); }
    size_t memory_bytes() const noexcept;

    const std::vector<ConstructionFrame>& frames() const { return frames_; }
    const std::unordered_map<std::string, uint32_t>& unigrams() const { return unigrams_; }
    const std::unordered_map<std::string, uint32_t>& bigrams() const { return bigrams_; }
    const std::unordered_map<std::string, uint32_t>& trigrams() const { return trigrams_; }
    uint64_t total_tokens() const noexcept { return total_tokens_; }

    void load_raw(std::vector<ConstructionFrame> frames,
                  std::unordered_map<std::string, uint32_t> unigrams,
                  std::unordered_map<std::string, uint32_t> bigrams,
                  std::unordered_map<std::string, uint32_t> trigrams,
                  uint64_t total_tokens);

private:
    std::string fill_frame(const std::string& pattern,
                           const std::string& subj,
                           const std::string& obj) const;
    std::string realize_step(const CausalStep& step) const;
    std::string capitalize_first(std::string s) const;

    std::vector<ConstructionFrame> frames_;
    std::unordered_map<std::string, uint32_t> unigrams_;
    std::unordered_map<std::string, uint32_t> bigrams_;
    std::unordered_map<std::string, uint32_t> trigrams_;
    uint64_t total_tokens_ = 0;
};

} // namespace cognicore
