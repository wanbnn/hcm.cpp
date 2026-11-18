#include "core/realizer.hpp"
#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>

namespace cognicore {

ConstructionGrammarRealizer::ConstructionGrammarRealizer() {
    // Seed a few minimal baseline construction frames
    frames_.push_back({"[SUBJ:ENT] provoca [OBJ:ENT]", RelationType::CAUSAL, 1});
    frames_.push_back({"[SUBJ:ENT] gera [OBJ:ENT]", RelationType::CAUSAL, 1});
    frames_.push_back({"[SUBJ:ENT] impede [OBJ:ENT]", RelationType::INHIBITS, 1});
    frames_.push_back({"[SUBJ:ENT] é um [OBJ:ENT]", RelationType::TAXONOMIC, 1});
}

void ConstructionGrammarRealizer::add_frame(const ConstructionFrame& frame) {
    for (auto& f : frames_) {
        if (f.pattern == frame.pattern && f.rel_type == frame.rel_type) {
            f.frequency += frame.frequency;
            return;
        }
    }
    frames_.push_back(frame);
}

void ConstructionGrammarRealizer::observe_sentence_tokens(const std::vector<std::string>& tokens) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        unigrams_[tokens[i]] += 1;
        ++total_tokens_;
        if (i + 1 < tokens.size()) {
            bigrams_[tokens[i] + "\t" + tokens[i + 1]] += 1;
        }
        if (i + 2 < tokens.size()) {
            trigrams_[tokens[i] + "\t" + tokens[i + 1] + "\t" + tokens[i + 2]] += 1;
        }
    }
}

void ConstructionGrammarRealizer::consolidate_frames() {
    std::vector<ConstructionFrame> merged;
    for (const auto& f : frames_) {
        bool found = false;
        for (auto& m : merged) {
            if (m.pattern == f.pattern && m.rel_type == f.rel_type) {
                m.frequency += f.frequency;
                found = true;
                break;
            }
        }
        if (!found) merged.push_back(f);
    }
    std::sort(merged.begin(), merged.end(),
              [](const ConstructionFrame& a, const ConstructionFrame& b) {
                  return a.frequency > b.frequency;
              });
    frames_ = std::move(merged);
}

double ConstructionGrammarRealizer::score_fluency(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return -1e9;
    const double vocab = static_cast<double>(std::max<size_t>(1, unigrams_.size()));
    const double denom_uni = static_cast<double>(total_tokens_) + vocab;

    double logp = 0.0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it_u = unigrams_.find(tokens[i]);
        const double cu = (it_u != unigrams_.end()) ? static_cast<double>(it_u->second) : 0.0;
        double p1 = (cu + 1.0) / denom_uni;

        double p = 0.2 * p1;
        if (i >= 1) {
            auto it_prev = unigrams_.find(tokens[i - 1]);
            const double c_prev = (it_prev != unigrams_.end()) ? static_cast<double>(it_prev->second) : 0.0;
            auto it_b = bigrams_.find(tokens[i - 1] + "\t" + tokens[i]);
            const double cb = (it_b != bigrams_.end()) ? static_cast<double>(it_b->second) : 0.0;
            double p2 = (cb + 0.5) / (c_prev + 0.5 * vocab);
            p += 0.4 * p2;
        }
        if (i >= 2) {
            auto it_bp = bigrams_.find(tokens[i - 2] + "\t" + tokens[i - 1]);
            const double c_bp = (it_bp != bigrams_.end()) ? static_cast<double>(it_bp->second) : 0.0;
            auto it_t = trigrams_.find(tokens[i - 2] + "\t" + tokens[i - 1] + "\t" + tokens[i]);
            const double ct = (it_t != trigrams_.end()) ? static_cast<double>(it_t->second) : 0.0;
            double p3 = (ct + 0.25) / (c_bp + 0.25 * vocab);
            p += 0.4 * p3;
        }
        logp += std::log(std::max(p, 1e-12));
    }
    return logp / static_cast<double>(tokens.size());
}

std::string ConstructionGrammarRealizer::fill_frame(const std::string& pattern,
                                                    const std::string& subj,
                                                    const std::string& obj) const {
    std::string out = pattern;
    const std::string s_tag = "[SUBJ:ENT]";
    const std::string o_tag = "[OBJ:ENT]";
    size_t pos = out.find(s_tag);
    if (pos != std::string::npos) {
        out.replace(pos, s_tag.size(), subj);
    }
    pos = out.find(o_tag);
    if (pos != std::string::npos) {
        out.replace(pos, o_tag.size(), obj);
    }
    return out;
}

std::string ConstructionGrammarRealizer::realize_step(const CausalStep& step, const std::string& lang) const {
    // Compile a matching learned SVO construction into typed predicate slots.
    // A frame with the same relation category but a different verb is unsafe.
    std::string construction = "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]";
    uint32_t frequency = 0;
    const std::string subj = "[SUBJ:ENT]", obj = "[OBJ:ENT]";
    for (const auto& frame : frames_) {
        if (frame.rel_type != step.type || !frame.pattern.starts_with(subj) || !frame.pattern.ends_with(obj)) continue;
        auto predicate = frame.pattern.substr(subj.size(), frame.pattern.size() - subj.size() - obj.size());
        const auto begin = predicate.find_first_not_of(' '), end = predicate.find_last_not_of(' ');
        if (begin == std::string::npos) continue;
        predicate = predicate.substr(begin, end - begin + 1);
        if (TextParser::lemmatize_simple(predicate) != TextParser::lemmatize_simple(step.rel) || frame.frequency < frequency) continue;
        construction = frame.pattern;
        construction.replace(subj.size(), frame.pattern.size() - subj.size() - obj.size(), " [PRED:VERB] ");
        frequency = frame.frequency;
    }
    auto subject = MorphologicalRealizer::definite_article(step.src_gender, step.src_number, lang) + " " + step.src;
    const auto lower = TextParser::utf8_tolower(step.src);
    for (const auto* article : {"o ", "a ", "os ", "as ", "the "}) if (lower.starts_with(article)) subject = step.src;
    return capitalize_first(MorphologicalRealizer{}.realize_clause(step, subject, lang, construction));
}

std::string ConstructionGrammarRealizer::capitalize_first(std::string s) const {
    if (!s.empty()) {
        unsigned char c = static_cast<unsigned char>(s[0]);
        if (c < 0x80) {
            s[0] = static_cast<char>(std::toupper(c));
        }
    }
    return s;
}

std::string ConstructionGrammarRealizer::realize(const SemanticPlan& plan) const {
    // All candidates are structural realizations of exactly the same steps.
    // Corpus statistics rank wording; they cannot select facts or change verbs.
    const MorphologicalRealizer morphology;
    const auto base = morphology.realize(plan);
    if (plan.paths.empty() || plan.paths[0].steps.empty() ||
        plan.intent == DialogIntent::GREET || plan.intent == DialogIntent::AGREE ||
        plan.intent == DialogIntent::ASK_CLARIFY) return base;

    const auto& steps = plan.paths[0].steps;
    std::vector<std::string> clauses;
    bool causal_chain = steps.size() > 1;
    for (size_t i = 0; i < steps.size(); ++i) {
        clauses.push_back(realize_step(steps[i], plan.language));
        if (steps[i].negated || !steps[i].modality.empty() || steps[i].type != RelationType::CAUSAL ||
            (i && steps[i-1].dst != steps[i].src)) causal_chain = false;
    }
    std::vector<std::string> candidates{base};
    std::string separate;
    for (const auto& clause : clauses) {
        if (!separate.empty()) separate += " ";
        separate += clause + ".";
    }
    candidates.push_back(std::move(separate));
    const bool english = plan.language == "en";
    if (clauses.size() == 1 && plan.intent == DialogIntent::EXPLAIN_CAUSE) {
        auto clause = clauses[0];
        if (!clause.empty() && static_cast<unsigned char>(clause[0]) < 128)
            clause[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(clause[0])));
        candidates.push_back((english ? "This happens because " : "Isso ocorre porque ") + clause + ".");
    }
    if (causal_chain) {
        const std::vector<std::string> connectors = english
            ? std::vector<std::string>{"; therefore, ", "; as a result, "}
            : std::vector<std::string>{"; portanto, ", "; por isso, ", "; assim, "};
        for (const auto& connector : connectors) {
            std::string composed = clauses[0];
            for (size_t i = 1; i < clauses.size(); ++i) {
                auto clause = clauses[i];
                if (!clause.empty() && static_cast<unsigned char>(clause[0]) < 128)
                    clause[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(clause[0])));
                composed += connector + clause;
            }
            candidates.push_back(composed + ".");
        }
    }
    TextParser tokenizer;
    std::string best = candidates.front();
    double score = -std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        const auto current = score_fluency(tokenizer.tokenize(candidate));
        if (current > score) { score = current; best = candidate; }
    }
    return best;
}

std::string serialize_semantic_plan(const SemanticPlan& plan) {
    std::ostringstream out;
    out << "<PLAN> <LANG: " << plan.language << "> <INTENT: " << intent_to_string(plan.intent) << ">";
    if (!plan.paths.empty()) for (const auto& step : plan.paths[0].steps) {
        out << " <SUBJ: " << step.src << "> <REL: " << step.rel << "> <OBJ: " << step.dst << ">"
            << " <TENSE: " << unsigned(step.tense) << "> <SUBJ_GENDER: " << unsigned(step.src_gender)
            << "> <SUBJ_NUMBER: " << unsigned(step.src_number) << "> <OBJ_GENDER: " << unsigned(step.dst_gender)
            << "> <OBJ_NUMBER: " << unsigned(step.dst_number) << "> <TYPE: " << unsigned(step.type) << ">";
        if (step.negated) out << " <NEG: true>";
        if (!step.modality.empty()) out << " <MODAL: " << step.modality << ">";
    }
    out << " </PLAN>";
    return out.str();
}

size_t ConstructionGrammarRealizer::memory_bytes() const noexcept {
    size_t bytes = frames_.size() * sizeof(ConstructionFrame);
    for (const auto& f : frames_) bytes += f.pattern.capacity();
    bytes += (unigrams_.size() + bigrams_.size() + trigrams_.size()) * 48;
    return bytes;
}

void ConstructionGrammarRealizer::load_raw(std::vector<ConstructionFrame> frames,
                                           std::unordered_map<std::string, uint32_t> unigrams,
                                           std::unordered_map<std::string, uint32_t> bigrams,
                                           std::unordered_map<std::string, uint32_t> trigrams,
                                           uint64_t total_tokens) {
    frames_ = std::move(frames);
    unigrams_ = std::move(unigrams);
    bigrams_ = std::move(bigrams);
    trigrams_ = std::move(trigrams);
    total_tokens_ = total_tokens;
}

} // namespace cognicore
