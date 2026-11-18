#include "core/realizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

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

std::string ConstructionGrammarRealizer::realize_step(const CausalStep& step) const {
    // Prefer exact step relation, or select a matching learned frame
    std::string direct = step.src + " " + step.rel + " " + step.dst;
    for (const auto& f : frames_) {
        if (f.rel_type == step.type && f.pattern.find(step.rel) != std::string::npos) {
            return fill_frame(f.pattern, step.src, step.dst);
        }
    }
    return direct;
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
    if (plan.intent == DialogIntent::GREET) {
        return "Olá! Estou pronto para raciocinar e aprender novos conceitos com você.";
    }
    if (plan.intent == DialogIntent::AGREE) {
        return "Sim, isso está coerente com as relações causais registradas na memória.";
    }
    if (plan.intent == DialogIntent::ASK_CLARIFY || plan.paths.empty()) {
        if (plan.query_concepts.empty()) {
            return "Poderia detalhar melhor os conceitos envolvidos para que eu possa inferir a relação?";
        }
        return "Ainda não tenho conexões suficientes sobre " + plan.query_concepts[0] +
               ". Você pode me ensinar uma relação causal envolvendo esse conceito?";
    }

    const auto& best_path = plan.paths[0];
    if (best_path.steps.empty()) {
        return "Não encontrei um caminho causal completo para essa consulta.";
    }

    TextParser tokenizer_helper;
    std::vector<std::string> candidates;

    if (best_path.steps.size() == 1) {
        const auto& st = best_path.steps[0];
        std::string base = realize_step(st);
        if (plan.intent == DialogIntent::HYPOTHESIZE) {
            candidates.push_back("Se " + base + ", então há um efeito direto sobre " + st.dst + ".");
            candidates.push_back("Hipoteticamente, " + base + ".");
        } else if (plan.intent == DialogIntent::EXPLAIN_CAUSE) {
            candidates.push_back("Isso ocorre porque " + base + ".");
            candidates.push_back(capitalize_first(base) + ".");
        } else {
            candidates.push_back(capitalize_first(base) + ".");
        }

        // Also try compatible frames of same RelationType
        for (const auto& f : frames_) {
            if (f.rel_type == st.type) {
                candidates.push_back(capitalize_first(fill_frame(f.pattern, st.src, st.dst)) + ".");
                if (candidates.size() >= 6) break;
            }
        }
    } else {
        // Multi-hop compositional synthesis (combines multiple steps into a novel sentence)
        const auto& s0 = best_path.steps[0];
        const auto& s_last = best_path.steps.back();
        std::string clause0 = realize_step(s0);

        std::string chain_mid;
        for (size_t i = 1; i < best_path.steps.size(); ++i) {
            const auto& si = best_path.steps[i];
            if (i == 1) {
                chain_mid += si.rel + " " + si.dst;
            } else {
                chain_mid += " e " + si.src + " " + si.rel + " " + si.dst;
            }
        }

        candidates.push_back(
            capitalize_first(clause0) + ", o que por sua vez " + chain_mid +
            "; portanto, " + s0.src + " leva a " + s_last.dst + ".");
        candidates.push_back(
            "Como " + clause0 + ", isso faz com que " + realize_step(best_path.steps[1]) +
            ", conectando " + s0.src + " a " + s_last.dst + ".");
        candidates.push_back(
            "Pela cadeia causal, " + clause0 + " e " + realize_step(best_path.steps[1]) + ".");
    }

    // Score candidates with N-gram / CTW fluency scorer
    std::string best_sent = candidates[0];
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& cand : candidates) {
        auto toks = tokenizer_helper.tokenize(cand);
        double sc = score_fluency(toks);
        if (sc > best_score) {
            best_score = sc;
            best_sent = cand;
        }
    }
    return best_sent;
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
