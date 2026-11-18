#include "core/coherence.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>

namespace cognicore {
namespace {

constexpr uint32_t MAGIC = 0x434F4831; // COH1
constexpr uint32_t VERSION = 1;
constexpr double DISCOUNT = 0.75;

void write_string(std::ofstream& out, const std::string& value) {
    const uint32_t size = static_cast<uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size) out.write(value.data(), size);
}

bool read_string(std::ifstream& in, std::string& value) {
    uint32_t size = 0;
    if (!in.read(reinterpret_cast<char*>(&size), sizeof(size)) || size > 1'000'000) return false;
    value.resize(size);
    return size == 0 || static_cast<bool>(in.read(value.data(), size));
}

bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

} // namespace

size_t CoherenceModel::memory_bytes() const {
    size_t bytes = (unigrams_.size() + bigrams_.size() + trigrams_.size()) * 64;
    for (const auto& [key, value] : unigrams_) bytes += key.capacity();
    for (const auto& [key, value] : bigrams_) bytes += key.capacity();
    for (const auto& [key, value] : trigrams_) bytes += key.capacity();
    return bytes;
}

void CoherenceModel::clear() {
    unigrams_.clear();
    bigrams_.clear();
    trigrams_.clear();
    total_ = 0;
}

void CoherenceModel::observe(const std::string& text, const TextParser& parser) {
    for (const auto& sentence : TextParser::split_sentences(text)) {
        const auto tokens = parser.tokenize(sentence);
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            ++unigrams_[token];
            ++total_;
            if (i >= 1) ++bigrams_[tokens[i - 1] + "\t" + token];
            if (i >= 2) ++trigrams_[tokens[i - 2] + "\t" + tokens[i - 1] + "\t" + token];
        }
    }
}

double CoherenceModel::sentence_cost(const std::vector<std::string>& tokens) const {
    if (tokens.empty() || unigrams_.empty()) return 0.0;
    const double vocabulary = std::max<size_t>(1, unigrams_.size());
    const double denominator = double(total_) + vocabulary;
    double bits = 0.0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto count = [&](const std::unordered_map<std::string, uint32_t>& table,
                         const std::string& key) {
            const auto it = table.find(key);
            return it == table.end() ? 0.0 : double(it->second);
        };
        const auto uni_it = unigrams_.find(tokens[i]);
        const double uni_count = uni_it == unigrams_.end() ? 0.0 : double(uni_it->second);
        const double p_uni = (uni_count + 1.0) / denominator;
        double probability = p_uni;
        if (i >= 1) {
            const auto key2 = tokens[i - 1] + "\t" + tokens[i];
            const double c2 = count(bigrams_, key2);
            const auto history2 = unigrams_.find(tokens[i - 1]);
            const double h2 = history2 == unigrams_.end() ? 0.0 : double(history2->second);
            const double p2 = h2 > 0 ? (std::max(c2 - DISCOUNT, 0.0) / h2 + 0.4 * p_uni) : p_uni;
            probability = p2;
            if (i >= 2) {
                const auto key3 = tokens[i - 2] + "\t" + tokens[i - 1] + "\t" + tokens[i];
                const double c3 = count(trigrams_, key3);
                const auto history3 = bigrams_.find(tokens[i - 2] + "\t" + tokens[i - 1]);
                const double h3 = history3 == bigrams_.end() ? 0.0 : double(history3->second);
                if (h3 > 0) probability = std::max(c3 - DISCOUNT, 0.0) / h3 + 0.4 * p2;
            }
        }
        bits += -std::log2(std::max(probability, 1e-9));
    }
    return bits / double(tokens.size());
}

CoherenceScore CoherenceModel::score(const std::string& candidate,
                                     const SemanticPlan& plan,
                                     const std::string& previous_sentence,
                                     const CausalGraph& graph,
                                     const LanguageRules& rules) const {
    CoherenceScore score;
    TextParser parser;
    const auto tokens = parser.tokenize(candidate);
    score.language = sentence_cost(tokens);
    const auto content = parser.extract_content_tokens(candidate);
    const auto previous = parser.extract_content_tokens(previous_sentence);
    if (!tokens.empty() && contains(rules.centering_pronouns, tokens.front())) score.centering += 2.5;
    for (const auto& token : content)
        if (contains(previous, token)) { score.centering += 0.8; break; }
    if (!content.empty() && !previous.empty() && content.front() == previous.front() && content.front().size() > 2)
        score.centering -= 0.8;
    std::vector<std::string> grounded = plan.query_concepts;
    for (const auto& path : plan.paths) for (const auto& step : path.steps) {
        grounded.push_back(step.src);
        grounded.push_back(step.rel);
        grounded.push_back(step.dst);
    }
    for (const auto& token : content) {
        if (contains(grounded, token) || graph.find_node(token) != UINT32_MAX) score.grounding += 0.4;
        else score.grounding -= 0.2;
    }
    score.total = -score.language + score.centering + score.grounding;
    return score;
}

std::string CoherenceModel::choose(const std::vector<std::string>& candidates,
                                   const SemanticPlan& plan,
                                   const std::string& previous_sentence,
                                   const CausalGraph& graph,
                                   const LanguageRules& rules) const {
    if (candidates.empty()) return {};
    std::string best = candidates.front();
    double best_score = -1e18;
    for (const auto& candidate : candidates) {
        const auto score = this->score(candidate, plan, previous_sentence, graph, rules);
        if (score.total > best_score) { best_score = score.total; best = candidate; }
    }
    return best;
}

bool CoherenceModel::save(const std::string& path) const {
    const auto temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    if (!out) return false;
    const uint32_t u = static_cast<uint32_t>(unigrams_.size());
    const uint32_t b = static_cast<uint32_t>(bigrams_.size());
    const uint32_t t = static_cast<uint32_t>(trigrams_.size());
    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
    out.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    out.write(reinterpret_cast<const char*>(&total_), sizeof(total_));
    out.write(reinterpret_cast<const char*>(&u), sizeof(u));
    out.write(reinterpret_cast<const char*>(&b), sizeof(b));
    out.write(reinterpret_cast<const char*>(&t), sizeof(t));
    auto write_table = [&](const std::unordered_map<std::string, uint32_t>& table) {
        for (const auto& [key, value] : table) {
            write_string(out, key);
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    };
    write_table(unigrams_);
    write_table(bigrams_);
    write_table(trigrams_);
    out.close();
    if (!out) return false;
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    return !ec;
}

bool CoherenceModel::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t magic = 0, version = 0, u = 0, b = 0, t = 0;
    uint64_t total = 0;
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != MAGIC ||
        !in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != VERSION ||
        !in.read(reinterpret_cast<char*>(&total), sizeof(total)) ||
        !in.read(reinterpret_cast<char*>(&u), sizeof(u)) || u > 20'000'000 ||
        !in.read(reinterpret_cast<char*>(&b), sizeof(b)) || b > 40'000'000 ||
        !in.read(reinterpret_cast<char*>(&t), sizeof(t)) || t > 80'000'000) return false;
    auto read_table = [&](std::unordered_map<std::string, uint32_t>& table, uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            std::string key;
            uint32_t value = 0;
            if (!read_string(in, key) || !in.read(reinterpret_cast<char*>(&value), sizeof(value))) return false;
            table.emplace(std::move(key), value);
        }
        return true;
    };
    std::unordered_map<std::string, uint32_t> unigrams, bigrams, trigrams;
    if (!read_table(unigrams, u) || !read_table(bigrams, b) || !read_table(trigrams, t)) return false;
    unigrams_ = std::move(unigrams);
    bigrams_ = std::move(bigrams);
    trigrams_ = std::move(trigrams);
    total_ = total;
    return true;
}

} // namespace cognicore
