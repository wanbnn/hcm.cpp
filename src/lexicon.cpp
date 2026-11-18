#include "core/lexicon.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace cognicore {
namespace {

constexpr uint32_t MAGIC = 0x4C455831; // LEX1
constexpr uint32_t VERSION = 1;

void write_string(std::ofstream& out, const std::string& value) {
    const uint32_t size = static_cast<uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size) out.write(value.data(), size);
}

bool read_string(std::ifstream& in, std::string& value) {
    uint32_t size = 0;
    if (!in.read(reinterpret_cast<char*>(&size), sizeof(size))) return false;
    if (size > 1'000'000) return false;
    value.resize(size);
    if (size && !in.read(value.data(), size)) return false;
    return true;
}

std::vector<std::vector<std::string>> parse_groups(const std::vector<std::string>& encoded) {
    std::vector<std::vector<std::string>> groups;
    for (const auto& row : encoded) {
        std::vector<std::string> group;
        size_t start = 0;
        while (start <= row.size()) {
            const auto separator = row.find('|', start);
            const auto piece = separator == std::string::npos ? row.substr(start) : row.substr(start, separator - start);
            if (!piece.empty()) group.push_back(TextParser::utf8_tolower(piece));
            if (separator == std::string::npos) break;
            start = separator + 1;
        }
        if (group.size() > 1) groups.push_back(std::move(group));
    }
    return groups;
}

} // namespace

size_t Lexicon::tone_index(const std::string& tone) {
    if (tone == "didactic" || tone == "didatico") return 1;
    if (tone == "casual") return 2;
    if (tone == "warning" || tone == "alerta") return 3;
    return 0;
}

size_t Lexicon::memory_bytes() const {
    size_t bytes = entries_.capacity() * sizeof(LexiconEntry);
    for (const auto& entry : entries_) bytes += entry.lemma.capacity() + entry.pos.capacity();
    for (const auto& [key, count] : collocations_) bytes += key.capacity() + sizeof(count) + 32;
    return bytes;
}

void Lexicon::clear() {
    entries_.clear();
    index_.clear();
    collocations_.clear();
}

void Lexicon::load_synonyms(const LanguageRules& rules) {
    verb_groups_ = parse_groups(rules.verb_synonyms);
    noun_groups_ = parse_groups(rules.noun_synonyms);
}

void Lexicon::observe(const std::string& text, const TextParser& parser, const std::string& tone) {
    const auto document = parser.parse_text(text);
    const auto tone_at = tone_index(tone);
    auto add = [&](const std::string& lemma, const std::string& pos,
                   GramGender gender, GramNumber number) {
        const auto key = TextParser::utf8_tolower(lemma);
        if (key.size() < 2 || parser.is_stopword(key)) return;
        auto it = index_.find(key);
        if (it == index_.end()) {
            LexiconEntry entry;
            entry.lemma = key;
            entry.pos = pos;
            entry.gender = static_cast<uint8_t>(gender);
            entry.number = static_cast<uint8_t>(number);
            index_[key] = static_cast<uint32_t>(entries_.size());
            entries_.push_back(std::move(entry));
            it = index_.find(key);
        }
        auto& entry = entries_[it->second];
        ++entry.frequency;
        if (tone_at < entry.tone.size()) ++entry.tone[tone_at];
    };
    for (const auto& sentence : document.morph_sentences)
        for (const auto& token : sentence)
            add(token.lemma.empty() ? token.form : token.lemma, token.upos, token.gender, token.number);
    for (const auto& triple : document.triples) {
        const auto verb = TextParser::utf8_tolower(TextParser::lemmatize_simple(triple.relation));
        const auto object = TextParser::utf8_tolower(triple.object);
        const auto subject = TextParser::utf8_tolower(triple.subject);
        if (verb.empty() || object.empty()) continue;
        ++collocations_[verb + "\t" + object];
        if (!subject.empty()) ++collocations_[subject + "\t" + verb];
    }
}

std::string Lexicon::choose_verb(const std::string& lemma, const std::string& object,
                                 const std::string& tone) const {
    const auto base = TextParser::utf8_tolower(TextParser::lemmatize_simple(lemma));
    const auto needle = base + "\t" + TextParser::utf8_tolower(object);
    uint32_t best_score = 0;
    std::string best = lemma;
    auto score_for = [&](const std::string& candidate) {
        const auto it = collocations_.find(candidate + "\t" + TextParser::utf8_tolower(object));
        const uint32_t collocation = it == collocations_.end() ? 0 : it->second;
        const auto entry = index_.find(candidate);
        const uint32_t frequency = entry == index_.end() ? 0 : entries_[entry->second].frequency;
        return collocation * 100 + frequency;
    };
    const uint32_t base_score = score_for(base);
    if (base_score) { best_score = base_score; }
    for (const auto& group : verb_groups_) {
        if (std::find(group.begin(), group.end(), base) == group.end()) continue;
        for (const auto& candidate : group) {
            if (candidate == base) continue;
            const auto candidate_score = score_for(candidate);
            // Only use a synonym with actual collocation evidence.
            const auto it = collocations_.find(candidate + "\t" + TextParser::utf8_tolower(object));
            if (it == collocations_.end() || it->second == 0) continue;
            if (candidate_score > best_score) { best_score = candidate_score; best = candidate; }
        }
        break;
    }
    (void)tone;
    (void)needle;
    return best;
}

std::string Lexicon::choose_noun(const std::string& noun, const std::string& relation,
                                 const std::string& tone) const {
    const auto base = TextParser::utf8_tolower(noun);
    const auto relation_lemma = TextParser::utf8_tolower(TextParser::lemmatize_simple(relation));
    const auto entry = index_.find(base);
    if (entry == index_.end()) return noun;
    const auto gender = entries_[entry->second].gender;
    const auto number = entries_[entry->second].number;
    auto score_for = [&](const std::string& candidate) {
        const auto forward = collocations_.find(relation_lemma + "\t" + candidate);
        const auto backward = collocations_.find(candidate + "\t" + relation_lemma);
        const uint32_t collocation = (forward == collocations_.end() ? 0 : forward->second) +
                                     (backward == collocations_.end() ? 0 : backward->second);
        const auto found = index_.find(candidate);
        const uint32_t frequency = found == index_.end() ? 0 : entries_[found->second].frequency;
        return collocation * 100 + frequency;
    };
    uint32_t best_score = score_for(base);
    std::string best = noun;
    for (const auto& group : noun_groups_) {
        if (std::find(group.begin(), group.end(), base) == group.end()) continue;
        for (const auto& candidate : group) {
            if (candidate == base) continue;
            const auto found = index_.find(candidate);
            if (found == index_.end()) continue;
            if (entries_[found->second].gender != gender || entries_[found->second].number != number) continue;
            const auto forward = collocations_.find(relation_lemma + "\t" + candidate);
            const auto backward = collocations_.find(candidate + "\t" + relation_lemma);
            if ((forward == collocations_.end() || forward->second == 0) &&
                (backward == collocations_.end() || backward->second == 0)) continue;
            const auto candidate_score = score_for(candidate);
            if (candidate_score > best_score) { best_score = candidate_score; best = candidate; }
        }
        break;
    }
    (void)tone;
    return best;
}

bool Lexicon::save(const std::string& path) const {
    const auto temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary);
    if (!out) return false;
    const uint32_t entry_count = static_cast<uint32_t>(entries_.size());
    const uint32_t collocation_count = static_cast<uint32_t>(collocations_.size());
    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
    out.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));
    out.write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));
    out.write(reinterpret_cast<const char*>(&collocation_count), sizeof(collocation_count));
    for (const auto& entry : entries_) {
        write_string(out, entry.lemma);
        write_string(out, entry.pos);
        out.write(reinterpret_cast<const char*>(&entry.frequency), sizeof(entry.frequency));
        out.write(reinterpret_cast<const char*>(entry.tone.data()), entry.tone.size() * sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&entry.gender), sizeof(entry.gender));
        out.write(reinterpret_cast<const char*>(&entry.number), sizeof(entry.number));
    }
    for (const auto& [key, count] : collocations_) {
        write_string(out, key);
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    }
    out.close();
    if (!out) return false;
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    return !ec;
}

bool Lexicon::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t magic = 0, version = 0, entry_count = 0, collocation_count = 0;
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || magic != MAGIC ||
        !in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != VERSION ||
        !in.read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count)) || entry_count > 2'000'000 ||
        !in.read(reinterpret_cast<char*>(&collocation_count), sizeof(collocation_count)) || collocation_count > 20'000'000)
        return false;
    std::vector<LexiconEntry> entries(entry_count);
    for (auto& entry : entries) {
        if (!read_string(in, entry.lemma) || !read_string(in, entry.pos) ||
            !in.read(reinterpret_cast<char*>(&entry.frequency), sizeof(entry.frequency)) ||
            !in.read(reinterpret_cast<char*>(entry.tone.data()), entry.tone.size() * sizeof(uint32_t)) ||
            !in.read(reinterpret_cast<char*>(&entry.gender), sizeof(entry.gender)) ||
            !in.read(reinterpret_cast<char*>(&entry.number), sizeof(entry.number))) return false;
    }
    std::unordered_map<std::string, uint32_t> collocations;
    for (uint32_t i = 0; i < collocation_count; ++i) {
        std::string key;
        uint32_t count = 0;
        if (!read_string(in, key) || !in.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;
        collocations.emplace(std::move(key), count);
    }
    entries_ = std::move(entries);
    index_.clear();
    for (uint32_t i = 0; i < entries_.size(); ++i) index_[entries_[i].lemma] = i;
    collocations_ = std::move(collocations);
    return true;
}

SemanticPlan LexicalChooser::apply(const SemanticPlan& plan, const Lexicon& lexicon) const {
    SemanticPlan selected = plan;
    const auto tone = plan.tone.empty() ? std::string("technical") : plan.tone;
    for (auto& path : selected.paths) {
        for (auto& step : path.steps) {
            step.surface_rel = lexicon.choose_verb(step.rel, step.dst, tone);
            step.surface_src = lexicon.choose_noun(step.src, step.surface_rel, tone);
            step.surface_dst = lexicon.choose_noun(step.dst, step.surface_rel, tone);
        }
    }
    return selected;
}

} // namespace cognicore
