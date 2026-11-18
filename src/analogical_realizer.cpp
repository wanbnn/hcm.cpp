#include "core/analogical_realizer.hpp"
#include "core/morphological_realizer.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <set>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
namespace cognicore {
namespace {
constexpr size_t MAX_CONTEXTS = 80000, MAX_ENTRIES = 400000, MAX_FRAGMENTS = 2048;
// Integer approximation of log2 in Q16; statistical ranking has no dense float operations.
int64_t log_q16(uint64_t n) {
    if (!n) return 0;
    const unsigned k = std::bit_width(n) - 1;
    const uint64_t base = uint64_t(1) << k;
    return (int64_t(k) << 16) + int64_t(((n - base) << 16) / base);
}
bool less_record(const PPMRecord& a, const PPMRecord& b) {
    if (a.order != b.order) return a.order < b.order;
    if (a.context != b.context) return a.context < b.context;
    return a.symbol < b.symbol;
}
SDMConfig index_config() { SDMConfig c; c.num_locations = 256; c.top_k = 8; return c; }
HyperVector shape_vector(PhraseShape shape, DiscourseTone tone, bool english, DialogIntent intent) {
    return HyperVector::from_token("INTENT:" + std::string(intent_to_string(intent))).bind(HyperVector::from_token("TONE:" + std::to_string(unsigned(tone))))
        .bind(HyperVector::from_token("SHAPE:" + std::to_string(unsigned(shape))))
        .bind(HyperVector::from_token(english ? "LANG:en" : "LANG:pt"));
}
HyperVector fragment_vector(uint32_t id) { return HyperVector::from_token("DOP_FRAGMENT:" + std::to_string(id)); }
void field(char* dst, size_t capacity, const std::string& text) {
    std::memset(dst, 0, capacity); std::memcpy(dst, text.data(), std::min(capacity - 1, text.size()));
}
std::string subject_phrase(const CausalStep& step, const std::string& lang) {
    const auto& name = step.surface_src.empty() ? step.src : step.surface_src;
    const auto lower = TextParser::utf8_tolower(name);
    for (const auto* a : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
        if (lower.starts_with(a)) return name;
    return MorphologicalRealizer::definite_article(step.src_gender, step.src_number, lang) + " " + name;
}
std::string capitalize(std::string s) { if (!s.empty() && static_cast<unsigned char>(s[0]) < 128) s[0] = char(std::toupper(static_cast<unsigned char>(s[0]))); return s; }
std::string lower_initial(std::string s) {
    if (!s.empty() && static_cast<unsigned char>(s[0]) < 128) s[0] = char(std::tolower(static_cast<unsigned char>(s[0])));
    return s;
}
uint64_t style_features(DiscourseTone tone, bool long_text, bool pronoun) {
    return (1ULL << unsigned(tone)) | (uint64_t(long_text) << 4) | (uint64_t(pronoun) << 5) | (1ULL << 6);
}
struct ArchiveHeader { uint32_t magic, version, fragments, records, clauses, flags; };
struct DiskClause { uint16_t positive, negative; int8_t sign; uint8_t padding[3]{}; };
}
void PPMModel::materialize() {
    if (mapped_.empty()) return;
    for (const auto& r : mapped_) {
        std::string key(reinterpret_cast<const char*>(r.context.data()), r.order);
        counts_[key][r.symbol] = r.count;
    }
    entries_ = mapped_.size(); mapped_ = {};
}
void PPMModel::observe(const std::string& text) {
    materialize();
    for (size_t i = 0; i < text.size(); ++i) for (size_t order = 0; order <= std::min<size_t>(5, i); ++order) {
        auto key = text.substr(i - order, order);
        auto context = counts_.find(key);
        if (context == counts_.end()) {
            if (counts_.size() >= MAX_CONTEXTS || entries_ >= MAX_ENTRIES) continue;
            context = counts_.emplace(std::move(key), std::map<uint8_t, uint32_t>{}).first;
        }
        const auto symbol = static_cast<uint8_t>(text[i]);
        auto found = context->second.find(symbol);
        if (found == context->second.end()) {
            if (entries_ >= MAX_ENTRIES) continue;
            context->second.emplace(symbol, 1); ++entries_;
        } else if (found->second < UINT32_MAX) ++found->second;
    }
}
std::vector<PPMRecord> PPMModel::records() const {
    if (!mapped_.empty()) return {mapped_.begin(), mapped_.end()};
    std::vector<PPMRecord> result;
    result.reserve(entries_);
    for (const auto& [context, symbols] : counts_) for (const auto& [symbol, count] : symbols) {
        PPMRecord r; r.order = context.size(); r.symbol = symbol; r.count = count;
        std::memcpy(r.context.data(), context.data(), context.size()); result.push_back(r);
    }
    std::sort(result.begin(), result.end(), less_record); return result;
}
void PPMModel::attach(std::span<const PPMRecord> data) { counts_.clear(); entries_ = 0; mapped_ = data; }
size_t PPMModel::memory_bytes() const {
    if (!mapped_.empty()) return mapped_.size_bytes();
    size_t bytes = counts_.size() * (sizeof(std::string) + 96);
    for (const auto& [key, symbols] : counts_) bytes += key.capacity() + symbols.size() * 48;
    return bytes;
}
size_t PPMModel::contexts() const {
    if (mapped_.empty()) return counts_.size();
    size_t n = 0;
    for (size_t i = 0; i < mapped_.size(); ++i)
        if (!i || mapped_[i].order != mapped_[i-1].order || mapped_[i].context != mapped_[i-1].context) ++n;
    return n;
}
int64_t PPMModel::cost_q16(const std::string& text) const {
    if (text.empty()) return 0;
    int64_t bits = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        bool found = false;
        for (int order = int(std::min<size_t>(5, i)); order >= 0; --order) {
            uint64_t total = 0, distinct = 0, count = 0;
            const auto symbol = static_cast<uint8_t>(text[i]);
            if (!mapped_.empty()) {
                PPMRecord key; key.order = order;
                std::memcpy(key.context.data(), text.data() + i - order, order);
                auto it = std::lower_bound(mapped_.begin(), mapped_.end(), key, less_record);
                for (; it != mapped_.end() && it->order == order && it->context == key.context; ++it) {
                    total += it->count; ++distinct; if (it->symbol == symbol) count = it->count;
                }
            } else {
                const auto it = counts_.find(text.substr(i - order, order));
                if (it != counts_.end()) for (const auto& [c, n] : it->second) {
                    total += n; ++distinct; if (c == symbol) count = n;
                }
            }
            if (!total) continue;
            // Escape method C, without exclusion: total+distinct denominator.
            if (count) { bits += log_q16(total + distinct) - log_q16(count); found = true; break; }
            bits += log_q16(total + distinct) - log_q16(distinct);
        }
        if (!found) bits += 8LL << 16; // order -1, fixed byte alphabet
    }
    return bits / int64_t(text.size());
}
PPMEvaluation PPMModel::evaluate(const std::string& text) const {
    PPMEvaluation result;
    result.bytes = text.size();
    if (text.empty()) return result;
    double bits = 0.0;
    for (size_t i = 0; i < text.size(); ++i) {
        bool found = false;
        for (int order = int(std::min<size_t>(5, i)); order >= 0; --order) {
            uint64_t total = 0, distinct = 0, count = 0;
            const auto symbol = static_cast<uint8_t>(text[i]);
            if (!mapped_.empty()) {
                PPMRecord key; key.order = static_cast<uint8_t>(order);
                std::memcpy(key.context.data(), text.data() + i - order, order);
                auto it = std::lower_bound(mapped_.begin(), mapped_.end(), key, less_record);
                for (; it != mapped_.end() && it->order == order && it->context == key.context; ++it) {
                    total += it->count; ++distinct;
                    if (it->symbol == symbol) count = it->count;
                }
            } else {
                const auto it = counts_.find(text.substr(i - order, order));
                if (it != counts_.end()) for (const auto& [candidate, n] : it->second) {
                    total += n; ++distinct;
                    if (candidate == symbol) count = n;
                }
            }
            if (!total) continue;
            if (count) {
                bits += std::log2(double(total + distinct)) - std::log2(double(count));
                found = true;
                break;
            }
            bits += std::log2(double(total + distinct)) - std::log2(double(distinct));
        }
        if (!found) bits += 8.0;
    }
    result.bits_per_byte = bits / double(text.size());
    result.perplexity = std::pow(2.0, result.bits_per_byte);
    return result;
}

struct AnalogicalFluencyRealizer::Mapping {
    int fd = -1; void* data = MAP_FAILED; size_t size = 0;
    ~Mapping() { if (data != MAP_FAILED) ::munmap(data, size); if (fd >= 0) ::close(fd); }
};
AnalogicalFluencyRealizer::AnalogicalFluencyRealizer() : index_(index_config()) {
    // Structural floor. Additional productions are learned only from observed markers.
    for (bool en : {false, true}) {
        PhraseFragment clause; clause.english = en; clause.relation = 255;
        field(clause.clause_pattern, sizeof(clause.clause_pattern), "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]"); add_fragment(clause);
        // Typed clause templates. They are only licensed for the matching graph
        // relation, so a taxonomic fact is never realized as an arbitrary action.
        PhraseFragment taxonomic = clause;
        taxonomic.relation = static_cast<uint8_t>(RelationType::TAXONOMIC);
        field(taxonomic.clause_pattern, sizeof(taxonomic.clause_pattern),
              en ? "[SUBJ:ENT] is [OBJ:ENT]" : "[SUBJ:ENT] é [OBJ:ENT]");
        add_fragment(taxonomic);
        PhraseFragment property = taxonomic;
        property.relation = static_cast<uint8_t>(RelationType::PROPERTY);
        add_fragment(property);
        PhraseFragment sequence = clause; sequence.shape = PhraseShape::SEQUENCE;
        field(sequence.connective, sizeof(sequence.connective), ". "); add_fragment(sequence);
        PhraseFragment contrast = sequence; contrast.shape = PhraseShape::CONTRAST;
        contrast.tone = DiscourseTone::WARNING;
        field(contrast.connective, sizeof(contrast.connective), en ? "; however, " : "; entretanto, ");
        add_fragment(contrast);
    }
}
std::span<const PhraseFragment> AnalogicalFluencyRealizer::fragments() const {
    return mapped_fragments_.empty() ? std::span<const PhraseFragment>(owned_) : mapped_fragments_;
}
size_t AnalogicalFluencyRealizer::num_fragments() const { return fragments().size(); }
size_t AnalogicalFluencyRealizer::memory_bytes() const {
    return ppm_.memory_bytes() + fragments().size_bytes() + index_.memory_bytes() + style_.memory_bytes();
}
void AnalogicalFluencyRealizer::add_fragment(PhraseFragment f) {
    if (!mapped_fragments_.empty()) { owned_.assign(mapped_fragments_.begin(), mapped_fragments_.end()); mapped_fragments_ = {}; }
    for (auto& old : owned_) if (old.shape == f.shape && old.tone == f.tone && old.english == f.english &&
        old.relation == f.relation &&
        std::strcmp(old.connective, f.connective) == 0 && std::strcmp(old.clause_pattern, f.clause_pattern) == 0 && std::strcmp(old.predicate, f.predicate) == 0) {
        if (old.frequency < UINT32_MAX) ++old.frequency;
        return;
    }
    if (owned_.size() >= MAX_FRAGMENTS) return;
    f.id = owned_.size(); owned_.push_back(f); trained_ = false;
}
void AnalogicalFluencyRealizer::observe(const std::string& source, const TextParser& parser, const ParsedDocument* parsed) {
    ppm_.observe(source);
    const auto lower = TextParser::utf8_tolower(source);
    struct Marker { const char* text; PhraseShape shape; DiscourseTone tone; bool english; };
    static const Marker markers[] = {
        {"; portanto, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::TECHNICAL, false},
        {"; por isso, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::DIDACTIC, false},
        {"; assim, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::DIDACTIC, false},
        {"; consequentemente, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::TECHNICAL, false},
        {". Além disso, ", PhraseShape::SEQUENCE, DiscourseTone::TECHNICAL, false},
        {", e ", PhraseShape::SEQUENCE, DiscourseTone::CASUAL, false},
        {"; ", PhraseShape::SEQUENCE, DiscourseTone::TECHNICAL, false},
        {". Também, ", PhraseShape::SEQUENCE, DiscourseTone::CASUAL, false},
        {", mas ", PhraseShape::CONTRAST, DiscourseTone::WARNING, false},
        {"; porém, ", PhraseShape::CONTRAST, DiscourseTone::WARNING, false},
        {"; contudo, ", PhraseShape::CONTRAST, DiscourseTone::WARNING, false},
        {"; entretanto, ", PhraseShape::CONTRAST, DiscourseTone::WARNING, false},
        {"; therefore, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::TECHNICAL, true},
        {"; as a result, ", PhraseShape::CAUSE_SEQUENCE, DiscourseTone::DIDACTIC, true},
        {". Furthermore, ", PhraseShape::SEQUENCE, DiscourseTone::TECHNICAL, true},
        {". Also, ", PhraseShape::SEQUENCE, DiscourseTone::CASUAL, true},
        {", but ", PhraseShape::CONTRAST, DiscourseTone::WARNING, true},
        {"; however, ", PhraseShape::CONTRAST, DiscourseTone::WARNING, true}};
    for (const auto& marker : markers) {
        auto needle = TextParser::utf8_tolower(marker.text);
        // Books often place a causal marker at a sentence boundary rather than a semicolon.
        auto bare = needle.size() > 3 ? needle.substr(2) : needle;
        if (lower.find(needle) == std::string::npos && lower.find(bare) == std::string::npos) continue;
        PhraseFragment f; f.shape = marker.shape; f.tone = marker.tone; f.english = marker.english;
        field(f.connective, sizeof(f.connective), marker.text);
        field(f.clause_pattern, sizeof(f.clause_pattern), "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]"); add_fragment(f);
    }
    const auto document = parsed ? ParsedDocument{} : parser.parse_text(source);
    const auto& triples = parsed ? parsed->triples : document.triples;
    for (const auto& t : triples) {
        if (t.subject.empty() || t.object.empty() || t.relation.size() >= 64) continue;
        PhraseFragment f; f.shape = PhraseShape::CLAUSE;
        f.relation = static_cast<uint8_t>(t.type);
        // Typed clause templates; the observed predicate is kept only for the
        // generic clause so it can be matched against the semantic role.
        if (t.type == RelationType::TAXONOMIC) {
            field(f.clause_pattern, sizeof(f.clause_pattern), "[SUBJ:ENT] é [OBJ:ENT]");
        } else if (t.type == RelationType::PROPERTY) {
            field(f.clause_pattern, sizeof(f.clause_pattern), "[SUBJ:ENT] é [OBJ:ENT]");
        } else {
            field(f.predicate, sizeof(f.predicate), t.relation);
            field(f.clause_pattern, sizeof(f.clause_pattern), "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]");
        }
        add_fragment(f);
    }
}
void AnalogicalFluencyRealizer::rebuild_index(bool gpu) {
    index_ = KanervaSDM(index_config());
    // Each shape/tone bucket stores a bundle of fragment IDs. Retrieval is gated
    // by grammar compatibility; SDM similarity is only an ordering hint.
    for (unsigned shape = 0; shape < 3; ++shape) for (unsigned tone = 0; tone < 4; ++tone) for (bool en : {false,true}) {
        std::vector<HyperVector> values;
        for (const auto& f : fragments()) if (unsigned(f.shape) == shape && unsigned(f.tone) == tone && bool(f.english) == en) values.push_back(fragment_vector(f.id));
        if (!values.empty()) index_.write(shape_vector(PhraseShape(shape), DiscourseTone(tone), en, shape == unsigned(PhraseShape::CAUSE_SEQUENCE) ? DialogIntent::EXPLAIN_CAUSE : DialogIntent::ANSWER_FACT), HyperVector::bundle(values), gpu);
    }
}
void AnalogicalFluencyRealizer::train(bool gpu) {
    TsetlinConfig cfg; cfg.num_features = 16; cfg.num_classes = 4; cfg.clauses_per_class = 24; cfg.seed = 93017;
    TsetlinMachine tm(cfg);
    std::vector<std::vector<uint64_t>> x; std::vector<size_t> y;
    for (unsigned tone = 0; tone < 4; ++tone) for (unsigned mask = 0; mask < 8; ++mask) {
        x.push_back({style_features(DiscourseTone(tone), mask & 1, mask & 2) | (uint64_t(mask & 4) << 5)}); y.push_back(tone);
    }
    // Observed productions contribute weak style supervision in addition to the
    // explicit feature calibration. No semantic truth labels are inferred here.
    for (const auto& f : fragments()) for (unsigned i = 0; i < std::min<uint32_t>(8, f.frequency); ++i) {
        x.push_back({style_features(f.tone, f.shape != PhraseShape::CLAUSE, false)}); y.push_back(unsigned(f.tone));
    }
    tm.fit(x, y, 40, gpu); style_ = tm.export_sparse(); rebuild_index(gpu); trained_ = true;
}
DiscourseTone AnalogicalFluencyRealizer::target_tone(const SemanticPlan& p) const {
    if (p.tone == "technical" || p.tone == "tecnico") return DiscourseTone::TECHNICAL;
    if (p.tone == "didactic" || p.tone == "didatico") return DiscourseTone::DIDACTIC;
    if (p.tone == "warning" || p.tone == "alerta") return DiscourseTone::WARNING;
    if (p.tone == "casual") return DiscourseTone::CASUAL;
    if (!p.paths.empty()) for (const auto& step : p.paths.front().steps)
        if (step.type == RelationType::INHIBITS || step.negated) return DiscourseTone::WARNING;
    if (p.intent == DialogIntent::EXPLAIN_CAUSE) return DiscourseTone::DIDACTIC;
    if (p.intent == DialogIntent::SMALLTALK || p.intent == DialogIntent::GREET) return DiscourseTone::CASUAL;
    return DiscourseTone::TECHNICAL;
}
bool AnalogicalFluencyRealizer::preserves_plan(const FluencyCandidate& c, const SemanticPlan& p) {
    if (p.paths.empty()) return c.realized_steps.empty();
    const auto& s = p.paths.front().steps;
    if (s.size() != c.realized_steps.size()) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        const auto& a = s[i]; const auto& b = c.realized_steps[i];
        if (a.src != b.src || a.rel != b.rel || a.dst != b.dst || a.type != b.type || a.negated != b.negated || a.tense != b.tense || a.modality != b.modality ||
            a.src_gender != b.src_gender || a.src_number != b.src_number || a.dst_gender != b.dst_gender || a.dst_number != b.dst_number) return false;
    }
    // Verify the printed yield as well as the semantic nodes: every clause must
    // be exactly the morphological realization of its declared clause template,
    // joined only by the declared discourse production. Metadata alone cannot
    // certify coverage.
    const auto text = TextParser::utf8_tolower(c.text);
    size_t position = 0;
    MorphologicalRealizer morphology;
    for (size_t i = 0; i < s.size(); ++i) {
        std::vector<std::string> subjects{subject_phrase(s[i], p.language)};
        if (i && s[i-1].src == s[i].src && s[i-1].dst != s[i].src && s[i-1].dst_gender != s[i].src_gender)
            subjects.push_back(p.language == "en" ? "it" : s[i].src_number == GramNumber::PLUR ?
                (s[i].src_gender == GramGender::FEM ? "elas" : "eles") : (s[i].src_gender == GramGender::FEM ? "ela" : "ele"));
        std::string pattern = i < c.clause_patterns.size() ? c.clause_patterns[i]
                                                            : "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]";
        if (pattern.empty()) pattern = "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]";
        std::string join = i < c.joiners.size() ? c.joiners[i] : std::string{};
        bool matched = false;
        for (const auto& subject : subjects) {
            auto clause = TextParser::utf8_tolower(morphology.realize_clause(s[i], subject, p.language, pattern));
            if (i && (join.starts_with(';') || (join.size() > 2 && join != ". ")))
                clause = lower_initial(clause);
            const auto expected = join + clause;
            if (text.compare(position, expected.size(), expected) != 0) continue;
            position += expected.size(); matched = true; break;
        }
        if (!matched) return false;
    }
    return text.substr(position) == ".";
}
FluencyDerivation AnalogicalFluencyRealizer::derive(const SemanticPlan& plan) const {
    FluencyDerivation result;
    MorphologicalRealizer morphology;
    if (!trained_ || plan.resource || plan.quoted_statement || !plan.conditional_conclusions.empty() || plan.paths.empty() || plan.paths.front().steps.empty() ||
        plan.intent == DialogIntent::AGREE || plan.intent == DialogIntent::GREET || plan.intent == DialogIntent::ASK_CLARIFY) {
        FluencyCandidate c; c.text = morphology.realize(plan); result.candidates.push_back(std::move(c)); return result;
    }
    const auto& steps = plan.paths.front().steps;
    const auto tone = target_tone(plan);
    auto query = shape_vector(PhraseShape::SEQUENCE, tone, plan.language == "en", plan.intent);
    auto recalled = index_.read(query); result.active_slots = index_.activate(query);
    const bool english = plan.language == "en";
    // A production is licensed by the graph relation and, when lexicalized, by
    // the exact predicate lemma. relation == 255 is a generic typed template;
    // relation == 0 with an empty predicate is the legacy generic template.
    auto relation_ok = [](const PhraseFragment& f, const CausalStep& step) {
        if (f.predicate[0] &&
            TextParser::lemmatize_simple(f.predicate) != TextParser::lemmatize_simple(step.rel)) return false;
        if (f.relation == 255) return true;
        if (f.relation == 0 && !f.predicate[0]) return true;
        return f.relation == static_cast<uint8_t>(step.type);
    };
    std::vector<const PhraseFragment*> productions, clauses;
    for (const auto& f : fragments()) {
        if (bool(f.english) != english) continue;
        if (f.shape == PhraseShape::CLAUSE) clauses.push_back(&f);
        else productions.push_back(&f);
    }
    std::stable_sort(productions.begin(), productions.end(), [&](auto a, auto b) {
        return recalled.hamming(fragment_vector(a->id)) < recalled.hamming(fragment_vector(b->id));
    });
    for (auto f : productions) result.retrieved_fragments.push_back(f->id);
    struct Partial {
        std::string text;
        std::vector<uint32_t> ids, clause_ids;
        std::vector<std::string> clause_patterns, joiners;
        std::vector<CausalStep> covered;
        DiscourseTone tone;
        bool pronoun = false;
        int64_t ppm_cost = 0;
    };
    std::vector<Partial> beam;
    // First clause is itself a DOP subtree, optionally lexicalized to the verb.
    for (const auto* f : clauses) {
        if (!relation_ok(*f, steps.front())) continue;
        auto first = morphology.realize_clause(steps.front(), subject_phrase(steps.front(), plan.language),
                                               plan.language, f->clause_pattern);
        beam.push_back({first, {f->id}, {f->id}, {f->clause_pattern}, {""}, {steps.front()}, f->tone});
    }
    if (beam.empty()) beam.push_back({morphology.realize_clause(steps.front(), subject_phrase(steps.front(), plan.language), plan.language),
                                      {}, {}, {}, {}, {steps.front()}, tone});
    for (size_t hop = 1; hop < steps.size(); ++hop) {
        std::vector<Partial> next;
        const auto& step = steps[hop]; const auto& previous = steps[hop-1];
        for (const auto& partial : beam) for (const auto* production : productions) {
            if (production->shape == PhraseShape::CAUSE_SEQUENCE &&
                (previous.dst != step.src || previous.type != RelationType::CAUSAL || step.type != RelationType::CAUSAL ||
                 previous.negated || step.negated || !previous.modality.empty() || !step.modality.empty())) continue;
            if (production->shape == PhraseShape::CONTRAST &&
                !(step.negated || step.type == RelationType::INHIBITS || previous.dst != step.src ||
                  previous.type != step.type || previous.negated))
                continue;
            for (const auto* clause_template : clauses) {
                if (!relation_ok(*clause_template, step)) continue;
                for (unsigned variant = 0; variant < 2; ++variant) {
                    const bool pronoun = variant && previous.src == step.src && previous.dst != step.src && previous.dst_gender != step.src_gender;
                    if (variant && !pronoun) continue;
                    std::string subject = subject_phrase(step, plan.language);
                    if (pronoun) subject = plan.language == "en" ? "it" : step.src_number == GramNumber::PLUR ?
                        (step.src_gender == GramGender::FEM ? "elas" : "eles") : (step.src_gender == GramGender::FEM ? "ela" : "ele");
                    auto clause = morphology.realize_clause(step, subject, plan.language, clause_template->clause_pattern);
                    auto connective = std::string(production->connective);
                    if (connective.starts_with(';') || (connective.size() > 2 && connective != ". ")) clause = lower_initial(clause);
                    auto p = partial; p.text += connective + clause;
                    p.ids.push_back(production->id); p.clause_ids.push_back(clause_template->id);
                    p.clause_patterns.push_back(clause_template->clause_pattern);
                    p.joiners.push_back(connective);
                    p.covered.push_back(step);
                    p.tone = production->tone; p.pronoun |= pronoun; p.ppm_cost = ppm_.cost_q16(p.text); next.push_back(std::move(p));
                }
            }
        }
        std::stable_sort(next.begin(), next.end(), [&](const auto& a, const auto& b) { return a.ppm_cost < b.ppm_cost; });
        std::set<std::string> unique; beam.clear();
        for (auto& candidate : next) if (unique.insert(candidate.text).second) { beam.push_back(std::move(candidate)); if (beam.size() == 16) break; }
    }
    for (const auto& partial : beam) {
        FluencyCandidate c; c.text = capitalize(partial.text) + "."; c.fragment_ids = partial.ids;
        c.clause_ids = partial.clause_ids; c.clause_patterns = partial.clause_patterns;
        c.joiners = partial.joiners; c.realized_steps = partial.covered;
        if (!preserves_plan(c, plan)) continue;
        c.ppm_cost_q16 = ppm_.cost_q16(c.text);
        const uint64_t features = style_features(partial.tone, c.text.size() > 100, partial.pronoun);
        const auto votes = style_.score_classes(&features);
        c.style_vote = votes.empty() ? 0 : votes[unsigned(tone)];
                size_t repeats = 0;
        for (size_t i = 2; i < c.fragment_ids.size(); ++i) if (c.fragment_ids[i] == c.fragment_ids[i-1]) ++repeats;
        c.rank_q16 = c.ppm_cost_q16 - int64_t(c.style_vote) * 2048 + int64_t(repeats) * 8192;
        result.candidates.push_back(std::move(c));
    }
    if (result.candidates.empty()) { FluencyCandidate c; c.text = morphology.realize(plan); result.candidates.push_back(std::move(c)); }
    // Planned alternative route: keep the verified main derivation and append a
    // graph-grounded secondary clause. It is realized by the morphological
    // engine, never invented by the statistical ranker.
    if (!plan.alternatives.empty() && !plan.alternatives.front().steps.empty()) {
        SemanticPlan secondary;
        secondary.intent = plan.intent;
        secondary.language = plan.language;
        secondary.paths.push_back(plan.alternatives.front());
        auto alternative = morphology.realize(secondary);
        if (!alternative.empty()) {
            if (static_cast<unsigned char>(alternative[0]) < 128)
                alternative[0] = char(std::tolower(static_cast<unsigned char>(alternative[0])));
            const bool contrast = !plan.rhetorical_roles.empty() &&
                plan.rhetorical_roles.size() > 1 && plan.rhetorical_roles[1] == 4;
            const std::string connector = plan.language == "en"
                ? (contrast ? " However, " : " Alternatively, ")
                : (contrast ? " Porém, " : " Alternativamente, ");
            for (auto& candidate : result.candidates) {
                if (candidate.text.find(alternative) != std::string::npos) continue;
                candidate.text += connector + alternative;
                candidate.ppm_cost_q16 = ppm_.cost_q16(candidate.text);
                size_t repeats = 0;
                for (size_t i = 2; i < candidate.fragment_ids.size(); ++i)
                    if (candidate.fragment_ids[i] == candidate.fragment_ids[i - 1]) ++repeats;
                candidate.rank_q16 = candidate.ppm_cost_q16 - int64_t(candidate.style_vote) * 2048 +
                                     int64_t(repeats) * 8192;
            }
        }
    }
    for (size_t i = 1; i < result.candidates.size(); ++i) if (result.candidates[i].rank_q16 < result.candidates[result.selected].rank_q16) result.selected = i;
    return result;
}
std::string AnalogicalFluencyRealizer::realize(const SemanticPlan& plan) const {
    auto result = derive(plan); return result.candidates[result.selected].text;
}
bool AnalogicalFluencyRealizer::save(const std::string& path) const {
    const auto records = ppm_.records(); const auto fs = fragments();
    const auto temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary); if (!out) return false;
    ArchiveHeader header{0x444F5031, 1, uint32_t(fs.size()), uint32_t(records.size()), uint32_t(style_.clauses.size()), uint32_t(trained_)};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(fs.data()), fs.size_bytes());
    out.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(PPMRecord));
    for (const auto& clause : style_.clauses) {
        DiskClause d{}; d.sign = clause.sign;
        for (auto f : clause.pos_literals) d.positive |= uint16_t(1U << f);
        for (auto f : clause.neg_literals) d.negative |= uint16_t(1U << f);
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));
    }
    out.close(); if (!out) return false;
    std::error_code ec; std::filesystem::rename(temporary, path, ec); return !ec;
}
bool AnalogicalFluencyRealizer::load(const std::string& path) {
    auto m = std::make_shared<Mapping>(); m->fd = ::open(path.c_str(), O_RDONLY); if (m->fd < 0) return false;
    const auto size = ::lseek(m->fd, 0, SEEK_END); if (size < off_t(sizeof(ArchiveHeader))) return false;
    m->size = size; m->data = ::mmap(nullptr, m->size, PROT_READ, MAP_PRIVATE, m->fd, 0); if (m->data == MAP_FAILED) return false;
    const auto* h = static_cast<const ArchiveHeader*>(m->data);
    if (h->magic != 0x444F5031 || h->version != 1 || h->fragments > MAX_FRAGMENTS || h->records > MAX_ENTRIES || (h->clauses && h->clauses != 96) || h->flags > 1) return false;
    const size_t required = sizeof(*h) + size_t(h->fragments) * sizeof(PhraseFragment) + size_t(h->records) * sizeof(PPMRecord) + size_t(h->clauses) * sizeof(DiskClause);
    if (required != m->size || (h->flags && h->clauses != 96)) return false;
    auto* ptr = static_cast<const char*>(m->data) + sizeof(*h);
    auto fs = std::span(reinterpret_cast<const PhraseFragment*>(ptr), h->fragments); ptr += fs.size_bytes();
    auto rs = std::span(reinterpret_cast<const PPMRecord*>(ptr), h->records); ptr += rs.size_bytes();
    for (size_t i = 0; i < fs.size(); ++i) if (fs[i].id != i || unsigned(fs[i].shape) > 3 || unsigned(fs[i].tone) > 3 || fs[i].english > 1 ||
        (fs[i].relation != 255 && fs[i].relation > 4) ||
        !std::memchr(fs[i].connective, 0, sizeof(fs[i].connective)) || !std::memchr(fs[i].clause_pattern, 0, sizeof(fs[i].clause_pattern)) || !std::memchr(fs[i].predicate, 0, sizeof(fs[i].predicate))) return false;
    for (size_t i = 0; i < rs.size(); ++i) if (rs[i].order > 5 || !rs[i].count || (i && !less_record(rs[i-1], rs[i]))) return false;
    SparseTsetlinModel style; style.num_features = 16; style.num_classes = 4; style.clauses_per_class = 24;
    for (uint32_t i = 0; i < h->clauses; ++i) {
        DiskClause d; std::memcpy(&d, ptr + i * sizeof(d), sizeof(d)); if (d.sign != 1 && d.sign != -1) return false;
        SparseClause c; c.sign = d.sign;
        for (unsigned f = 0; f < 16; ++f) { if (d.positive & (1U << f)) c.pos_literals.push_back(f); if (d.negative & (1U << f)) c.neg_literals.push_back(f); }
        style.clauses.push_back(std::move(c));
    }
    // Release old views only after every byte has passed validation.
    ppm_.attach({}); mapped_fragments_ = {}; mapping_ = std::move(m); owned_.clear();
    mapped_fragments_ = fs; ppm_.attach(rs); style_ = std::move(style); trained_ = h->flags; rebuild_index(); return true;
}
} // namespace cognicore
