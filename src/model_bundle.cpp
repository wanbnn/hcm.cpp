#include "core/model_bundle.hpp"
#include "core/morphological_realizer.hpp"
#include "core/discourse_planner.hpp"
#include "core/assistant.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <sstream>
#include <cstring>
#include <unordered_set>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace cognicore {

ModelBundle::ModelBundle() {
    language_rules.load_default();
    lexicon.load_synonyms(language_rules);
    if (code_patterns.empty()) code_patterns.load_default();
}

namespace {

class BinaryWriter {
public:
    explicit BinaryWriter(const std::string& path) : ofs_(path, std::ios::binary) {}
    bool ok() const { return static_cast<bool>(ofs_); }

    template <typename T>
    void write_pod(const T& val) {
        ofs_.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }

    void write_bytes(const void* data, size_t n) {
        if (n > 0) ofs_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    }

    void write_string(const std::string& s) {
        uint32_t len = static_cast<uint32_t>(s.size());
        write_pod(len);
        write_bytes(s.data(), len);
    }

private:
    std::ofstream ofs_;
};

class MappedReader {
public:
    MappedReader(const std::string& path, bool use_mmap) : use_mmap_(use_mmap) {
        if (use_mmap_) {
            fd_ = ::open(path.c_str(), O_RDONLY);
            if (fd_ < 0) return;
            struct stat st{};
            if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
                ::close(fd_);
                fd_ = -1;
                return;
            }
            size_ = static_cast<size_t>(st.st_size);
            void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (p == MAP_FAILED) {
                ::close(fd_);
                fd_ = -1;
                return;
            }
            data_ = static_cast<const uint8_t*>(p);
            valid_ = true;
        } else {
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (!ifs) return;
            size_ = static_cast<size_t>(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            buf_.resize(size_);
            if (size_ > 0 && !ifs.read(reinterpret_cast<char*>(buf_.data()), static_cast<std::streamsize>(size_))) {
                return;
            }
            data_ = buf_.data();
            valid_ = true;
        }
    }

    ~MappedReader() {
        if (use_mmap_ && data_ != nullptr) {
            ::munmap(const_cast<uint8_t*>(data_), size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    bool valid() const { return valid_; }
    bool eof() const { return offset_ >= size_; }

    template <typename T>
    bool read_pod(T& out) {
        if (offset_ + sizeof(T) > size_) return false;
        std::memcpy(&out, data_ + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool read_bytes(void* dst, size_t n) {
        if (offset_ + n > size_) return false;
        if (n > 0) std::memcpy(dst, data_ + offset_, n);
        offset_ += n;
        return true;
    }

    bool read_string(std::string& out) {
        uint32_t len = 0;
        if (!read_pod(len)) return false;
        if (offset_ + len > size_) return false;
        out.assign(reinterpret_cast<const char*>(data_ + offset_), len);
        offset_ += len;
        return true;
    }

private:
    bool use_mmap_ = true;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t offset_ = 0;
    bool valid_ = false;
    std::vector<uint8_t> buf_;
};

constexpr uint32_t MAGIC_CB = 0x434F4743; // COGC
constexpr uint32_t MAGIC_SDM = 0x434F4753; // COGS
constexpr uint32_t MAGIC_TM = 0x434F4754; // COGT
constexpr uint32_t MAGIC_GR_LEGACY = 0x434F4747;
constexpr uint32_t MAGIC_GR = 0x434F4732; // COGG
constexpr uint32_t MAGIC_RL = 0x434F4752; // COGR

} // namespace

std::vector<uint64_t> ModelBundle::encode_dialogue_features(const std::string& user_text) const {
    const size_t n_feat = has_sparse_tsetlin ? sparse_tsetlin.num_features : tsetlin.config().num_features;
    const size_t n_words = (n_feat + 63) / 64;
    std::vector<uint64_t> feats(n_words, 0ULL);

    auto toks = parser.tokenize(user_text);
    HyperVector hv = codebook.encode_bag(toks, true);
    const size_t copy_words = std::min<size_t>(VSA_WORDS, n_words);
    for (size_t w = 0; w < copy_words; ++w) {
        feats[w] = hv.words[w];
    }

    // Add 64 context/graph state bits if num_features > VSA_BITS
    if (n_feat > VSA_BITS && n_words > VSA_WORDS) {
        uint64_t ctx_bits = 0;
        if (user_text.find('?') != std::string::npos) ctx_bits |= (1ULL << 0);
        for (const auto& t : toks) {
            if (t == "porque" || t == "causa" || t == "why") ctx_bits |= (1ULL << 1);
            if (t == "o" || t == "que" || t == "qual" || t == "what") ctx_bits |= (1ULL << 2);
            if (t == "compare" || t == "diferenca" || t == "diferença" || t == "igual") ctx_bits |= (1ULL << 3);
            if (t == "se" || t == "hipotese" || t == "hipótese" || t == "if") ctx_bits |= (1ULL << 4);
            if (t == "ola" || t == "olá" || t == "oi" || t == "hello" || t == "hi") ctx_bits |= (1ULL << 5);
            if (graph.find_node(t) != UINT32_MAX) ctx_bits |= (1ULL << 6);
        }
        const auto lower = TextParser::utf8_tolower(user_text);
        auto has = [&](std::initializer_list<const char*> words) {
            for (const auto* word : words) if (lower.find(word) != std::string::npos) return true;
            return false;
        };
        if (lower.starts_with("bom dia") || lower.starts_with("boa tarde") || lower.starts_with("boa noite")) ctx_bits |= (1ULL << 5);
        if (has({"por que", "por quê"})) ctx_bits |= (1ULL << 1);
        if (has({"não entendi", "nao entendi", "explicar melhor", "esclareça", "esclareca", "de outro jeito", "não consegui entender"})) ctx_bits |= (1ULL << 12);
        if (has({"concordo", "faz sentido", "exatamente"}) || lower == "sim" || lower.starts_with("sim,")) ctx_bits |= (1ULL << 13);
        if (has({"obrigad", "valeu", "agradeç", "agradec", "thank"})) ctx_bits |= (1ULL << 7);
        if (has({"tchau", "até logo", "até mais", "desped", "goodbye"})) ctx_bits |= (1ULL << 8);
        if (has({"seu nome", "quem é você", "quem voce", "quem você", "se apresente", "your name", "who are you"})) ctx_bits |= (1ULL << 9);
        if (has({"ajud", "pode fazer", "consegue fazer", "help"})) ctx_bits |= (1ULL << 10);
        if (has({"tudo bem", "como você está", "como voce esta", "como vai", "conversar", "bater papo"})) ctx_bits |= (1ULL << 11);
        feats[VSA_WORDS] = ctx_bits;
    }
    return feats;
}

DialogIntent ModelBundle::classify_dialogue(const std::string& user_text) const {
    const auto features = encode_dialogue_features(user_text);
    const uint64_t specific = ((1ULL << 14) - 1) & ~((1ULL << 0) | (1ULL << 2) | (1ULL << 6));
    if (has_context_policy && features.size() > VSA_WORDS && (features[VSA_WORDS] & specific))
        return static_cast<DialogIntent>(context_policy.predict(&features[VSA_WORDS]));
    const size_t predicted = has_sparse_tsetlin ? sparse_tsetlin.predict(features.data()) : tsetlin.predict(features.data());
    return static_cast<DialogIntent>(predicted);
}

ModelBundle::InferenceTrace ModelBundle::infer_and_respond(const std::string& user_text, bool online_learn, const ISurfaceRealizer* surface) {
    InferenceTrace trace;
    auto tick = std::chrono::steady_clock::now();
    auto stage = [&](const std::string& name) {
        const auto now = std::chrono::steady_clock::now();
        trace.stage_ms.emplace_back(name, std::chrono::duration<double, std::milli>(now - tick).count());
        tick = now;
    };
    auto all_toks = parser.tokenize(user_text);
    auto content_toks = parser.extract_content_tokens(user_text);
    const bool english_input = std::any_of(all_toks.begin(), all_toks.end(), [](const std::string& token) {
        return token == "why" || token == "what" || token == "how" || token == "the" ||
               token == "hello" || token == "is" || token == "does" || token == "and" || token == "the";
    });

    stage("PARSER");
    const auto lower_input = TextParser::utf8_tolower(user_text);
    const bool hypothetical = language_rules.is_hypothetical(lower_input);
    std::string referent;
    // Contextual memory resolves pronouns/deictics using salience, recency and
    // grammatical agreement. Explicit mentions are already grounded by the
    // topic-activation stage and do not need an anaphora substitution.
    if (auto resolution = context_memory.resolve(user_text, parser)) {
        std::ostringstream evidence;
        evidence << "superfície '" << resolution->surface << "' -> "
                 << (resolution->referent.empty() ? "[ambígua]" : resolution->referent)
                 << " | " << resolution->reason << " | score=" << resolution->score;
        if (resolution->candidates.size() > 1) {
            evidence << " | candidatos:";
            for (const auto& candidate : resolution->candidates) evidence << " " << candidate;
        }
        trace.context_evidence = evidence.str();
        if (!resolution->ambiguous && resolution->surface != resolution->referent)
            referent = resolution->referent;
    }
    // Legacy fallback: last unambiguous episodic fact for a deictic turn.
    // This keeps older session files usable when context.bin is absent.
    const bool deictic = std::find(all_toks.begin(), all_toks.end(), "lá") != all_toks.end() ||
        std::find(all_toks.begin(), all_toks.end(), "ele") != all_toks.end() ||
        std::find(all_toks.begin(), all_toks.end(), "ela") != all_toks.end() ||
        lower_input.find("nesse servidor") != std::string::npos;
    if (referent.empty() && deictic && !episodic_memory.facts().empty()) {
        const auto& last = episodic_memory.facts().back();
        // More than one entity in the last assertion is ambiguous.
        bool ambiguous = false;
        for (auto it = episodic_memory.facts().rbegin(); it != episodic_memory.facts().rend() && it->source == last.source; ++it)
            if (it->entity != last.entity) ambiguous = true;
        if (!ambiguous) referent = last.entity;
    }
    // Online one-shot learning if the user states a declarative fact, rule or
    // conditional. Rules are stored separately from asserted graph edges.
    const bool is_question = (user_text.find('?') != std::string::npos);
    bool learned_fact = false;
    const auto parsed_conditionals = parser.parse_text(user_text);
    const bool is_rule = hypothetical || !parsed_conditionals.conditionals.empty();
    if (online_learn && !is_question && is_rule) {
        auto rule_property = [](const ParsedConditional& conditional) {
            std::string property = "conditional_implies/";
            property += TextParser::condition_kind_name(conditional.kind);
            property += "/" + (conditional.quantifier.empty() ? std::string("-") : conditional.quantifier);
            property += conditional.negated ? "/1" : "/0";
            return property;
        };
        if (!parsed_conditionals.conditionals.empty()) {
            for (const auto& conditional : parsed_conditionals.conditionals) {
                episodic_memory.remember(conditional.antecedent, rule_property(conditional),
                                         conditional.consequent, user_text);
                learned_fact = true;
            }
        } else {
            auto marker = lower_input.find(" então ");
            size_t width = std::string(" então ").size();
            if (marker == std::string::npos) { marker = lower_input.find(" then "); width = 6; }
            if (marker != std::string::npos) {
                auto antecedent = lower_input.substr(3, marker - 3);
                auto consequent = lower_input.substr(marker + width);
                while (!consequent.empty() && (consequent.back() == '.' || consequent.back() == ' ')) consequent.pop_back();
                episodic_memory.remember(antecedent, "conditional_implies", consequent, user_text);
                learned_fact = true;
            }
        }
    }
    const bool instruction_turn = language_rules.is_instruction_or_correction(lower_input);
    if (online_learn && !is_question && !is_rule && !instruction_turn) {
        auto parsed = parser.parse_text(user_text);
        if (!parsed.triples.empty()) {
            for (const auto& tr : parsed.triples) {
                graph.add_fact(tr.subject, tr.relation, tr.object, tr.type, tr.confidence, true,
                               tr.negated, tr.tense, tr.modality, tr.subj_gender, tr.subj_number,
                               tr.obj_gender, tr.obj_number);
                codebook.get_or_add(tr.subject);
                codebook.get_or_add(tr.object);
                HyperVector addr = HyperVector::from_token(tr.subject);
                HyperVector data = HyperVector::from_token(tr.object);
                sdm.write(addr, data);
                episodic_memory.remember(tr.subject, tr.relation, tr.object, user_text);
                learned_fact = true;
            }
            for (auto& pair : realizer_document_pairs(parsed)) realizer_pairs.push_back(std::move(pair));
            for (const auto& cf : parsed.frames) {
                realizer.add_frame(cf);
            }
        }
    }

    // Explicit capacity assertions are episodic properties, not inferred machine specifications.
    const bool negative_assertion = std::find(all_toks.begin(), all_toks.end(), "não") != all_toks.end() ||
        std::find(all_toks.begin(), all_toks.end(), "nao") != all_toks.end() ||
        std::find(all_toks.begin(), all_toks.end(), "not") != all_toks.end();
    if (online_learn && !is_question && !hypothetical && !negative_assertion &&
        lower_input.find("menos") == std::string::npos && lower_input.find("mais de") == std::string::npos &&
        lower_input.find("cerca") == std::string::npos && lower_input.find("até") == std::string::npos &&
        (lower_input.find("ram") != std::string::npos || lower_input.find("memória") != std::string::npos)) {
        if (const auto quantity = parse_memory_quantity(lower_input)) {
            auto end = lower_input.find(" só tem ");
            if (end == std::string::npos) end = lower_input.find(" tem ");
            if (end != std::string::npos) {
                auto entity = lower_input.substr(0, end);
                if (entity.starts_with("meu ")) entity.erase(0, 4);
                if (entity.starts_with("minha ")) entity.erase(0, 6);
                episodic_memory.remember(entity, "ram_capacity", quantity->surface, user_text);
                context_memory.observe_entity(entity, TextParser::infer_gender(entity), TextParser::infer_number(entity));
                learned_fact = true;
            }
        }
    }
    if (!referent.empty()) trace.resolved_reference = "recência discursiva -> " + referent;
    for (auto it = episodic_memory.facts().rbegin(); it != episodic_memory.facts().rend(); ++it) {
        if (it->property != "ram_capacity") continue;
        if (referent != it->entity && lower_input.find(it->entity) == std::string::npos) continue;
        auto recalled = episodic_memory.recall(it->entity, it->property);
        if (recalled) trace.recalled_episodes.push_back(*recalled);
        break;
    }
    // 1. VSA + SDM topic activation
    for (const auto& t : content_toks) {
        if (graph.find_node(t) != UINT32_MAX) {
            trace.activated_concepts.push_back(t);
        }
    }
    // Also check multi-word phrases in graph
    for (size_t i = 0; i + 1 < all_toks.size(); ++i) {
        std::string bi = all_toks[i] + " " + all_toks[i + 1];
        if (graph.find_node(bi) != UINT32_MAX) {
            trace.activated_concepts.push_back(bi);
        }
    }

    // Ground partial nominal phrases before associative expansion. An unanchored
    // top-K SDM read always returns something, even for unrelated/OOV questions.
    if (trace.activated_concepts.empty()) {
        for (const auto& node : graph.nodes()) {
            const auto space = node.name.find(' ');
            if (space == std::string::npos) continue;
            const auto head = node.name.substr(0, space);
            if (std::find(content_toks.begin(), content_toks.end(), head) != content_toks.end())
                trace.activated_concepts.push_back(node.name);
        }
    }
    // Concepts grounded directly in the user turn. SDM expansion below is a
    // recall hint; graph reasoning must start from what the user actually said
    // (plus explicit contextual carry-over), never from an unrelated expanded
    // concept that happens to have a high-confidence edge.
    const auto grounded_concepts = trace.activated_concepts;
    // SDM expands grounded concepts; it is not evidence for an arbitrary topic.
    if (!content_toks.empty() && !trace.activated_concepts.empty()) {
        HyperVector q_hv = codebook.encode_bag(content_toks, true);
        trace.active_slots = sdm.activate(q_hv);
        for (auto slot : trace.active_slots) trace.slot_distances.push_back(q_hv.hamming(sdm.addresses()[slot]));
        HyperVector recalled = sdm.read(q_hv);
        auto near = codebook.query_nearest(recalled, 3, true);
        for (const auto& [cand, sim] : near) {
            if (sim > 0.20 && graph.find_node(cand) != UINT32_MAX) {
                if (std::find(trace.activated_concepts.begin(), trace.activated_concepts.end(), cand) ==
                    trace.activated_concepts.end()) {
                    trace.activated_concepts.push_back(cand);
                }
            }
        }
    }

    if (!referent.empty() && graph.find_node(referent) != UINT32_MAX)
        trace.activated_concepts.push_back(referent);
    // Topic carry-over: a follow-up that introduces no known topic inherits
    // the last salient topic, but only when an explicit follow-up cue is present.
    const auto carried = context_memory.carryover_concepts(user_text, parser, graph);
    for (const auto& topic : carried) {
        if (graph.find_node(topic) == UINT32_MAX) continue;
        if (std::find(trace.activated_concepts.begin(), trace.activated_concepts.end(), topic) ==
            trace.activated_concepts.end()) {
            trace.activated_concepts.push_back(topic);
            trace.carried_concepts.push_back(topic);
        }
    }
    stage("VSA_SDM_EPISODIOS");
    // 2. Tsetlin Machine dialog intent
    if (learned_fact) {
        trace.intent = DialogIntent::AGREE;
    } else {
        trace.intent = classify_dialogue(user_text);
    }

    const auto features = encode_dialogue_features(user_text);
    const uint64_t specific = ((1ULL << 14) - 1) & ~((1ULL << 0) | (1ULL << 2) | (1ULL << 6));
    const bool context = has_context_policy && features.size() > VSA_WORDS && (features[VSA_WORDS] & specific);
    const auto policy = context ? context_policy : has_sparse_tsetlin ? sparse_tsetlin : tsetlin.export_sparse();
    const auto* bits = context ? &features[VSA_WORDS] : features.data();
    trace.policy_source = context ? "context_policy" : has_sparse_tsetlin ? "sparse_tsetlin" : "tsetlin_export";
    trace.class_votes = policy.score_classes(bits);
    for (size_t i = 0; i < policy.clauses.size(); ++i) {
        const auto& clause = policy.clauses[i];
        bool active = true;
        for (auto f : clause.pos_literals) if (!(bits[f / 64] & (1ULL << (f % 64)))) active = false;
        for (auto f : clause.neg_literals) if (bits[f / 64] & (1ULL << (f % 64))) active = false;
        if (active) { trace.fired_clauses.push_back(i); trace.fired_literals.push_back(clause); }
    }
    if (learned_fact) trace.decision_override = "AGREE: fato aprendido neste turno";
    stage("TSETLIN");
    // 3. Causal Graph multi-hop reasoning
    std::vector<std::string> reasoning_seeds = grounded_concepts;
    for (const auto& topic : trace.carried_concepts)
        if (std::find(reasoning_seeds.begin(), reasoning_seeds.end(), topic) == reasoning_seeds.end())
            reasoning_seeds.push_back(topic);
    if (!referent.empty() && std::find(reasoning_seeds.begin(), reasoning_seeds.end(), referent) == reasoning_seeds.end())
        reasoning_seeds.push_back(referent);
    if (learned_fact) mapped_graph.reset();
    if (!reasoning_seeds.empty()) {
        std::vector<CausalPath> candidates;
        if (mapped_graph && mapped_graph->valid())
            candidates = mapped_graph->find_paths(reasoning_seeds, 3, 5);
        if (candidates.empty())
            candidates = graph.find_paths(reasoning_seeds, 3, 5, &codebook);
        // Explicit relation questions ("O que derrete o gelo?", "O que o calor
        // aquece?") are directional. Ask the graph for the requested relation
        // instead of relying only on generic path ranking.
        std::vector<CausalPath> explicit_paths;
        auto query_direct = [&](const std::string& name) {
            if (mapped_graph && mapped_graph->valid()) return mapped_graph->query_direct(name, 16);
            return graph.query_direct(name, 16);
        };
        for (size_t ti = 0; ti < all_toks.size(); ++ti) {
            const auto lemma = TextParser::lemmatize_simple(all_toks[ti]);
            const bool known_relation = std::find(graph.relations().begin(), graph.relations().end(), lemma) != graph.relations().end();
            if (!known_relation) continue;
            for (const auto& seed : reasoning_seeds) {
                const auto seed_token = parser.tokenize(seed);
                if (seed_token.empty()) continue;
                const auto seed_at = std::find(all_toks.begin(), all_toks.end(), seed_token.front());
                if (seed_at == all_toks.end()) continue;
                const size_t seed_position = static_cast<size_t>(seed_at - all_toks.begin());
                for (const auto& step : query_direct(seed)) {
                    if (TextParser::lemmatize_simple(step.rel) != lemma) continue;
                    const bool outgoing = seed_position < ti && step.src == seed;
                    const bool incoming = seed_position > ti && step.dst == seed;
                    if (!outgoing && !incoming) continue;
                    CausalPath path;
                    path.steps.push_back(step);
                    path.score = 50.0f + step.confidence;
                    explicit_paths.push_back(std::move(path));
                }
            }
        }
        if (!explicit_paths.empty()) {
            candidates.insert(candidates.begin(), explicit_paths.begin(), explicit_paths.end());
        }
        const auto discourse = DiscoursePlanner{}.plan(candidates, trace.intent, hypothetical, reasoning_seeds);
        trace.paths = discourse.paths;
        trace.alternatives = discourse.alternatives;
        trace.rhetorical_roles.clear();
        for (const auto role : discourse.roles)
            trace.rhetorical_roles.push_back(static_cast<uint8_t>(role));
        trace.planning_evidence = discourse.rationale;
    }

    for (const auto& path : trace.paths) for (const auto& step : path.steps)
        if (step.edge_id != UINT32_MAX && std::find(trace.edge_ids.begin(), trace.edge_ids.end(), step.edge_id) == trace.edge_ids.end())
            trace.edge_ids.push_back(step.edge_id);
    stage("GRAFO");
    // Conditional rules remain separate from asserted graph edges. Match every
    // content literal, including negative markers, before deriving a consequent.
    std::vector<EpisodicRecall> applied_rules;
    std::map<uint64_t, ConditionKind> applied_rule_kinds;
    for (const auto& rule : episodic_memory.facts()) {
        if (!rule.property.starts_with("conditional_implies")) continue;
        // Property layout: conditional_implies/<KIND>/<QUANT>/<NEGATED>.
        ConditionKind kind = ConditionKind::CONDITIONAL;
        std::string quantifier;
        bool rule_negated = false;
        {
            const auto first = rule.property.find('/');
            if (first != std::string::npos) {
                const auto second = rule.property.find('/', first + 1);
                const auto third = rule.property.find('/', second == std::string::npos ? first + 1 : second + 1);
                const auto kind_text = second == std::string::npos ? std::string{}
                    : rule.property.substr(first + 1, second - first - 1);
                if (kind_text == "TEMPORAL") kind = ConditionKind::TEMPORAL;
                else if (kind_text == "CAUSAL") kind = ConditionKind::CAUSAL;
                else if (kind_text == "CONCESSIVE") kind = ConditionKind::CONCESSIVE;
                else if (kind_text == "CONSECUTIVE") kind = ConditionKind::CONSECUTIVE;
                if (second != std::string::npos) {
                    const auto end = third == std::string::npos ? rule.property.size() : third;
                    quantifier = rule.property.substr(second + 1, end - second - 1);
                    if (quantifier == "-") quantifier.clear();
                }
                if (third != std::string::npos) rule_negated = rule.property.substr(third + 1) == "1";
            }
        }
        const bool applicable_context = (hypothetical && is_question) ||
            ((kind == ConditionKind::CAUSAL || kind == ConditionKind::CONCESSIVE) && is_question);
        if (!applicable_context) continue;
        auto required = parser.extract_content_tokens(rule.entity);
        required.erase(std::remove_if(required.begin(), required.end(), [&](const std::string& token) {
            return std::find(language_rules.quantifier_words.begin(), language_rules.quantifier_words.end(), token) !=
                   language_rules.quantifier_words.end();
        }), required.end());
        for (const auto& token : parser.tokenize(rule.entity))
            if (token == "sem" || token == "não" || token == "nao" || token == "not" ||
                token == "without" || token == "com" || token == "with" || token == "nunca")
                required.push_back(token);
        bool match = !required.empty();
        if (match && (quantifier == "ANY" || quantifier == "SOME")) {
            match = false;
            for (const auto& token : required)
                if (std::find(all_toks.begin(), all_toks.end(), token) != all_toks.end()) { match = true; break; }
        } else {
            for (const auto& token : required)
                if (std::find(all_toks.begin(), all_toks.end(), token) == all_toks.end()) match = false;
        }
        // Do not apply a positive premise to an explicitly negated question.
        const auto premise_tokens = parser.tokenize(rule.entity);
        for (const auto* neg : {"não", "nao", "not"})
            if (std::find(all_toks.begin(), all_toks.end(), neg) != all_toks.end() &&
                std::find(premise_tokens.begin(), premise_tokens.end(), neg) == premise_tokens.end()) match = false;
        if (!match) continue;
        if (auto recalled = episodic_memory.recall(rule.entity, rule.property)) {
            if (std::any_of(applied_rules.begin(), applied_rules.end(), [&](const auto& r) { return r.fact.id == recalled->fact.id; })) continue;
            applied_rule_kinds[recalled->fact.id] = kind;
            applied_rules.push_back(*recalled);
            trace.recalled_episodes.push_back(*recalled);
        }
        (void)rule_negated;
    }
    // 4. Surface Realizer compositional synthesis
    SemanticPlan plan;
    plan.intent = trace.intent;
    plan.paths = trace.paths;
    plan.alternatives = trace.alternatives;
    plan.rhetorical_roles = trace.rhetorical_roles;
    for (const auto& rule : applied_rules) {
        plan.conditional_conclusions.emplace_back(rule.fact.entity, rule.fact.value);
        const auto kind = applied_rule_kinds.find(rule.fact.id);
        plan.conditional_kinds.push_back(kind == applied_rule_kinds.end()
            ? static_cast<uint8_t>(ConditionKind::CONDITIONAL) : static_cast<uint8_t>(kind->second));
        const auto tails = graph.find_paths({rule.fact.value}, 3, 1, &codebook);
        if (applied_rules.size() == 1) plan.paths = tails;
        else plan.paths.clear();
        plan.alternatives.clear();
        plan.rhetorical_roles.clear();
        plan.intent = DialogIntent::EXPLAIN_CAUSE;
    }
    plan.language = english_input ? "en" : "pt";
    plan.query_concepts = trace.activated_concepts.empty() ? content_toks : trace.activated_concepts;
    if (trace.activated_concepts.empty()) {
        const auto lower = TextParser::utf8_tolower(user_text);
        if (auto at = lower.find(" sobre "); at != std::string::npos)
            plan.query_concepts = parser.extract_content_tokens(lower.substr(at + 7));
    }
    if (hypothetical) plan.assumptions.push_back(user_text);
    plan.evidence_edges = trace.edge_ids;
    for (const auto& recalled : trace.recalled_episodes) plan.evidence_episodes.push_back(recalled.fact.id);
    if (!trace.recalled_episodes.empty() && !negative_assertion &&
        (lower_input.find("carregar") != std::string::npos || lower_input.find("alocar") != std::string::npos || lower_input.find("caber") != std::string::npos)) {
        const auto& remembered = trace.recalled_episodes.front().fact;
        if (remembered.property == "ram_capacity") if (const auto capacity = parse_memory_quantity(remembered.value)) {
            ResourceAssessment r;
            r.entity = remembered.entity; r.capacity_bytes = capacity->bytes;
            r.capacity_surface = remembered.value; r.episode_id = remembered.id;
            if (const auto demand = parse_memory_quantity(lower_input)) {
                r.demand_known = true; r.demand_bytes = demand->bytes; r.demand_surface = demand->surface;
                r.result = r.demand_bytes > r.capacity_bytes ? ResourceAssessment::Result::EXCEEDS : ResourceAssessment::Result::WITHIN;
            }
            plan.resource = r;
            // Capacity arithmetic cannot prove swap/OOM or mmap behavior.
            plan.paths.clear(); plan.alternatives.clear(); plan.rhetorical_roles.clear(); plan.evidence_edges.clear();
            plan.conditional_conclusions.clear();
            plan.conditional_kinds.clear();
            plan.evidence_episodes = {remembered.id};
        }
    }
    // Provenance refers to the selected plan, including conditional continuations.
    plan.evidence_edges.clear();
    for (const auto& path : plan.paths) for (const auto& step : path.steps)
        if (step.edge_id != UINT32_MAX && std::find(plan.evidence_edges.begin(), plan.evidence_edges.end(), step.edge_id) == plan.evidence_edges.end())
            plan.evidence_edges.push_back(step.edge_id);
    if (lower_input.starts_with("olá") || lower_input == "oi" || lower_input.starts_with("oi,") || lower_input.starts_with("hello")) {
        plan.intent = DialogIntent::GREET;
        trace.decision_override = "GREET: construção explícita de saudação";
    }
    if (lower_input.find("o que eu disse") != std::string::npos || lower_input.find("what did i say") != std::string::npos) {
        plan = SemanticPlan{};
        plan.quoted_statement = episodes.empty() ? "Ainda não há episódios anteriores nesta sessão." : episodes.back().first;
        trace.decision_override = "consulta literal ao diário; sem inferência associativa";
        for (auto it = episodic_memory.facts().rbegin(); it != episodic_memory.facts().rend(); ++it) {
            if (it->source != *plan.quoted_statement) continue;
            if (const auto recalled = episodic_memory.recall(it->entity, it->property)) {
                trace.recalled_episodes.push_back(*recalled);
                plan.evidence_episodes.push_back(it->id);
                plan.quoted_statement = recalled->fact.source;
                trace.decision_override = "citação de episódio recuperado e verificado na SDM";
            }
            break;
        }
    }
    trace.plan = plan;
    stage("PLANO");
    const auto lexical_plan = LexicalChooser{}.apply(plan, lexicon);
    if (const auto* fluent = dynamic_cast<const AnalogicalFluencyRealizer*>(surface)) {
        const auto derivation = fluent->derive(lexical_plan);
        trace.response = derivation.candidates[derivation.selected].text;
        if (!coherence.empty() && derivation.candidates.size() > 1) {
            std::vector<std::string> texts;
            for (const auto& candidate : derivation.candidates) texts.push_back(candidate.text);
            const auto previous = episodes.empty() ? std::string{} : episodes.back().second;
            const auto coherent = coherence.choose(texts, plan, previous, graph, language_rules);
            for (size_t i = 0; i < texts.size(); ++i)
                if (texts[i] == coherent) { trace.response = coherent; break; }
        }
        std::ostringstream audit;
        audit << "candidatas=" << derivation.candidates.size() << " selecionada=" << derivation.selected << " fragmentos recuperados=";
        for (auto id : derivation.retrieved_fragments) audit << id << ",";
        audit << " slots SDM="; for (auto id : derivation.active_slots) audit << id << ",";
        for (size_t i = 0; i < derivation.candidates.size(); ++i) {
            const auto& c = derivation.candidates[i];
            audit << "\n  #" << i << " PPM_Q16=" << c.ppm_cost_q16 << " voto_estilo=" << c.style_vote << " rank_Q16=" << c.rank_q16 << " fragmentos=";
            for (auto id : c.fragment_ids) audit << id << ",";
        }
        trace.fluency_evidence = audit.str();
    } else trace.response = surface ? surface->realize(lexical_plan) : realizer.realize(lexical_plan);
    stage("REALIZADOR");

    // Persist the short-term discourse state for the next turn.
    context_memory.observe_concepts(trace.activated_concepts);
    context_memory.observe_user_turn(user_text, parser, graph);
    return trace;
}

std::string ModelBundle::InferenceTrace::debug_report() const {
    std::ostringstream out;
    out << "[ROTEADOR TSETLIN] " << router_evidence << "\n";
    for (const auto& [stage, ms] : stage_ms) out << "[" << stage << "] " << ms << " ms\n";
    out << "[VSA] conceitos:";
    for (const auto& c : activated_concepts) out << " [" << c << "]";
    out << "\n[SDM] slots reais (id:distância Hamming):";
    for (size_t i = 0; i < active_slots.size(); ++i) out << " " << active_slots[i] << ":" << slot_distances[i];
    out << "\n[REFERENCIA] " << (resolved_reference.empty() ? "nenhuma resolução" : resolved_reference);
    if (!context_evidence.empty()) out << "\n[CONTEXTO] " << context_evidence;
    if (!carried_concepts.empty()) {
        out << "\n[CONTEXTO] tópico herdado:";
        for (const auto& topic : carried_concepts) out << " [" << topic << "]";
    }
    for (const auto& e : recalled_episodes) {
        out << "\n[EPISODIO #" << e.fact.id << "] " << e.fact.entity << " / " << e.fact.property
            << " = " << e.fact.value << " | similaridade SDM=" << e.similarity << " | slots=";
        for (auto slot : e.active_slots) out << slot << ",";
        out << " | fonte: " << e.fact.source;
    }
    out << "\n[TSETLIN] modelo=" << policy_source << " intenção=" << intent_to_string(intent) << " votos:";
    for (size_t i = 0; i < class_votes.size(); ++i) out << " " << i << "=" << class_votes[i];
    out << "\n[TSETLIN] cláusulas disparadas=" << fired_clauses.size() << " (primeiras 16; índices globais)";
    for (size_t i = 0; i < std::min<size_t>(16, fired_clauses.size()); ++i) {
        out << "\n  #" << fired_clauses[i] << " sinal=" << int(fired_literals[i].sign) << " IF";
        for (auto f : fired_literals[i].pos_literals) out << " x[" << f << "]";
        for (auto f : fired_literals[i].neg_literals) out << " !x[" << f << "]";
        if (fired_literals[i].pos_literals.empty() && fired_literals[i].neg_literals.empty()) out << " TRUE (vazia)";
    }
    if (!decision_override.empty()) out << "\n[POLITICA EXPLICITA] " << decision_override;
    out << "\n[PLANEJAMENTO] " << (planning_evidence.empty() ? "sem caminho" : planning_evidence);
    out << "\n[GRAFO] arestas reais:";
    for (auto id : edge_ids) out << " " << id;
    for (const auto& p : paths) for (const auto& step : p.steps)
        out << "\n  " << step.src << " --" << (step.negated ? "NOT " : "") << step.rel << "--> " << step.dst << " conf=" << step.confidence;
    if (!fluency_evidence.empty()) out << "\n[FLUENCIA DOP/PPM/TSETLIN] " << fluency_evidence;
    if (!code_evidence.empty()) out << "\n[CODIGO] " << code_evidence;
    out << "\n[PLANO ANTES DO REALIZADOR] " << serialize_semantic_plan(plan) << "\n";
    return out.str();
}

bool ModelBundle::save(const std::string& dir_path) const {
    std::error_code ec;
    fs::create_directories(dir_path, ec);
    if (ec) return false;
    if (!language_rules.save(dir_path + "/language_rules.json")) return false;
    if (!lexicon.save(dir_path + "/lexicon.bin")) return false;
    if (!coherence.save(dir_path + "/coherence.bin")) return false;
    if (!code_patterns.save(dir_path + "/code_patterns.jsonl")) return false;

    {
        BinaryWriter w(dir_path + "/episodic_facts.bin");
        if (!w.ok()) return false;
        w.write_pod(uint32_t(0x45504631)); w.write_pod(uint32_t(episodic_memory.facts().size()));
        for (const auto& f : episodic_memory.facts()) {
            w.write_string(f.entity); w.write_string(f.property); w.write_string(f.value); w.write_string(f.source);
        }
    }
    if (!fluent_realizer.save(dir_path + "/fluency.bin")) return false;
    // 1. Codebook
    {
        BinaryWriter w(dir_path + "/codebook.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_CB);
        const uint32_t n = static_cast<uint32_t>(codebook.size());
        w.write_pod(n);
        for (const auto& tok : codebook.tokens()) w.write_string(tok);
        if (n > 0) {
            w.write_bytes(codebook.base_vectors().data(), n * sizeof(HyperVector));
            w.write_bytes(codebook.semantic_vectors().data(), n * sizeof(HyperVector));
        }
    }

    // 2. SDM
    {
        BinaryWriter w(dir_path + "/sdm.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_SDM);
        w.write_pod(sdm.config());
        uint8_t bin = sdm.is_binarized() ? 1 : 0;
        w.write_pod(bin);
        uint64_t wc = sdm.write_count();
        w.write_pod(wc);
        const size_t m = sdm.num_locations();
        w.write_bytes(sdm.addresses().data(), m * sizeof(HyperVector));
        w.write_bytes(sdm.touched_flags().data(), m * sizeof(uint8_t));
        if (bin) {
            w.write_bytes(sdm.binarized_contents().data(), m * sizeof(HyperVector));
        } else {
            w.write_bytes(sdm.counters().data(), m * VSA_BITS * sizeof(int8_t));
        }
    }

    // 3. Tsetlin (save sparse if consolidated, else full)
    {
        BinaryWriter w(dir_path + "/tsetlin.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_TM);
        uint8_t is_sparse = has_sparse_tsetlin ? 1 : 0;
        w.write_pod(is_sparse);
        if (is_sparse) {
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.num_features));
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.num_classes));
            w.write_pod(static_cast<uint32_t>(sparse_tsetlin.clauses_per_class));
            for (const auto& cl : sparse_tsetlin.clauses) {
                w.write_pod(cl.sign);
                uint32_t np = static_cast<uint32_t>(cl.pos_literals.size());
                uint32_t nn = static_cast<uint32_t>(cl.neg_literals.size());
                w.write_pod(np);
                w.write_pod(nn);
                w.write_bytes(cl.pos_literals.data(), np * sizeof(uint16_t));
                w.write_bytes(cl.neg_literals.data(), nn * sizeof(uint16_t));
            }
        } else {
            w.write_pod(tsetlin.config());
            w.write_bytes(tsetlin.ta_states().data(), tsetlin.ta_states().size() * sizeof(int8_t));
            w.write_bytes(tsetlin.clause_signs().data(), tsetlin.clause_signs().size() * sizeof(int8_t));
        }
    }

    {
        BinaryWriter w(dir_path + "/dialogue_context.bin");
        if (!w.ok()) return false;
        w.write_pod(uint32_t(0x43545831)); w.write_pod(uint8_t(has_context_policy));
        if (has_context_policy) {
            w.write_pod(uint32_t(context_policy.num_features)); w.write_pod(uint32_t(context_policy.num_classes));
            w.write_pod(uint32_t(context_policy.clauses_per_class));
            for (const auto& clause : context_policy.clauses) {
                w.write_pod(clause.sign); w.write_pod(uint32_t(clause.pos_literals.size())); w.write_pod(uint32_t(clause.neg_literals.size()));
                w.write_bytes(clause.pos_literals.data(), clause.pos_literals.size() * sizeof(uint16_t));
                w.write_bytes(clause.neg_literals.data(), clause.neg_literals.size() * sizeof(uint16_t));
            }
        }
    }

    // 4. Causal Graph
    {
        BinaryWriter w(dir_path + "/graph.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_GR);
        uint32_t nn = static_cast<uint32_t>(graph.nodes().size());
        uint32_t nr = static_cast<uint32_t>(graph.relations().size());
        uint32_t ne = static_cast<uint32_t>(graph.edges().size());
        w.write_pod(nn);
        w.write_pod(nr);
        w.write_pod(ne);
        for (const auto& node : graph.nodes()) {
            w.write_pod(node.id);
            w.write_pod(node.frequency);
            w.write_pod(static_cast<uint8_t>(node.gender));
            w.write_pod(static_cast<uint8_t>(node.number));
            w.write_string(node.name);
        }
        for (const auto& rel : graph.relations()) {
            w.write_string(rel);
        }
        if (ne > 0) {
            w.write_bytes(graph.edges().data(), ne * sizeof(GraphEdge));
        }
    }
    if (!MappedCausalGraph::write(dir_path + "/graph_mmap.bin", graph)) return false;

    // Optional Phase-2 code index, independent of the legacy model files.
    {
        BinaryWriter w(dir_path + "/code_index.bin");
        if (!w.ok()) return false;
        w.write_pod(uint32_t(0x434F4433));
        auto strings = [&](const std::vector<std::string>& values) {
            w.write_pod(uint32_t(values.size()));
            for (const auto& value : values) w.write_string(value);
        };
        w.write_pod(uint32_t(code_graph.symbols().size()));
        for (const auto& sym : code_graph.symbols()) {
            w.write_string(sym.name); w.write_string(sym.return_type);
            strings(sym.param_types); strings(sym.param_names); strings(sym.includes_or_imports);
            w.write_string(sym.documentation); w.write_string(sym.canonical_ast);
            w.write_string(sym.body_snippet); w.write_string(sym.file_path);
            w.write_pod(uint8_t(sym.lang));
            w.write_pod(uint8_t(sym.is_method)); w.write_pod(uint8_t(sym.is_static));
            w.write_string(sym.owner);
            w.write_pod(sym.ast_vsa);
        }
        w.write_pod(uint32_t(code_graph.edges().size()));
        for (const auto& edge : code_graph.edges()) {
            w.write_pod(edge.src); w.write_pod(edge.dst); w.write_pod(uint8_t(edge.type));
        }
        w.write_pod(uint32_t(code_graph.patterns().size()));
        for (const auto& pattern : code_graph.patterns()) {
            w.write_string(pattern.id_tag); w.write_pod(uint8_t(pattern.lang));
            w.write_string(pattern.template_code); strings(pattern.required_includes); w.write_pod(pattern.pattern_vsa);
        }
        std::ofstream pairs(dir_path + "/realizer_pairs.jsonl");
        if (!pairs) return false;
        for (const auto& pair : realizer_pairs) pairs << pair << '\n';
    }

    {
        BinaryWriter w(dir_path + "/episodes.bin");
        if (!w.ok()) return false;
        w.write_pod(uint32_t(0x45504931)); w.write_pod(uint32_t(episodes.size()));
        for (const auto& [user, assistant] : episodes) { w.write_string(user); w.write_string(assistant); }
    }

    // Short-term discourse state (optional; older models load without it).
    {
        BinaryWriter w(dir_path + "/context.bin");
        if (!w.ok()) return false;
        w.write_pod(uint32_t(0x43545832)); w.write_pod(uint32_t(1));
        w.write_pod(uint64_t(context_memory.turn()));
        w.write_pod(uint32_t(context_memory.entities().size()));
        for (const auto& entity : context_memory.entities()) {
            w.write_string(entity.name);
            w.write_pod(uint8_t(entity.gender)); w.write_pod(uint8_t(entity.number));
            w.write_pod(entity.mentions); w.write_pod(entity.last_turn); w.write_pod(entity.salience);
        }
        w.write_pod(uint32_t(context_memory.last_concepts().size()));
        for (const auto& topic : context_memory.last_concepts()) w.write_string(topic);
    }

    // 5. Realizer
    {
        BinaryWriter w(dir_path + "/realizer.bin");
        if (!w.ok()) return false;
        w.write_pod(MAGIC_RL);
        w.write_pod(realizer.total_tokens());
        uint32_t nf = static_cast<uint32_t>(realizer.frames().size());
        w.write_pod(nf);
        for (const auto& f : realizer.frames()) {
            w.write_pod(static_cast<uint8_t>(f.rel_type));
            w.write_pod(f.frequency);
            w.write_string(f.pattern);
        }
        auto write_map = [&](const std::unordered_map<std::string, uint32_t>& m) {
            uint32_t sz = static_cast<uint32_t>(m.size());
            w.write_pod(sz);
            for (const auto& [k, v] : m) {
                w.write_string(k);
                w.write_pod(v);
            }
        };
        write_map(realizer.unigrams());
        write_map(realizer.bigrams());
        write_map(realizer.trigrams());
    }

    return true;
}

bool ModelBundle::load(const std::string& dir_path, bool use_mmap) {
    if (fs::exists(dir_path + "/language_rules.json"))
        language_rules.load(dir_path + "/language_rules.json");
    else if (language_rules.empty())
        language_rules.load_default();
    lexicon.load_synonyms(language_rules);
    if (code_patterns.empty()) code_patterns.load_default();
    if (fs::exists(dir_path + "/lexicon.bin")) lexicon.load(dir_path + "/lexicon.bin");
    else lexicon.clear();
    if (fs::exists(dir_path + "/coherence.bin")) coherence.load(dir_path + "/coherence.bin");
    else coherence.clear();
    if (fs::exists(dir_path + "/code_patterns.jsonl")) code_patterns.load(dir_path + "/code_patterns.jsonl");
    else if (code_patterns.empty()) code_patterns.load_default();
    // 1. Codebook
    {
        MappedReader r(dir_path + "/codebook.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, n = 0;
        if (!r.read_pod(magic) || magic != MAGIC_CB || !r.read_pod(n)) return false;
        std::vector<std::string> toks(n);
        for (uint32_t i = 0; i < n; ++i) {
            if (!r.read_string(toks[i])) return false;
        }
        std::vector<HyperVector> base_v(n), sem_v(n);
        if (n > 0) {
            if (!r.read_bytes(base_v.data(), n * sizeof(HyperVector))) return false;
            if (!r.read_bytes(sem_v.data(), n * sizeof(HyperVector))) return false;
        }
        codebook.load_raw(std::move(toks), std::move(base_v), std::move(sem_v));
    }

    // 2. SDM
    {
        MappedReader r(dir_path + "/sdm.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0;
        SDMConfig cfg;
        uint8_t bin = 0;
        uint64_t wc = 0;
        if (!r.read_pod(magic) || magic != MAGIC_SDM || !r.read_pod(cfg) ||
            !r.read_pod(bin) || !r.read_pod(wc)) return false;
        const size_t m = cfg.num_locations;
        std::vector<HyperVector> addrs(m);
        std::vector<uint8_t> touched(m);
        if (!r.read_bytes(addrs.data(), m * sizeof(HyperVector))) return false;
        if (!r.read_bytes(touched.data(), m * sizeof(uint8_t))) return false;
        std::vector<int8_t> counters;
        std::vector<HyperVector> bin_c;
        if (bin) {
            bin_c.resize(m);
            if (!r.read_bytes(bin_c.data(), m * sizeof(HyperVector))) return false;
        } else {
            counters.resize(m * VSA_BITS);
            if (!r.read_bytes(counters.data(), m * VSA_BITS * sizeof(int8_t))) return false;
        }
        sdm.load_raw(cfg, std::move(addrs), std::move(counters), std::move(bin_c),
                     std::move(touched), bin != 0, wc);
    }

    // 3. Tsetlin
    {
        MappedReader r(dir_path + "/tsetlin.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0;
        uint8_t is_sparse = 0;
        if (!r.read_pod(magic) || magic != MAGIC_TM || !r.read_pod(is_sparse)) return false;
        if (is_sparse) {
            uint32_t nf = 0, nc = 0, cpc = 0;
            if (!r.read_pod(nf) || !r.read_pod(nc) || !r.read_pod(cpc)) return false;
            sparse_tsetlin.num_features = nf;
            sparse_tsetlin.num_classes = nc;
            sparse_tsetlin.clauses_per_class = cpc;
            sparse_tsetlin.clauses.resize(static_cast<size_t>(nc) * cpc);
            for (auto& cl : sparse_tsetlin.clauses) {
                uint32_t np = 0, nn = 0;
                if (!r.read_pod(cl.sign) || !r.read_pod(np) || !r.read_pod(nn)) return false;
                cl.pos_literals.resize(np);
                cl.neg_literals.resize(nn);
                if (!r.read_bytes(cl.pos_literals.data(), np * sizeof(uint16_t))) return false;
                if (!r.read_bytes(cl.neg_literals.data(), nn * sizeof(uint16_t))) return false;
            }
            has_sparse_tsetlin = true;
        } else {
            TsetlinConfig cfg;
            if (!r.read_pod(cfg)) return false;
            std::vector<int8_t> ta(cfg.num_classes * cfg.clauses_per_class * 2 * cfg.num_features);
            std::vector<int8_t> signs(cfg.num_classes * cfg.clauses_per_class);
            if (!r.read_bytes(ta.data(), ta.size() * sizeof(int8_t))) return false;
            if (!r.read_bytes(signs.data(), signs.size() * sizeof(int8_t))) return false;
            tsetlin.load_raw(cfg, std::move(ta), std::move(signs));
            has_sparse_tsetlin = false;
        }
    }

    has_context_policy = false;
    context_policy = SparseTsetlinModel{};
    if (fs::exists(dir_path + "/dialogue_context.bin")) {
        MappedReader r(dir_path + "/dialogue_context.bin", use_mmap);
        uint32_t magic; uint8_t enabled;
        if (!r.valid() || !r.read_pod(magic) || magic != 0x43545831 || !r.read_pod(enabled) || enabled > 1) return false;
        if (enabled) {
            uint32_t nf, nc, cpc;
            if (!r.read_pod(nf) || !r.read_pod(nc) || !r.read_pod(cpc) || nf != 16 || !nc || nc > size_t(DialogIntent::COUNT) || !cpc || cpc > 512) return false;
            context_policy.num_features = nf; context_policy.num_classes = nc; context_policy.clauses_per_class = cpc;
            context_policy.clauses.resize(size_t(nc) * cpc);
            for (auto& clause : context_policy.clauses) {
                uint32_t np, nn;
                if (!r.read_pod(clause.sign) || !r.read_pod(np) || !r.read_pod(nn) || np > nf || nn > nf) return false;
                clause.pos_literals.resize(np); clause.neg_literals.resize(nn);
                if (!r.read_bytes(clause.pos_literals.data(), np * sizeof(uint16_t)) || !r.read_bytes(clause.neg_literals.data(), nn * sizeof(uint16_t))) return false;
                for (auto f : clause.pos_literals) if (f >= nf) return false;
                for (auto f : clause.neg_literals) if (f >= nf) return false;
            }
            has_context_policy = true;
        }
    }

    // 4. Causal Graph
    {
        MappedReader r(dir_path + "/graph.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, nn = 0, nr = 0, ne = 0;
        if (!r.read_pod(magic) || (magic != MAGIC_GR && magic != MAGIC_GR_LEGACY) ||
            !r.read_pod(nn) || !r.read_pod(nr) || !r.read_pod(ne)) return false;
        std::vector<GraphNode> nodes(nn);
        for (uint32_t i = 0; i < nn; ++i) {
            uint8_t g = 0, num = 0;
            if (!r.read_pod(nodes[i].id) || !r.read_pod(nodes[i].frequency)) return false;
            if (magic == MAGIC_GR && (!r.read_pod(g) || !r.read_pod(num))) return false;
            if (!r.read_string(nodes[i].name)) return false;
            if (magic == MAGIC_GR_LEGACY) {
                g = static_cast<uint8_t>(TextParser::infer_gender(nodes[i].name));
                num = static_cast<uint8_t>(TextParser::infer_number(nodes[i].name));
            }
            nodes[i].gender = static_cast<GramGender>(g);
            nodes[i].number = static_cast<GramNumber>(num);
        }
        std::vector<std::string> rels(nr);
        for (uint32_t i = 0; i < nr; ++i) {
            if (!r.read_string(rels[i])) return false;
        }
        std::vector<GraphEdge> edges(ne);
        if (magic == MAGIC_GR_LEGACY) {
            struct LegacyEdge {
                uint32_t src, rel, dst;
                RelationType type;
                uint32_t frequency;
                float confidence;
                bool user_taught;
            };
            for (auto& e : edges) {
                LegacyEdge old{};
                if (!r.read_pod(old)) return false;
                e.src = old.src; e.rel = old.rel; e.dst = old.dst;
                e.type = old.type; e.frequency = old.frequency;
                e.confidence = old.confidence; e.user_taught = old.user_taught;
            }
        } else if (ne > 0 && !r.read_bytes(edges.data(), ne * sizeof(GraphEdge))) return false;
        for (const auto& e : edges)
            if (e.src >= nn || e.dst >= nn || e.rel >= nr || e.modality[15] != '\0') return false;
        graph.load_raw(std::move(nodes), std::move(rels), std::move(edges));
    }
    mapped_graph.reset();
    if (use_mmap && fs::exists(dir_path + "/graph_mmap.bin"))
        mapped_graph = MappedCausalGraph::open(dir_path + "/graph_mmap.bin");

    code_graph = CodeGraph{};
    if (fs::exists(dir_path + "/code_index.bin")) {
        MappedReader r(dir_path + "/code_index.bin", use_mmap);
        uint32_t magic = 0, count = 0;
        auto size = [&](uint32_t& n) { return r.read_pod(n) && n <= 1000000; };
        auto strings = [&](std::vector<std::string>& values) {
            uint32_t n = 0;
            if (!size(n)) return false;
            values.resize(n);
            for (auto& value : values) if (!r.read_string(value)) return false;
            return true;
        };
        if (!r.valid() || !r.read_pod(magic) || (magic != 0x434F4432 && magic != 0x434F4433) || !size(count)) return false;
        const bool extended = magic == 0x434F4433;
        std::vector<CodeSymbol> symbols(count);
        for (uint32_t i = 0; i < count; ++i) {
            auto& sym = symbols[i]; uint8_t lang = 0; sym.id = i;
            if (!r.read_string(sym.name) || !r.read_string(sym.return_type) ||
                !strings(sym.param_types) || !strings(sym.param_names) || !strings(sym.includes_or_imports) ||
                !r.read_string(sym.documentation) || !r.read_string(sym.canonical_ast) ||
                !r.read_string(sym.body_snippet) || !r.read_string(sym.file_path) ||
                !r.read_pod(lang) || lang > uint8_t(CodeLanguage::UNKNOWN)) return false;
            if (extended) {
                uint8_t method = 0, is_static = 0;
                if (!r.read_pod(method) || !r.read_pod(is_static) || !r.read_string(sym.owner)) return false;
                sym.is_method = method != 0; sym.is_static = is_static != 0;
            }
            if (!r.read_pod(sym.ast_vsa)) return false;
            sym.lang = static_cast<CodeLanguage>(lang);
        }
        if (!size(count)) return false;
        std::vector<CodeEdge> edges(count);
        for (auto& edge : edges) {
            uint8_t type;
            if (!r.read_pod(edge.src) || !r.read_pod(edge.dst) || !r.read_pod(type) ||
                edge.src >= symbols.size() || edge.dst >= symbols.size() || type > 5) return false;
            edge.type = static_cast<CodeEdgeType>(type);
        }
        if (!size(count)) return false;
        std::vector<AstPattern> patterns(count);
        for (auto& pattern : patterns) {
            uint8_t lang;
            if (!r.read_string(pattern.id_tag) || !r.read_pod(lang) || !r.read_string(pattern.template_code) ||
                !strings(pattern.required_includes) || !r.read_pod(pattern.pattern_vsa)) return false;
            pattern.lang = static_cast<CodeLanguage>(lang);
        }
        code_graph.load_raw(std::move(symbols), std::move(edges), std::move(patterns));
    }
    realizer_pairs.clear();
    {
        std::ifstream pairs(dir_path + "/realizer_pairs.jsonl");
        std::string pair;
        while (std::getline(pairs, pair)) if (!pair.empty()) realizer_pairs.push_back(std::move(pair));
    }

    episodic_memory.clear();
    if (fs::exists(dir_path + "/episodic_facts.bin")) {
        MappedReader r(dir_path + "/episodic_facts.bin", use_mmap);
        uint32_t magic = 0, count = 0;
        if (!r.valid() || !r.read_pod(magic) || magic != 0x45504631 || !r.read_pod(count) || count > 100000) return false;
        for (uint32_t i = 0; i < count; ++i) {
            std::string entity, property, value, source;
            if (!r.read_string(entity) || !r.read_string(property) || !r.read_string(value) || !r.read_string(source)) return false;
            episodic_memory.remember(entity, property, value, source);
        }
    }
    episodes.clear();
    if (fs::exists(dir_path + "/episodes.bin")) {
        MappedReader r(dir_path + "/episodes.bin", use_mmap);
        uint32_t magic = 0, count = 0;
        if (!r.valid() || !r.read_pod(magic) || magic != 0x45504931 || !r.read_pod(count) || count > 128) return false;
        episodes.resize(count);
        for (auto& [user, assistant] : episodes)
            if (!r.read_string(user) || !r.read_string(assistant)) return false;
    }
    context_memory.clear();
    if (fs::exists(dir_path + "/context.bin")) {
        MappedReader r(dir_path + "/context.bin", use_mmap);
        uint32_t magic = 0, version = 0, count = 0;
        uint64_t turn = 0;
        if (!r.valid() || !r.read_pod(magic) || magic != 0x43545832 || !r.read_pod(version) || version != 1 ||
            !r.read_pod(turn) || !r.read_pod(count) || count > 1024) return false;
        std::vector<ContextEntity> entities(count);
        for (auto& entity : entities) {
            uint8_t gender = 0, number = 0;
            if (!r.read_string(entity.name) || !r.read_pod(gender) || gender > 2 ||
                !r.read_pod(number) || number > 1 || !r.read_pod(entity.mentions) ||
                !r.read_pod(entity.last_turn) || !r.read_pod(entity.salience)) return false;
            entity.gender = static_cast<GramGender>(gender);
            entity.number = static_cast<GramNumber>(number);
        }
        uint32_t concepts = 0;
        if (!r.read_pod(concepts) || concepts > 1024) return false;
        std::vector<std::string> last(concepts);
        for (auto& topic : last) if (!r.read_string(topic)) return false;
        context_memory.restore(std::move(entities), std::move(last), turn);
    }

    if (fs::exists(dir_path + "/fluency.bin")) {
        if (!fluent_realizer.load(dir_path + "/fluency.bin")) return false;
    } else fluent_realizer = AnalogicalFluencyRealizer{};
    // 5. Realizer
    {
        MappedReader r(dir_path + "/realizer.bin", use_mmap);
        if (!r.valid()) return false;
        uint32_t magic = 0, nf = 0;
        uint64_t total_toks = 0;
        if (!r.read_pod(magic) || magic != MAGIC_RL ||
            !r.read_pod(total_toks) || !r.read_pod(nf)) return false;
        std::vector<ConstructionFrame> frames(nf);
        for (uint32_t i = 0; i < nf; ++i) {
            uint8_t rt = 0;
            if (!r.read_pod(rt) || !r.read_pod(frames[i].frequency) ||
                !r.read_string(frames[i].pattern)) return false;
            frames[i].rel_type = static_cast<RelationType>(rt);
        }
        auto read_map = [&](std::unordered_map<std::string, uint32_t>& m) -> bool {
            uint32_t sz = 0;
            if (!r.read_pod(sz)) return false;
            m.reserve(sz);
            for (uint32_t i = 0; i < sz; ++i) {
                std::string k;
                uint32_t v = 0;
                if (!r.read_string(k) || !r.read_pod(v)) return false;
                m.emplace(std::move(k), v);
            }
            return true;
        };
        std::unordered_map<std::string, uint32_t> uni, bi, tri;
        if (!read_map(uni) || !read_map(bi) || !read_map(tri)) return false;
        realizer.load_raw(std::move(frames), std::move(uni), std::move(bi), std::move(tri), total_toks);
    }

    return true;
}

bool ModelBundle::refresh_mapped_graph(const std::string& dir_path) {
    mapped_graph.reset();
    if (!fs::exists(dir_path + "/graph_mmap.bin")) return false;
    mapped_graph = MappedCausalGraph::open(dir_path + "/graph_mmap.bin");
    return mapped_graph != nullptr;
}

size_t ModelBundle::disk_size_bytes(const std::string& dir_path) {
    size_t total = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir_path, ec)) {
        if (entry.is_regular_file()) {
            total += entry.file_size();
        }
    }
    return total;
}

size_t ModelBundle::estimated_ram_bytes() const noexcept {
    size_t total = codebook.size() * (2 * sizeof(HyperVector) + 32);
    total += sdm.memory_bytes() + episodic_memory.memory_bytes() + context_memory.memory_bytes();
    total += lexicon.memory_bytes() + coherence.memory_bytes();
    total += has_sparse_tsetlin ? sparse_tsetlin.memory_bytes() : tsetlin.memory_bytes();
    if (has_context_policy) total += context_policy.memory_bytes();
    total += graph.memory_bytes();
    total += realizer.memory_bytes() + fluent_realizer.memory_bytes() + code_graph.memory_bytes();
    for (const auto& pair : realizer_pairs) total += pair.capacity();
    for (const auto& [user, assistant] : episodes) total += user.capacity() + assistant.capacity();
    return total;
}

} // namespace cognicore
