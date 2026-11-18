#include "core/context_memory.hpp"

#include <algorithm>
#include <string_view>
#include <unordered_set>

namespace cognicore {
namespace {

constexpr size_t MAX_ENTITIES = 64;
constexpr size_t MAX_TOPIC_CONCEPTS = 8;

bool contains_whole_name(const std::string& haystack, const std::string& needle) {
    if (needle.empty() || haystack.empty()) return false;
    auto alnum = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                              (c >= '0' && c <= '9') || c >= 0x80; };
    size_t at = 0;
    while ((at = haystack.find(needle, at)) != std::string::npos) {
        const bool left = at == 0 || !alnum(static_cast<unsigned char>(haystack[at - 1]));
        const size_t end = at + needle.size();
        const bool right = end >= haystack.size() || !alnum(static_cast<unsigned char>(haystack[end]));
        if (left && right) return true;
        at = end;
    }
    return false;
}

struct Pronoun {
    const char* word;
    GramGender gender;
    GramNumber number;
    bool locative;
};

const Pronoun pronouns[] = {
    {"ele", GramGender::MASC, GramNumber::SING, false},
    {"ela", GramGender::FEM, GramNumber::SING, false},
    {"eles", GramGender::MASC, GramNumber::PLUR, false},
    {"elas", GramGender::FEM, GramNumber::PLUR, false},
    {"isso", GramGender::NEUT, GramNumber::SING, false},
    {"isto", GramGender::NEUT, GramNumber::SING, false},
    {"esse", GramGender::MASC, GramNumber::SING, false},
    {"essa", GramGender::FEM, GramNumber::SING, false},
    {"este", GramGender::MASC, GramNumber::SING, false},
    {"esta", GramGender::FEM, GramNumber::SING, false},
    {"aquele", GramGender::MASC, GramNumber::SING, false},
    {"aquela", GramGender::FEM, GramNumber::SING, false},
    {"lá", GramGender::NEUT, GramNumber::SING, true},
    {"la", GramGender::NEUT, GramNumber::SING, true},
    {"aí", GramGender::NEUT, GramNumber::SING, true},
    {"there", GramGender::NEUT, GramNumber::SING, true},
    {"it", GramGender::NEUT, GramNumber::SING, false},
    {"they", GramGender::NEUT, GramNumber::PLUR, false},
    {"he", GramGender::MASC, GramNumber::SING, false},
    {"she", GramGender::FEM, GramNumber::SING, false},
    {"this", GramGender::NEUT, GramNumber::SING, false},
    {"that", GramGender::NEUT, GramNumber::SING, false},
    {"these", GramGender::NEUT, GramNumber::PLUR, false},
    {"those", GramGender::NEUT, GramNumber::PLUR, false}
};

bool is_follow_up(const std::string& lower) {
    static const char* cues[] = {
        "e", "and", "por quê", "porque", "why", "continue", "continua",
        "explique", "explain", "mais sobre", "more about", "e isso", "e ele", "e ela",
        "nesse", "nessa", "sobre isso", "about that", "fale mais", "tell me more"
    };
    for (const auto* cue : cues) {
        if (std::string_view(cue).find(' ') != std::string_view::npos) {
            if (lower.find(cue) != std::string::npos) return true;
        } else if (contains_whole_name(lower, cue)) {
            return true;
        }
    }
    return false;
}

bool locative_preferred(const std::string& name) {
    static const char* words[] = {
        "servidor", "máquina", "maquina", "computador", "sistema", "cpu", "local",
        "workspace", "projeto", "repositório", "repositorio", "server", "machine", "host"
    };
    for (const auto* word : words) if (name.find(word) != std::string::npos) return true;
    return false;
}

} // namespace

ContextEntity* ContextMemory::find(const std::string& name) {
    for (auto& entity : entities_) if (entity.name == name) return &entity;
    return nullptr;
}

const ContextEntity* ContextMemory::find(const std::string& name) const {
    for (const auto& entity : entities_) if (entity.name == name) return &entity;
    return nullptr;
}

void ContextMemory::clear() {
    entities_.clear();
    last_concepts_.clear();
    last_user_.clear();
    turn_ = 0;
}

void ContextMemory::restore(std::vector<ContextEntity> entities,
                            std::vector<std::string> last_concepts, uint64_t turn) {
    entities_ = std::move(entities);
    if (entities_.size() > MAX_ENTITIES) entities_.erase(entities_.begin(), entities_.end() - MAX_ENTITIES);
    last_concepts_ = std::move(last_concepts);
    if (last_concepts_.size() > MAX_TOPIC_CONCEPTS) last_concepts_.resize(MAX_TOPIC_CONCEPTS);
    turn_ = turn;
    last_user_.clear();
}

size_t ContextMemory::memory_bytes() const {
    size_t bytes = entities_.capacity() * sizeof(ContextEntity) + last_concepts_.capacity() * sizeof(std::string);
    for (const auto& entity : entities_) bytes += entity.name.capacity();
    for (const auto& topic : last_concepts_) bytes += topic.capacity();
    return bytes;
}

void ContextMemory::observe_entity(const std::string& raw_name, GramGender gender, GramNumber number) {
    std::string name = raw_name;
    for (const auto* prefix : {"meu ", "minha ", "meus ", "minhas ", "seu ", "sua "}) {
        const size_t size = std::char_traits<char>::length(prefix);
        if (name.starts_with(prefix)) { name.erase(0, size); break; }
    }
    for (const auto* suffix : {" só", " so", " apenas"}) {
        const size_t size = std::char_traits<char>::length(suffix);
        if (name.size() > size && name.ends_with(suffix)) { name.resize(name.size() - size); break; }
    }
    if (name.empty() || name.size() > 120) return;
    if (auto* existing = find(name)) {
        existing->mentions += 1;
        existing->last_turn = static_cast<uint32_t>(turn_);
        existing->salience = std::min<int32_t>(existing->salience + 8, 400);
        existing->gender = gender;
        existing->number = number;
        return;
    }
    if (entities_.size() >= MAX_ENTITIES) {
        auto weakest = std::min_element(entities_.begin(), entities_.end(), [](const auto& a, const auto& b) {
            return a.salience < b.salience;
        });
        if (weakest != entities_.end()) entities_.erase(weakest);
    }
    entities_.push_back(ContextEntity{name, gender, number, 1,
        static_cast<uint32_t>(turn_), 8});
}

void ContextMemory::observe_user_turn(const std::string& text, const TextParser& parser, const CausalGraph& graph) {
    ++turn_;
    last_user_ = text;
    // Decay salience slowly so recent, repeated entities win.
    for (auto& entity : entities_) entity.salience = entity.salience * 7 / 8;
    const auto parsed = parser.parse_text(text);
    for (const auto& triple : parsed.triples) {
        observe_entity(triple.subject, triple.subj_gender, triple.subj_number);
        observe_entity(triple.object, triple.obj_gender, triple.obj_number);
    }
    const auto lower = TextParser::utf8_tolower(text);
    for (const auto& node : graph.nodes()) {
        if (node.name.size() < 3) continue;
        if (contains_whole_name(lower, node.name))
            observe_entity(node.name, node.gender, node.number);
    }
}

void ContextMemory::observe_concepts(const std::vector<std::string>& concepts) {
    for (const auto& topic : concepts) {
        if (std::find(last_concepts_.begin(), last_concepts_.end(), topic) != last_concepts_.end()) continue;
        last_concepts_.push_back(topic);
    }
    if (last_concepts_.size() > MAX_TOPIC_CONCEPTS)
        last_concepts_.erase(last_concepts_.begin(), last_concepts_.end() - MAX_TOPIC_CONCEPTS);
}

std::optional<ReferenceResolution> ContextMemory::resolve(const std::string& text, const TextParser& parser) const {
    if (entities_.empty()) return std::nullopt;
    const auto lower = TextParser::utf8_tolower(text);
    const auto tokens = parser.tokenize(text);
    const Pronoun* pronoun = nullptr;
    for (const auto& token : tokens) {
        for (const auto& candidate : pronouns) {
            if (token == candidate.word) { pronoun = &candidate; break; }
        }
        if (pronoun) break;
    }
    // A pronoun/deictic is resolved before explicit substring mentions. This
    // prevents a generic noun inside the same sentence (e.g. "RAM") from
    // intercepting a deictic such as "lá".
    if (!pronoun) {
        for (auto it = entities_.rbegin(); it != entities_.rend(); ++it) {
            if (!contains_whole_name(lower, it->name)) continue;
            ReferenceResolution result;
            result.surface = it->name;
            result.referent = it->name;
            result.reason = "menção explícita";
            result.score = 1.0;
            result.candidates = {it->name};
            return result;
        }
        return std::nullopt;
    }

    struct Scored { const ContextEntity* entity; int score; };
    std::vector<Scored> scored;
    for (const auto& entity : entities_) {
        int score = entity.salience + static_cast<int>(entity.mentions) * 4;
        if (entity.last_turn + 1 >= turn_) score += 12;
        if (pronoun->gender != GramGender::NEUT && entity.gender == pronoun->gender) score += 10;
        if (pronoun->gender != GramGender::NEUT && entity.gender != pronoun->gender) score -= 10;
        if (entity.number == pronoun->number) score += 6;
        else score -= 6;
        if (pronoun->locative && locative_preferred(entity.name)) score += 20;
        if (pronoun->locative && !locative_preferred(entity.name)) score -= 4;
        // "isso"/"that" refers to the most recent discourse object/subject.
        if (pronoun->gender == GramGender::NEUT && !pronoun->locative && entity.last_turn + 1 >= turn_) score += 8;
        scored.push_back({&entity, score});
    }
    if (scored.empty()) return std::nullopt;
    std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    if (scored.front().score <= 0) return std::nullopt;

    ReferenceResolution result;
    result.surface = pronoun->word;
    result.referent = scored.front().entity->name;
    result.score = scored.front().score;
    result.reason = "resolução contextual (" + std::string(pronoun->locative ? "dêixis" : "pronome") +
                    ", gênero/número e recência)";
    for (const auto& item : scored) {
        result.candidates.push_back(item.entity->name);
        if (result.candidates.size() == 4) break;
    }
    if (scored.size() > 1 && scored.front().score == scored[1].score &&
        pronoun->gender == GramGender::NEUT) {
        result.ambiguous = true;
        result.referent.clear();
        result.reason = "referência ambígua: múltiplas entidades com a mesma saliência";
    }
    return result;
}

std::vector<std::string> ContextMemory::carryover_concepts(const std::string& text, const TextParser& parser,
                                                           const CausalGraph& graph) const {
    if (last_concepts_.empty()) return {};
    const auto lower = TextParser::utf8_tolower(text);
    if (!is_follow_up(lower)) return {};
    const auto content = parser.extract_content_tokens(text);
    size_t known = 0;
    for (const auto& token : content) {
        if (graph.find_node(token) != UINT32_MAX) { ++known; continue; }
        for (const auto& entity : entities_) if (entity.name == token) { ++known; break; }
    }
    if (known > 0) return {};
    std::vector<std::string> out;
    for (const auto& topic : last_concepts_) {
        out.push_back(topic);
        if (out.size() == 3) break;
    }
    return out;
}

} // namespace cognicore
