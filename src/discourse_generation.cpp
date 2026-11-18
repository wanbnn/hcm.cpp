#include "core/discourse_generation.hpp"

#include "core/morphological_realizer.hpp"
#include "core/text_parser.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <unordered_set>

namespace cognicore {
namespace {

std::string lower(std::string value) { return TextParser::utf8_tolower(value); }

std::string capitalize(std::string value) {
    if (!value.empty() && static_cast<unsigned char>(value[0]) < 128)
        value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
    return value;
}

std::string lang_choice(const std::vector<std::string>& values, const std::string& lang,
                        size_t index, const std::string& fallback) {
    std::vector<std::string> selected;
    const auto prefix = lang + "|";
    for (const auto& value : values)
        if (value.starts_with(prefix)) selected.push_back(value.substr(prefix.size()));
    if (selected.empty()) {
        for (const auto& value : values) {
            const auto bar = value.find('|');
            selected.push_back(bar == std::string::npos ? value : value.substr(bar + 1));
        }
    }
    if (selected.empty()) return fallback;
    return selected[index % selected.size()];
}

std::string article_np(const Entity& entity, const std::string& lang) {
    const auto lowered = lower(entity.canonical);
    for (const auto* article : {"o ", "a ", "os ", "as ", "um ", "uma ", "the ", "an "})
        if (lowered.starts_with(article)) return entity.canonical;
    return MorphologicalRealizer::definite_article(entity.gender, entity.number, lang) + " " + entity.canonical;
}

std::string pronoun(const Entity& entity, const std::string& lang) {
    if (lang == "en") return entity.number == GramNumber::PLUR ? "they" : "it";
    if (entity.gender == GramGender::FEM)
        return entity.number == GramNumber::PLUR ? "elas" : "ela";
    return entity.number == GramNumber::PLUR ? "eles" : "ele";
}

std::string prep_article(const std::string& preposition, GramGender gender, GramNumber number,
                         const std::string& lang) {
    if (lang != "pt") return preposition + " " + MorphologicalRealizer::definite_article(gender, number, lang);
    const bool fem = gender == GramGender::FEM;
    const bool plur = number == GramNumber::PLUR;
    if (preposition == "por") return fem ? (plur ? "pelas" : "pela") : (plur ? "pelos" : "pelo");
    if (preposition == "de") return fem ? (plur ? "das" : "da") : (plur ? "dos" : "do");
    return preposition;
}

std::string prep_np(const std::string& preposition, const Entity& entity, const std::string& lang) {
    return prep_article(preposition, entity.gender, entity.number, lang) + " " + entity.canonical;
}

std::string join_list(const std::vector<std::string>& values) {
    if (values.empty()) return {};
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += (i + 1 == values.size()) ? " e " : ", ";
        out += values[i];
    }
    return out;
}

std::string entity_surface(const Entity& entity, ReferenceForm form, const std::string& lang) {
    if (form == ReferenceForm::PRONOUN) return pronoun(entity, lang);
    if (form == ReferenceForm::ZERO) return {};
    return article_np(entity, lang);
}

const Entity& entity_by_id(const std::vector<Entity>& entities, EntityId id) {
    static const Entity empty;
    return id < entities.size() ? entities[id] : empty;
}

} // namespace

bool DiscourseState::ambiguous(GramGender gender, GramNumber number) const {
    return std::find(ambiguous_features.begin(), ambiguous_features.end(),
                     std::make_pair(gender, number)) != ambiguous_features.end();
}

void DiscourseState::observe(const std::vector<Proposition>& propositions,
                             const EntityRegistry& registry) {
    std::vector<EntityId> current;
    auto add = [&](EntityId id) {
        if (id != INVALID_ENTITY && std::find(current.begin(), current.end(), id) == current.end())
            current.push_back(id);
    };
    for (const auto& proposition : propositions) add(proposition.subject);
    for (const auto& proposition : propositions) add(proposition.object);

    backward_center = INVALID_ENTITY;
    for (auto id : current)
        if (std::find(forward_centers.begin(), forward_centers.end(), id) != forward_centers.end()) {
            backward_center = id;
            break;
        }
    if (backward_center == INVALID_ENTITY && !current.empty()) backward_center = current.front();

    std::unordered_map<uint64_t, int> feature_counts;
    auto feature_key = [](GramGender gender, GramNumber number) {
        return (uint64_t(static_cast<uint8_t>(gender)) << 8) | uint8_t(number);
    };
    for (auto id : current) {
        const auto& entity = registry.entity(id);
        feature_counts[feature_key(entity.gender, entity.number)]++;
        salience[id]++;
        Mention mention;
        mention.entity = id;
        mention.surface = entity.canonical;
        mention.position = last_mention.count(id) ? last_mention[id].position + 1 : 0;
        last_mention[id] = mention;
    }
    ambiguous_features.clear();
    for (auto id : current) {
        const auto& entity = registry.entity(id);
        if (feature_counts[feature_key(entity.gender, entity.number)] > 1)
            ambiguous_features.emplace_back(entity.gender, entity.number);
    }
    forward_centers = std::move(current);
}

EntityId EntityRegistry::get_or_add(const std::string& canonical, GramGender gender,
                                    GramNumber number) {
    const auto key = lower(canonical);
    const auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    Entity entity;
    entity.id = static_cast<EntityId>(entities_.size());
    entity.canonical = canonical;
    entity.gender = gender;
    entity.number = number;
    entities_.push_back(entity);
    index_[key] = entity.id;
    return entity.id;
}

EntityId EntityRegistry::find(const std::string& canonical) const {
    const auto it = index_.find(lower(canonical));
    return it == index_.end() ? INVALID_ENTITY : it->second;
}

const Entity& EntityRegistry::entity(EntityId id) const {
    static const Entity empty;
    return id < entities_.size() ? entities_[id] : empty;
}

ReferencePlan ReferenceGenerator::decide(EntityId entity, GrammaticalRole role, size_t position,
                                         const DiscourseState& state,
                                         const EntityRegistry& registry,
                                         size_t max_distance) const {
    ReferencePlan plan;
    plan.entity = entity;
    if (entity == INVALID_ENTITY) return plan;
    const auto& value = registry.entity(entity);
    const auto seen = state.last_mention.find(entity) != state.last_mention.end();
    if (!seen) { plan.form = ReferenceForm::FULL_NP; return plan; }
    if (role != GrammaticalRole::SUBJECT) { plan.form = ReferenceForm::FULL_NP; return plan; }
    if (state.ambiguous(value.gender, value.number)) { plan.form = ReferenceForm::DEFINITE_NP; return plan; }
    if (entity == state.backward_center) { plan.form = ReferenceForm::PRONOUN; return plan; }
    const auto at = std::find(state.forward_centers.begin(), state.forward_centers.end(), entity);
    if (at != state.forward_centers.end() &&
        static_cast<size_t>(std::distance(state.forward_centers.begin(), at)) <= max_distance) {
        plan.form = ReferenceForm::PRONOUN;
        return plan;
    }
    (void)position;
    plan.form = ReferenceForm::DEFINITE_NP;
    return plan;
}

std::vector<CoordinationPlan> Aggregator::group(const std::vector<Proposition>& propositions,
                                                const EntityRegistry& registry) const {
    std::vector<CoordinationPlan> plans;
    std::set<std::pair<EntityId, int>> seen;
    for (const auto& proposition : propositions) {
        if (proposition.step.type != RelationType::HAS_PART &&
            proposition.step.type != RelationType::PART_OF) continue;
        const int kind = static_cast<int>(proposition.step.type);
        if (!seen.insert({proposition.subject, kind}).second) continue;
        CoordinationPlan plan;
        plan.subject = proposition.subject;
        plan.relation = proposition.step.type;
        plan.preposition = proposition.step.type == RelationType::HAS_PART ? "por" : "de";
        bool any_masc = false;
        std::set<EntityId> conjuncts;
        for (const auto& other : propositions) {
            if (other.subject != proposition.subject || other.step.type != proposition.step.type) continue;
            if (!conjuncts.insert(other.object).second) continue;
            plan.conjuncts.push_back(other.object);
            if (registry.entity(other.object).gender == GramGender::MASC) any_masc = true;
        }
        plan.gender = any_masc ? GramGender::MASC : GramGender::FEM;
        plan.number = plan.conjuncts.size() > 1 ? GramNumber::PLUR : GramNumber::SING;
        plans.push_back(std::move(plan));
    }
    return plans;
}

TextPlan TextPlanner::plan(const DocumentParagraph& paragraph, const LanguageRules& rules,
                           const DiscourseConfig* config) const {
    TextPlan text_plan;
    EntityRegistry registry;
    std::vector<Proposition> all;
    std::set<std::string> seen_propositions;
    for (const auto& sentence : paragraph.sentences)
        for (const auto& path : sentence.plan.paths)
            for (const auto& step : path.steps) {
                const auto key = lower(step.src) + "|" + lower(step.rel) + "|" + lower(step.dst);
                if (!seen_propositions.insert(key).second) continue;
                Proposition proposition;
                proposition.step = step;
                proposition.subject = registry.get_or_add(step.src, step.src_gender, step.src_number);
                proposition.object = registry.get_or_add(step.dst, step.dst_gender, step.dst_number);
                all.push_back(std::move(proposition));
            }
    if (all.empty()) return text_plan;

    ParagraphPlan paragraph_plan;
    paragraph_plan.heading = paragraph.topic;
    DiscourseState state;
    Aggregator aggregator;
    ReferenceGenerator references;

    auto make_sentence = [&](SentenceKind kind, const std::vector<Proposition>& propositions) {
        if (propositions.empty()) return;
        SentencePlan sentence;
        sentence.kind = kind;
        sentence.propositions = propositions;
        sentence.topic = propositions.front().subject;
        sentence.focus = propositions.back().object;
        sentence.given_topic = state.last_mention.count(sentence.topic) != 0;
        sentence.coordinations = aggregator.group(propositions, registry);
        EntityId previous_subject = INVALID_ENTITY;
        for (size_t i = 0; i < propositions.size(); ++i) {
            const auto& proposition = propositions[i];
            auto reference = references.decide(proposition.subject, GrammaticalRole::SUBJECT,
                                               i, state, registry,
                                               config ? config->max_pronoun_distance : 2);
            if (i > 0 && proposition.subject == previous_subject)
                reference.form = ReferenceForm::PRONOUN;
            sentence.references.push_back(reference);
            previous_subject = proposition.subject;
        }
        paragraph_plan.sentences.push_back(std::move(sentence));
        state.observe(propositions, registry);
    };

    std::vector<Proposition> definitions, parts, functions, sequence;
    for (const auto& proposition : all) {
        if (proposition.step.type == RelationType::TAXONOMIC ||
            proposition.step.type == RelationType::PROPERTY) definitions.push_back(proposition);
        else if (proposition.step.type == RelationType::HAS_PART ||
                 proposition.step.type == RelationType::PART_OF) parts.push_back(proposition);
        else if (proposition.step.type == RelationType::SEQUENCE ||
                 proposition.step.type == RelationType::PRECEDES) sequence.push_back(proposition);
        else functions.push_back(proposition);
    }
    make_sentence(SentenceKind::DEFINITION, definitions);
    make_sentence(SentenceKind::PART_OF, parts);
    make_sentence(SentenceKind::FUNCTION, functions);
    make_sentence(SentenceKind::SEQUENCE, sequence);
    paragraph_plan.state = std::move(state);
    text_plan.paragraphs.push_back(std::move(paragraph_plan));
    text_plan.entities = registry.entities();
    return text_plan;
}

double GenerationSearch::score(const std::string& text, const DiscourseConfig* config) const {
    if (text.empty()) return 1e9;
    const auto tokens = TextParser{}.tokenize(text);
    std::unordered_set<std::string> seen;
    const double repetition = config ? config->repetition_penalty : 0.35;
    const double pronoun_bonus = config ? config->pronoun_bonus : 0.15;
    const double length_penalty = config ? config->length_penalty : 0.01;
    const double connector_penalty = config ? config->connector_penalty : 0.05;
    double cost = 0.0;
    int content = 0;
    for (const auto& token : tokens) {
        if (token.size() < 3) continue;
        ++content;
        if (!seen.insert(token).second) cost += repetition;
    }
    cost += content * length_penalty;
    if (text.find("Ele ") != std::string::npos || text.find("Ela ") != std::string::npos) cost -= pronoun_bonus;
    if (text.find("Além disso") != std::string::npos) cost += connector_penalty;
    return cost;
}

std::string GenerationSearch::select(std::vector<Candidate> candidates) const {
    if (candidates.empty()) return {};
    auto best = candidates.front();
    for (const auto& candidate : candidates)
        if (candidate.cost < best.cost) best = candidate;
    return best.text;
}

namespace {

std::string realize_sentence(const SentencePlan& sentence, const std::vector<Entity>& entities,
                             const LanguageRules& rules, int variant,
                             const DiscourseConfig* config) {
    const std::string lang = rules.language.empty() ? "pt" : rules.language;
    MorphologicalRealizer morphology(rules);
    std::string output;

    auto subject_surface = [&](size_t index) {
        if (index >= sentence.references.size()) return std::string{};
        const auto& ref = sentence.references[index];
        return entity_surface(entity_by_id(entities, ref.entity), ref.form, lang);
    };

    if (sentence.kind == SentenceKind::PART_OF) {
        if (sentence.coordinations.empty()) return {};
        const auto& coordination = sentence.coordinations.front();
        std::vector<std::string> objects;
        for (auto id : coordination.conjuncts)
            objects.push_back(prep_np(coordination.preposition, entity_by_id(entities, id), lang));
        auto subject = subject_surface(0);
        if (subject.empty()) subject = article_np(entity_by_id(entities, coordination.subject), lang);
        std::string text;
        const auto constructions = config ? config->constructions_for(coordination.relation)
                                          : std::vector<const Construction*>{};
        if (!constructions.empty())
            text = constructions[static_cast<size_t>(variant < 0 ? 0 : variant) % constructions.size()]->pattern;
        else
            text = lang_choice(rules.part_of_templates, lang,
                               variant < 0 ? 0 : static_cast<size_t>(variant),
                               "{SUBJ} é formado {OBJS}.");
        const auto subj_at = text.find("{SUBJ}");
        if (subj_at != std::string::npos) text.replace(subj_at, 6, subject);
        const auto objs_at = text.find("{OBJS}");
        if (objs_at != std::string::npos) text.replace(objs_at, 6, join_list(objects));
        return MorphologicalRealizer::contract_prepositions(text);
    }

    if (sentence.kind == SentenceKind::SEQUENCE) {
        std::string text;
        for (size_t i = 0; i < sentence.propositions.size(); ++i) {
            const auto& proposition = sentence.propositions[i];
            auto subject = subject_surface(i);
            if (subject.empty()) subject = article_np(entity_by_id(entities, proposition.subject), lang);
            auto clause = morphology.realize_clause(proposition.step, subject, lang);
            if (sentence.propositions.size() == 1) return MorphologicalRealizer::contract_prepositions(clause);
            if (static_cast<unsigned char>(clause[0]) < 128)
                clause[0] = static_cast<char>(std::tolower(static_cast<unsigned char>(clause[0])));
            auto ordinal = i + 1 == sentence.propositions.size()
                ? lang_choice(rules.sequence_ordinals, lang, 3, "Por fim")
                : lang_choice(rules.sequence_ordinals, lang, i, i == 0 ? "Primeiro" : "Em seguida");
            if (i == 0) text = capitalize(ordinal) + ", " + clause;
            else text += "; " + lower(ordinal) + ", " + clause;
        }
        return MorphologicalRealizer::contract_prepositions(text);
    }

    std::string last_subject;
    for (size_t i = 0; i < sentence.propositions.size(); ++i) {
        const auto& proposition = sentence.propositions[i];
        auto subject = subject_surface(i);
        if (subject.empty()) subject = article_np(entity_by_id(entities, proposition.subject), lang);
        if (i > 0 && proposition.step.src == last_subject)
            subject = pronoun(entity_by_id(entities, proposition.subject), lang);
        auto clause = MorphologicalRealizer::contract_prepositions(
            morphology.realize_clause(proposition.step, subject, lang));
        if (!output.empty()) output += ". ";
        output += capitalize(clause);
        last_subject = proposition.step.src;
    }
    return output;
}

} // namespace

std::string DiscourseGenerator::realize(const DocumentParagraph& paragraph,
                                        const LanguageRules& rules,
                                        const DiscourseConfig* config) const {
    TextPlanner planner;
    const auto text_plan = planner.plan(paragraph, rules, config);
    if (text_plan.paragraphs.empty()) return {};
    GenerationSearch search;
    std::vector<GenerationSearch::Candidate> candidates;
    for (int variant = 0; variant < 3; ++variant) {
        std::string text;
        for (const auto& sentence : text_plan.paragraphs.front().sentences) {
            const auto realized = realize_sentence(sentence, text_plan.entities, rules, variant, config);
            if (realized.empty()) continue;
            if (!text.empty()) text += ' ';
            text += capitalize(realized);
            if (text.back() != '.' && text.back() != '!' && text.back() != '?') text += '.';
        }
        if (text.empty()) continue;
        const bool duplicate = std::any_of(candidates.begin(), candidates.end(),
            [&](const GenerationSearch::Candidate& candidate) { return candidate.text == text; });
        if (!duplicate) candidates.push_back({text, search.score(text, config)});
    }
    return search.select(std::move(candidates));
}

} // namespace cognicore
