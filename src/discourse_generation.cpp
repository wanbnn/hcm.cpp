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
    if (entity.type == "CITY" || entity.type == "PERSON" || entity.type == "LANDMARK")
        return entity.canonical;
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
                         const std::string& lang, const DiscourseConfig* config) {
    if (config) {
        const auto contracted = config->contraction(preposition, gender, number);
        if (!contracted.empty()) return contracted;
    }
    if (lang != "pt") return preposition + " " + MorphologicalRealizer::definite_article(gender, number, lang);
    const bool fem = gender == GramGender::FEM;
    const bool plur = number == GramNumber::PLUR;
    if (preposition == "por") return fem ? (plur ? "pelas" : "pela") : (plur ? "pelos" : "pelo");
    if (preposition == "de") return fem ? (plur ? "das" : "da") : (plur ? "dos" : "do");
    return preposition;
}

std::string prep_np(const std::string& preposition, const Entity& entity, const std::string& lang,
                    const DiscourseConfig* config) {
    if (entity.type == "CITY" || entity.type == "PERSON" || entity.type == "LANDMARK")
        return preposition + " " + entity.canonical;
    // Coordinate compound objects per conjunct: "as bielas e os êmbolos"
    // becomes "pelas bielas e pelos êmbolos", not "pelas bielas e os êmbolos".
    const auto& name = entity.canonical;
    const auto split = name.find(" e ");
    if (split != std::string::npos) {
        const auto left = name.substr(0, split);
        const auto right = name.substr(split + 3);
        auto starts_with_article = [](const std::string& value) {
            return value.starts_with("o ") || value.starts_with("a ") ||
                   value.starts_with("os ") || value.starts_with("as ") ||
                   value.starts_with("um ") || value.starts_with("uma ");
        };
        if (starts_with_article(left) && starts_with_article(right)) {
            TextParser parser;
            auto conjunct = [&](const std::string& value) {
                const auto space = value.find(' ');
                const auto noun = space == std::string::npos ? value : value.substr(space + 1);
                GramGender gender = parser.infer_gender(noun);
                GramNumber number = parser.infer_number(noun);
                if (value.starts_with("as ") || value.starts_with("a ") || value.starts_with("uma ")) gender = GramGender::FEM;
                if (value.starts_with("os ") || value.starts_with("o ") || value.starts_with("um ")) gender = GramGender::MASC;
                if (value.starts_with("as ") || value.starts_with("os ")) number = GramNumber::PLUR;
                return prep_article(preposition, gender, number, lang, config) + " " + noun;
            };
            return conjunct(left) + " e " + conjunct(right);
        }
    }
    return prep_article(preposition, entity.gender, entity.number, lang, config) + " " + name;
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

std::string capitalize_proper_nouns(std::string text, const DiscourseConfig* config) {
    if (!config) return text;
    for (const auto& proper : config->proper_nouns) {
        if (proper.empty()) continue;
        const auto needle = lower(proper);
        size_t at = 0;
        while ((at = lower(text).find(needle, at)) != std::string::npos) {
            text.replace(at, proper.size(), proper);
            at += proper.size();
        }
    }
    return text;
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
                                    GramNumber number, const std::string& type) {
    const auto key = lower(canonical);
    const auto it = index_.find(key);
    if (it != index_.end()) return it->second;
    Entity entity;
    entity.id = static_cast<EntityId>(entities_.size());
    entity.canonical = canonical;
    entity.gender = gender;
    entity.number = number;
    entity.type = type;
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
                const auto relation_key = (step.type == RelationType::TAXONOMIC ||
                                           step.type == RelationType::PROPERTY)
                    ? std::string("COP") : lower(step.rel);
                const auto key = lower(step.src) + "|" + relation_key + "|" + lower(step.dst);
                if (!seen_propositions.insert(key).second) continue;
                Proposition proposition;
                proposition.step = step;
                proposition.subject = registry.get_or_add(
                    step.src, step.src_gender, step.src_number,
                    config ? config->entity_type(step.src) : std::string{});
                proposition.object = registry.get_or_add(
                    step.dst, step.dst_gender, step.dst_number,
                    config ? config->entity_type(step.dst) : std::string{});
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

uint64_t CohesionRanker::features(const std::string& text) {
    const auto tokens = TextParser{}.tokenize(text);
    std::unordered_set<std::string> seen;
    int repeats = 0;
    int content = 0;
    for (const auto& token : tokens) {
        if (token.size() < 3) continue;
        ++content;
        if (!seen.insert(token).second) ++repeats;
    }
    uint64_t mask = 0;
    const auto lower_text = lower(text);
    if (lower_text.find("ele ") != std::string::npos || lower_text.find("ela ") != std::string::npos)
        mask |= 1ULL << 0;
    if (repeats > 1) mask |= 1ULL << 1;
    if (content > 40) mask |= 1ULL << 2;
    if (lower_text.find("ele ") != std::string::npos && lower_text.find("ele ", lower_text.find("ele ") + 1) != std::string::npos)
        mask |= 1ULL << 3;
    if (lower_text.find("além disso") != lower_text.rfind("além disso")) mask |= 1ULL << 4;
    if (lower_text.find(" e ") != std::string::npos && lower_text.find(',') != std::string::npos)
        mask |= 1ULL << 5;
    if (lower_text.find("primeiro") != std::string::npos || lower_text.find("em seguida") != std::string::npos)
        mask |= 1ULL << 6;
    if (lower_text.find("formado") != std::string::npos || lower_text.find("contém") != std::string::npos)
        mask |= 1ULL << 7;
    return mask;
}

CohesionRanker::CohesionRanker() {
    TsetlinConfig config;
    config.num_features = 8;
    config.num_classes = 2;
    config.clauses_per_class = 16;
    config.n_states = 20;
    config.margin_T = 10;
    config.seed = 0x51A7C0DEULL;
    TsetlinMachine machine(config);
    std::vector<std::vector<uint64_t>> x;
    std::vector<size_t> y;
    for (uint64_t mask = 0; mask < 256; ++mask) {
        const bool pronoun = mask & 1;
        const bool repeated = mask & 2;
        const bool long_text = mask & 4;
        const bool ambiguous = mask & 8;
        const bool connector_repeat = mask & 16;
        const bool coordination = mask & 32;
        const bool sequence = mask & 64;
        const bool part_of = mask & 128;
        const bool good = !repeated && !ambiguous && !connector_repeat && !long_text &&
                          (pronoun || coordination || sequence || part_of);
        x.push_back({mask});
        y.push_back(good ? 1 : 0);
    }
    machine.fit(x, y, 40, false);
    model_ = machine.export_sparse();
}

double CohesionRanker::score(const std::string& text) const {
    const auto mask = features(text);
    const auto scores = model_.score_classes(&mask);
    if (scores.size() < 2) return 0.0;
    return (scores[1] - scores[0]) / 100.0;
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
    static const CohesionRanker ranker;
    cost -= ranker.score(text) * 0.1;
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
        const auto constructions = config ? config->constructions_for(coordination.relation)
                                          : std::vector<const Construction*>{};
        std::string text;
        if (!constructions.empty())
            text = constructions[static_cast<size_t>(variant < 0 ? 0 : variant) % constructions.size()]->pattern;
        else
            text = lang_choice(rules.part_of_templates, lang,
                               variant < 0 ? 0 : static_cast<size_t>(variant),
                               "{SUBJ} é formado {OBJS}.");
        std::vector<std::string> objects;
        const bool direct_object = text.find("contém") != std::string::npos ||
                                   text.find("inclui") != std::string::npos ||
                                   text.find("possui") != std::string::npos;
        for (auto id : coordination.conjuncts) {
            const auto& entity = entity_by_id(entities, id);
            objects.push_back(direct_object ? article_np(entity, lang)
                                            : prep_np(coordination.preposition, entity, lang, config));
        }
        auto subject = subject_surface(0);
        if (subject.empty()) subject = article_np(entity_by_id(entities, coordination.subject), lang);
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

    // Coordinate VPs that share the same subject:
    // "Ele transforma X, produz Y e usa Z." instead of one sentence per fact.
    std::string last_relation, current_subject;
    std::vector<std::string> predicates;
    std::string output_text;
    auto object_phrase = [&](const CausalStep& step) {
        return MorphologicalRealizer::contract_prepositions(
            morphology.realize_clause(step, "", lang, "[OBJ:ENT]"));
    };
    auto verb_phrase = [&](const CausalStep& step) {
        return MorphologicalRealizer::contract_prepositions(
            morphology.realize_clause(step, "", lang, "[PRED:VERB]"));
    };
    auto flush = [&]() {
        if (predicates.empty()) return;
        if (!output_text.empty()) output_text += ' ';
        for (size_t i = 0; i < predicates.size(); ++i) {
            if (i) output_text += (i + 1 == predicates.size()) ? " e " : ", ";
            output_text += predicates[i];
        }
        if (output_text.back() != '.' && output_text.back() != '!' && output_text.back() != '?')
            output_text += '.';
        predicates.clear();
    };
    for (size_t i = 0; i < sentence.propositions.size(); ++i) {
        const auto& proposition = sentence.propositions[i];
        auto subject = subject_surface(i);
        if (subject.empty()) subject = article_np(entity_by_id(entities, proposition.subject), lang);
        auto clause = MorphologicalRealizer::contract_prepositions(
            morphology.realize_clause(proposition.step, subject, lang));
        if (current_subject.empty() || proposition.step.src != current_subject) {
            flush();
            current_subject = proposition.step.src;
            predicates.push_back(capitalize(clause));
        } else if (proposition.step.rel == last_relation) {
            predicates.back() += " e " + object_phrase(proposition.step);
        } else {
            predicates.push_back(verb_phrase(proposition.step) + " " + object_phrase(proposition.step));
        }
        last_relation = proposition.step.rel;
    }
    flush();
    return output_text;
}

} // namespace

std::string DiscourseGenerator::realize(const DocumentParagraph& paragraph,
                                        const LanguageRules& rules,
                                        const DiscourseConfig* config) const {
    TextPlanner planner;
    const auto text_plan = planner.plan(paragraph, rules, config);
    if (text_plan.paragraphs.empty()) return {};
    GenerationSearch search;
    const size_t beam_width = std::max<size_t>(1, config ? config->beam_width : 4);
    std::vector<GenerationSearch::Candidate> beam{{"", 0.0}};
    for (const auto& sentence : text_plan.paragraphs.front().sentences) {
        std::vector<std::string> options;
        for (int variant = 0; variant < 3; ++variant) {
            auto realized = realize_sentence(sentence, text_plan.entities, rules, variant, config);
            if (realized.empty()) continue;
            realized = capitalize(realized);
            if (realized.find(realized) != std::string::npos &&
                realized.back() != '.' && realized.back() != '!' && realized.back() != '?')
                realized += '.';
            if (std::find(options.begin(), options.end(), realized) == options.end())
                options.push_back(std::move(realized));
        }
        if (options.empty()) continue;
        std::vector<GenerationSearch::Candidate> next;
        for (const auto& candidate : beam) {
            for (const auto& option : options) {
                std::string text = candidate.text;
                if (!text.empty()) text += ' ';
                text += option;
                next.push_back({text, search.score(text, config)});
            }
        }
        std::sort(next.begin(), next.end(),
                  [](const auto& a, const auto& b) { return a.cost < b.cost; });
        if (next.size() > beam_width) next.resize(beam_width);
        beam = std::move(next);
    }
    return beam.empty() ? std::string{} : capitalize_proper_nouns(beam.front().text, config);
}

} // namespace cognicore
