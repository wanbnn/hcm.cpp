#pragma once

#include "core/realizer.hpp"

#include <string>
#include <vector>

namespace cognicore {

/// Level-A (0% neural) Surface Realizer inspired by SimpleNLG / Meaning-Text Theory.
/// Performs discourse planning, anaphora/pronominal substitution, verbal conjugation,
/// nominal gender/number agreement, and preposition contractions (PT-BR & EN).
class MorphologicalRealizer : public ISurfaceRealizer {
public:
    std::string realize(const SemanticPlan& plan) const override;

    static std::string conjugate_verb(const std::string& lemma,
                                      GramTense tense,
                                      GramNumber number,
                                      const std::string& lang = "pt");

    static std::string definite_article(GramGender gender,
                                        GramNumber number,
                                        const std::string& lang = "pt");

    static std::string contract_prepositions(const std::string& text);

    /// Unifies typed construction slots with the inflected clause.
    std::string realize_clause(const CausalStep& step,
                               const std::string& subject_surface,
                               const std::string& lang,
                               const std::string& pattern = "[SUBJ:ENT] [PRED:VERB] [OBJ:ENT]") const;
};

} // namespace cognicore
