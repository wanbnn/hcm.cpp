#pragma once

#include "core/document_planner.hpp"
#include "core/language_rules.hpp"

#include <string>

namespace cognicore {

/// Turns a paragraph of graph-backed sentence plans into cohesive prose:
/// coordinates objects, pronominalizes repeated subjects and uses sequence
/// ordinals. It uses only model-owned rules/templates for surface wording.
class NaturalProseRealizer {
public:
    std::string realize(const DocumentParagraph& paragraph, const LanguageRules& rules) const;
};

} // namespace cognicore
