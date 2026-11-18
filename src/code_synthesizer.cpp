#include "core/code_synthesizer.hpp"
#include "core/text_parser.hpp"

#include <algorithm>
#include <set>
#include <map>
#include <sstream>
#include <filesystem>
#include <cctype>

namespace cognicore {

namespace {

TsetlinConfig make_code_tm_cfg() {
    TsetlinConfig cfg;
    cfg.num_features = VSA_BITS + 64;
    cfg.num_classes = static_cast<size_t>(CodePatternTag::COUNT);
    cfg.clauses_per_class = 20;
    cfg.n_states = 30;
    cfg.margin_T = 10;
    cfg.s_param = 4.0f;
    return cfg;
}


} // namespace

CodeSynthesizer::CodeSynthesizer() : pattern_tm_(make_code_tm_cfg()) {
    // Baseline structural pattern training
    train_patterns({
        {"ler arquivo de texto", CodePatternTag::READ_AND_PROCESS},
        {"read file and parse", CodePatternTag::READ_AND_PROCESS},
        {"carregar arquivo e processar", CodePatternTag::READ_AND_PROCESS},
        {"transformar vetor de numeros", CodePatternTag::TRANSFORM_VECTOR},
        {"filtrar e somar lista", CodePatternTag::TRANSFORM_VECTOR},
        {"process vector items", CodePatternTag::TRANSFORM_VECTOR},
        {"combinar duas funcoes", CodePatternTag::CALL_CHAIN},
        {"encadear chamadas da api", CodePatternTag::CALL_CHAIN},
        {"tratar erro com try catch", CodePatternTag::SAFE_TRY_CATCH},
        {"executar com seguranca e excecao", CodePatternTag::SAFE_TRY_CATCH}
    }, 15);
}

std::vector<uint64_t> CodeSynthesizer::encode_task_features(const std::string& task, CodeLanguage lang) const {
    const size_t n_words = (VSA_BITS + 64 + 63) / 64;
    std::vector<uint64_t> feats(n_words, 0ULL);

    TextParser tp;
    auto toks = tp.tokenize(task);
    std::vector<HyperVector> hvs;
    for (const auto& t : toks) hvs.push_back(HyperVector::from_token(t));
    HyperVector bag = HyperVector::bundle(hvs);
    for (size_t w = 0; w < VSA_WORDS; ++w) feats[w] = bag.words[w];

    uint64_t ctx = 0;
    ctx |= (1ULL << static_cast<uint8_t>(lang));
    for (const auto& t : toks) {
        if (t == "arquivo" || t == "file" || t == "ler" || t == "read") ctx |= (1ULL << 8);
        if (t == "vetor" || t == "vector" || t == "lista" || t == "list" || t == "soma") ctx |= (1ULL << 9);
        if (t == "erro" || t == "try" || t == "catch" || t == "excecao") ctx |= (1ULL << 10);
    }
    feats[VSA_WORDS] = ctx;
    return feats;
}

void CodeSynthesizer::train_patterns(const std::vector<std::pair<std::string, CodePatternTag>>& examples,
                                     size_t epochs) {
    if (examples.empty()) return;
    std::vector<std::vector<uint64_t>> X;
    std::vector<size_t> y;
    for (const auto& [txt, tag] : examples) {
        X.push_back(encode_task_features(txt, CodeLanguage::CPP));
        y.push_back(static_cast<size_t>(tag));
    }
    pattern_tm_.fit(X, y, epochs);
}

CodeSynthesisResult CodeSynthesizer::synthesize(const std::string& task_description,
                                                CodeLanguage lang,
                                                const CodeGraph& graph,
                                                const Codebook& codebook) const {
    (void)codebook;
    CodeSynthesisResult res;

    auto feats = encode_task_features(task_description, lang);
    res.selected_pattern = static_cast<CodePatternTag>(pattern_tm_.predict(feats.data()));

    // Find matching symbols in CodeGraph for the target language
    TextParser tp;
    auto tokens = tp.extract_content_tokens(task_description);

    std::vector<CodeSymbol> matched;
    for (const auto& sym : graph.symbols()) {
        if (sym.lang != lang || sym.body_snippet.empty() || (sym.name.find("::") != std::string::npos || sym.name.find('.') != std::string::npos)) continue;
        std::string s_low = TextParser::utf8_tolower(sym.name);
        bool hit = false;
        for (const auto& t : tokens) {
            if (s_low.find(t) != std::string::npos) {
                hit = true;
                break;
            }
        }
        if (hit) matched.push_back(sym);
    }

    // Fallback to VSA query if lexical match found fewer than 2 symbols
    if (matched.size() < 2) {
        std::vector<HyperVector> qparts;
        qparts.push_back(HyperVector::from_token("ROLE_LANG").bind(HyperVector::from_token(code_lang_to_string(lang))));
        for (const auto& t : tokens) {
            qparts.push_back(HyperVector::from_token("ROLE_SHORT").bind(HyperVector::from_token(t)));
            qparts.push_back(HyperVector::from_token("ROLE_NAME").bind(HyperVector::from_token(t)));
        }
        auto vsa_hits = graph.query_vsa(HyperVector::bundle(qparts), 4);
        for (const auto& s : vsa_hits) {
            if (s.lang != lang || s.body_snippet.empty() || (s.name.find("::") != std::string::npos || s.name.find('.') != std::string::npos)) continue;
            bool exists = std::any_of(matched.begin(), matched.end(),
                                      [&](const CodeSymbol& m) { return m.name == s.name; });
            if (!exists) matched.push_back(s);
        }
    }

    auto normalize_type = [](std::string type) {
        for (const auto* qualifier : {"const", "volatile"}) {
            size_t pos;
            while ((pos = type.find(qualifier)) != std::string::npos) type.erase(pos, std::char_traits<char>::length(qualifier));
        }
        type.erase(std::remove_if(type.begin(), type.end(), [](unsigned char c) { return std::isspace(c) || c == '&'; }), type.end());
        return type;
    };
    auto compatible = [&](const CodeSymbol& first, const CodeSymbol& second) {
        if (first.param_types.size() != 1 || second.param_types.size() != 1) return false;
        auto ret = normalize_type(first.return_type), arg = normalize_type(second.param_types[0]);
        return ret != "void" && (ret == arg || (lang != CodeLanguage::CPP &&
            (ret == "Any" || arg == "Any" || ret == "any" || arg == "any")));
    };
    std::vector<CodeSymbol> chosen;
    for (const auto& first : matched) {
        for (const auto& second : matched) {
            if (first.id != second.id && compatible(first, second)) { chosen = {first, second}; break; }
        }
        if (!chosen.empty()) break;
    }
    if (chosen.empty() && matched.size() == 1 && matched[0].param_types.size() == 1) chosen = {matched[0]};
    if (chosen.empty()) {
        res.error = "Não há uma composição unária com tipos compatíveis no grafo do projeto.";
        return res;
    }
    for (const auto& symbol : chosen) res.used_symbols.push_back(symbol.name);

    // Typed intermediate AST. The printer consumes nodes, never task text.
    struct CallNode { std::string callee, argument, binding; bool returns; };
    std::vector<CallNode> body;
    body.push_back({chosen[0].name, "input", chosen.size() > 1 ? "step1" : "", chosen.size() == 1});
    if (chosen.size() > 1) body.push_back({chosen[1].name, "step1", "", true});
    const bool guarded = res.selected_pattern == CodePatternTag::SAFE_TRY_CATCH;
    std::ostringstream out;
    std::set<std::string> includes;
    for (const auto& symbol : chosen)
        for (const auto& inc : symbol.includes_or_imports) includes.insert(inc);
    for (const auto& inc : includes) out << inc << "\n";
    if (lang == CodeLanguage::CPP) {
        // Definitions in a .cpp may have no public header; declare the actual APIs.
        for (const auto& symbol : chosen)
            out << symbol.return_type << " " << symbol.name << "(" << symbol.param_types[0] << ");\n";
        out << "\n" << chosen.back().return_type << " synthesized_pipeline("
            << chosen.front().param_types[0] << " input) {\n";
        if (guarded) out << "    try {\n";
        for (const auto& node : body)
            out << (guarded ? "        " : "    ") << (node.returns ? "return " : "auto " + node.binding + " = ")
                << node.callee << "(" << node.argument << ");\n";
        if (guarded) out << "    } catch (...) {\n        throw;\n    }\n";
        out << "}\n";
    } else if (lang == CodeLanguage::PYTHON) {
        std::map<std::string, std::vector<std::string>> imports;
        for (const auto& symbol : chosen) {
            const auto module = std::filesystem::path(symbol.file_path).stem().string();
            if (!module.empty()) imports[module].push_back(symbol.name);
        }
        for (const auto& [module, names] : imports) {
            out << "from " << module << " import ";
            for (size_t i = 0; i < names.size(); ++i) out << (i ? ", " : "") << names[i];
            out << "\n";
        }
        out << "\ndef synthesized_pipeline(input):\n";
        if (guarded) out << "    try:\n";
        for (const auto& node : body)
            out << (guarded ? "        " : "    ") << (node.returns ? "return " : node.binding + " = ")
                << node.callee << "(" << node.argument << ")\n";
        if (guarded) out << "    except Exception:\n        raise\n";
    } else {
        for (const auto& symbol : chosen)
            out << "import { " << symbol.name << " } from './" << std::filesystem::path(symbol.file_path).filename().string() << "';\n";
        out << "function synthesizedPipeline(input) {\n";
        for (const auto& node : body)
            out << "  " << (node.returns ? "return " : "const " + node.binding + " = ")
                << node.callee << "(" << node.argument << ");\n";
        out << "}\n";
    }
    res.code = out.str();
    res.syntax_valid = TreeSitterCodeParser::validate_syntax(res.code, lang);
    if (!res.syntax_valid) { res.code.clear(); res.error = "A validação AST rejeitou a composição."; }
    return res;
}

} // namespace cognicore
