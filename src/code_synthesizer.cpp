#include "core/code_synthesizer.hpp"
#include "core/text_parser.hpp"
#include "core/symbolic_math.hpp"

#include <algorithm>
#include <optional>
#include <set>
#include <map>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <cctype>
#include <cstdlib>

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

std::string normalized_type(std::string type) {
    type.erase(std::remove_if(type.begin(), type.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }), type.end());
    for (const auto* qualifier : {"const", "volatile", "static", "inline"}) {
        size_t pos;
        while ((pos = type.find(qualifier)) != std::string::npos)
            type.erase(pos, std::char_traits<char>::length(qualifier));
    }
    type.erase(std::remove(type.begin(), type.end(), '&'), type.end());
    if (type.starts_with("std::")) type.erase(0, 5);
    return type;
}

bool wildcard_type(const std::string& type) {
    return type.empty() || type == "auto" || type == "any" || type == "Any" || type == "var" ||
           type == "object" || type == "unknown";
}

bool type_compatible(const std::string& value_type, const std::string& param_type) {
    if (wildcard_type(value_type) || wildcard_type(param_type)) return true;
    const auto value = normalized_type(value_type);
    const auto param = normalized_type(param_type);
    return value == param;
}

std::optional<std::string> default_literal(const std::string& raw_type, CodeLanguage lang) {
    const auto type = normalized_type(raw_type);
    if (lang == CodeLanguage::PYTHON) {
        if (type == "bool") return "False";
        if (type == "str" || type == "string") return "\"\"";
        if (type == "Any" || type == "any") return "None";
        static const char* numeric[] = {"int", "float", "complex", "size_t"};
        for (const auto* name : numeric) if (type == name) return "0";
        return std::nullopt;
    }
    if (lang == CodeLanguage::JAVASCRIPT || lang == CodeLanguage::TYPESCRIPT) {
        if (type == "boolean") return "false";
        if (type == "string") return "\"\"";
        if (type == "any" || type == "unknown" || type == "object") return "null";
        if (type == "number") return "0";
        return std::nullopt;
    }
    if (type == "bool" || type == "boolean") return "false";
    if (type == "string" || type == "String") return "\"\"";
    if (type == "char") return "'\\0'";
    static const char* numeric[] = {
        "int", "long", "short", "unsigned", "unsignedint", "size_t", "ssize_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t",
        "uint32_t", "uint64_t", "float", "double", "longdouble", "char32_t"
    };
    for (const auto* name : numeric) if (type == name) return "0";
    if (!type.empty() && type.back() == '*') return "nullptr";
    if (type.find("shared_ptr") != std::string::npos || type.find("unique_ptr") != std::string::npos)
        return "nullptr";
    return std::nullopt;
}

std::string lower_short_name(std::string name) {
    const auto at = name.find_last_of(":.");
    if (at != std::string::npos) name.erase(0, at + 1);
    return TextParser::utf8_tolower(name);
}

bool identifier_boundary(const std::string& text, size_t position, size_t length) {
    auto identifier = [](unsigned char ch) { return std::isalnum(ch) || ch == '_'; };
    const bool left = position == 0 || !identifier(static_cast<unsigned char>(text[position - 1]));
    const bool right = position + length >= text.size() ||
                       !identifier(static_cast<unsigned char>(text[position + length]));
    return left && right;
}

} // namespace

CodeSynthesizer::CodeSynthesizer() : pattern_tm_(make_code_tm_cfg()) {
    rules_.load_default();
    patterns_.load_default();
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
    res.pattern_votes = pattern_tm_.score_classes(feats.data());
    res.selected_pattern = static_cast<CodePatternTag>(std::max_element(res.pattern_votes.begin(), res.pattern_votes.end()) - res.pattern_votes.begin());
    const auto policy = pattern_tm_.export_sparse();
    for (size_t i = 0; i < policy.clauses.size(); ++i) {
        const auto& c = policy.clauses[i]; bool active = true;
        for (auto f : c.pos_literals) if (!(feats[f / 64] & (1ULL << (f % 64)))) active = false;
        for (auto f : c.neg_literals) if (feats[f / 64] & (1ULL << (f % 64))) active = false;
        if (active) res.fired_clauses.push_back(i);
    }
    const auto lower_task = TextParser::utf8_tolower(task_description);
    if (lower_task.find("corromp") != std::string::npos || lower_task.find("checksum") != std::string::npos ||
        lower_task.find("magic") != std::string::npos || lower_task.find("corrupt") != std::string::npos) {
        res.error = "O compositor atual não possui um padrão de validação binária verificável para esse formato. Preciso do esquema e da API de validação reais; não vou inventar campos, magic number ou checksum.";
        return res;
    }

    // Find matching symbols in CodeGraph for the target language. Both the
    // qualified name and the short name are accepted, with word boundaries so
    // "read" does not match "thread".
    std::vector<std::pair<size_t, CodeSymbol>> located;
    for (const auto& sym : graph.symbols()) {
        if (sym.lang != lang || sym.body_snippet.empty()) continue;
        const auto full_name = TextParser::utf8_tolower(sym.name);
        auto position = lower_task.find(full_name);
        if (position != std::string::npos && !identifier_boundary(lower_task, position, full_name.size()))
            position = std::string::npos;
        if (position == std::string::npos) {
            const auto short_name = lower_short_name(sym.name);
            position = lower_task.find(short_name);
            if (position == std::string::npos || !identifier_boundary(lower_task, position, short_name.size()))
                continue;
        }
        located.emplace_back(position, sym);
    }
    std::stable_sort(located.begin(), located.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    struct Requested {
        std::string name;
        std::vector<CodeSymbol> overloads;
    };
    std::vector<Requested> requested;
    for (auto& [position, symbol] : located) {
        if (requested.empty() || requested.back().name != symbol.name)
            requested.push_back({symbol.name, {}});
        requested.back().overloads.push_back(std::move(symbol));
    }
    if (requested.empty()) {
        // No workspace API matched. Use a code construction from the model
        // library, binding typed slots and validating the resulting AST.
        const auto language = std::string(code_lang_to_string(lang));
        if (const auto* pattern = patterns_.match(lower_task, language)) {
            auto code = pattern->template_code;
            auto replace_all = [](std::string& value, const std::string& from, const std::string& to) {
                size_t at;
                while ((at = value.find(from)) != std::string::npos) value.replace(at, from.size(), to);
            };
            replace_all(code, "{name}", pattern->name);
            if (code.find("{expression}") != std::string::npos) {
                const auto expression = SymbolicMath::extract_expression(task_description, rules_);
                if (expression.empty()) {
                    res.error = "A construção requer uma expressão aritmética válida.";
                    return res;
                }
                replace_all(code, "{expression}", expression);
            }
            if (TreeSitterCodeParser::validate_syntax(code, lang)) {
                res.code = code;
                res.syntax_valid = true;
                res.selected_pattern = CodePatternTag::TRANSFORM_VECTOR;
                res.repair_notes = "construção de código do modelo: " + pattern->id;
                return res;
            }
        }
        res.error = "Nenhuma API do workspace ou construção de código corresponde ao pedido.";
        return res;
    }

    const bool patch_request = lower_task.find("patch") != std::string::npos ||
        lower_task.find("refator") != std::string::npos ||
        lower_task.find("corrija") != std::string::npos ||
        lower_task.find("modifique") != std::string::npos ||
        lower_task.find("substitua") != std::string::npos ||
        lower_task.find("rewrite") != std::string::npos ||
        lower_task.find("fix ") != std::string::npos;
    CodeSymbol patch_target;
    if (patch_request && requested.size() == 1) {
        CodeSymbol target = requested.front().overloads.front();
        patch_target = target;
        requested.clear();
        // A patch is a new implementation of the target composed from the APIs
        // the target already calls. This is structural rather than a free-text
        // diff: the generated function is validated by Tree-sitter.
        for (const auto& edge : graph.edges()) {
            if (edge.src != target.id || edge.type != CodeEdgeType::CALLS) continue;
            if (edge.dst >= graph.symbols().size()) continue;
            const auto& callee = graph.symbols()[edge.dst];
            if (callee.body_snippet.empty() || callee.lang != lang || callee.name == target.name) continue;
            if (std::none_of(requested.begin(), requested.end(), [&](const Requested& value) {
                    return value.overloads.front().name == callee.name;
                }))
                requested.push_back({callee.name, {callee}});
        }
        if (requested.empty()) {
            res.error = "O alvo do patch não possui chamadas internas indexadas e compatíveis; não vou inventar um corpo arbitrário.";
            return res;
        }
        res.is_patch = true;
        res.patch_target = target.name;
    }

    struct Value { std::string expr, type; };
    struct Call { std::string expression, binding; bool returns = false; };
    struct Receiver { std::string name, owner; };
    std::vector<Value> pool;
    std::vector<std::pair<std::string, std::string>> input_params;
    if (patch_request && res.is_patch) {
        for (size_t i = 0; i < patch_target.param_types.size(); ++i) {
            const auto name = i < patch_target.param_names.size() && !patch_target.param_names[i].empty()
                ? patch_target.param_names[i] : "arg" + std::to_string(i + 1);
            input_params.emplace_back(name, patch_target.param_types[i]);
            pool.push_back({name, patch_target.param_types[i]});
        }
    } else {
        std::string entry_type;
        for (const auto& group : requested) {
            for (const auto& overload : group.overloads)
                if (!overload.param_types.empty()) { entry_type = overload.param_types.front(); break; }
            if (!entry_type.empty()) break;
        }
        if (!entry_type.empty()) {
            input_params.emplace_back("input", entry_type);
            pool.push_back({"input", entry_type});
        }
    }

    auto try_bind = [&](const CodeSymbol& symbol) -> std::optional<std::vector<std::string>> {
        std::vector<std::string> arguments;
        std::vector<bool> consumed(pool.size(), false);
        for (const auto& param : symbol.param_types) {
            int chosen_value = -1;
            for (int k = static_cast<int>(pool.size()) - 1; k >= 0; --k) {
                if (consumed[static_cast<size_t>(k)]) continue;
                if (type_compatible(pool[static_cast<size_t>(k)].type, param)) { chosen_value = k; break; }
            }
            if (chosen_value >= 0) {
                consumed[static_cast<size_t>(chosen_value)] = true;
                arguments.push_back(pool[static_cast<size_t>(chosen_value)].expr);
            } else if (auto literal = default_literal(param, lang)) {
                arguments.push_back(*literal);
            } else {
                return std::nullopt;
            }
        }
        return arguments;
    };

    std::vector<Call> calls;
    std::vector<CodeSymbol> chosen;
    std::vector<Receiver> receivers;
    bool produces_value = false;
    auto receiver_for = [&](const std::string& owner) {
        for (const auto& receiver : receivers) if (receiver.owner == owner) return receiver.name;
        Receiver receiver{"receiver_" + std::to_string(receivers.size()), owner};
        receivers.push_back(receiver);
        return receiver.name;
    };
    for (size_t group_index = 0; group_index < requested.size(); ++group_index) {
        const auto& group = requested[group_index];
        std::optional<std::vector<std::string>> arguments;
        const CodeSymbol* selected = nullptr;
        for (const auto& candidate : group.overloads) {
            if (auto bound = try_bind(candidate)) {
                arguments = std::move(bound);
                selected = &candidate;
                break;
            }
        }
        if (!selected || !arguments) {
            res.error = "Não há valor compatível para os parâmetros de " + group.name +
                        ". Informe uma cadeia de tipos compatível ou uma API que produza esse tipo.";
            return res;
        }
        CodeSymbol symbol = *selected;
        std::string callee = symbol.name;
        if (symbol.is_method && !symbol.is_static) {
            auto at = symbol.name.rfind("::");
            if (at == std::string::npos) at = symbol.name.rfind('.');
            const auto original_method = at == std::string::npos ? symbol.name : symbol.name.substr(at + (symbol.name[at] == ':' ? 2 : 1));
            callee = receiver_for(symbol.owner) + "." + original_method;
        }
        std::ostringstream expression;
        expression << callee << "(";
        for (size_t a = 0; a < arguments->size(); ++a) expression << (a ? ", " : "") << (*arguments)[a];
        expression << ")";
        const auto normalized_return = normalized_type(symbol.return_type);
        const bool returns = !symbol.return_type.empty() && normalized_return != "void";
        Call statement;
        statement.expression = expression.str();
        if (returns) {
            statement.binding = "step" + std::to_string(group_index + 1);
            pool.push_back({statement.binding, symbol.return_type});
            produces_value = true;
        }
        calls.push_back(std::move(statement));
        chosen.push_back(std::move(symbol));
    }
    if (calls.empty()) {
        res.error = "Não foi possível compor uma chamada.";
        return res;
    }
    if (produces_value && !calls.back().binding.empty()) calls.back().returns = true;
    for (const auto& symbol : chosen) {
        res.used_symbols.push_back(symbol.name);
        res.symbol_ids.push_back(symbol.id);
    }

    const bool guarded = res.selected_pattern == CodePatternTag::SAFE_TRY_CATCH;
    std::ostringstream out;
    std::set<std::string> includes;
    for (const auto& symbol : chosen)
        for (const auto& inc : symbol.includes_or_imports) includes.insert(inc);
    if (res.is_patch) for (const auto& inc : patch_target.includes_or_imports) includes.insert(inc);
    for (const auto& inc : includes) out << inc << "\n";
    std::string return_type = chosen.back().return_type.empty() ? "void" : chosen.back().return_type;
    std::string function_name = "synthesized_pipeline";
    if (res.is_patch) {
        return_type = patch_target.return_type.empty() ? "void" : patch_target.return_type;
        function_name = "patched_" + lower_short_name(patch_target.name);
    }
    auto parameter_declaration = [&](const std::pair<std::string, std::string>& param, bool with_name) {
        return param.second + (with_name ? " " + param.first : std::string{});
    };
    if (lang == CodeLanguage::CPP) {
        for (const auto& symbol : chosen) {
            if (symbol.is_method || symbol.name.find("::") != std::string::npos) continue;
            out << symbol.return_type << " " << symbol.name << "(";
            for (size_t i = 0; i < symbol.param_types.size(); ++i) {
                if (i) out << ", ";
                out << symbol.param_types[i];
                if (i < symbol.param_names.size() && !symbol.param_names[i].empty())
                    out << " " << symbol.param_names[i];
            }
            if (symbol.param_types.empty()) out << "void";
            out << ");\n";
        }
        out << "\n" << return_type << " " << function_name << "(";
        for (size_t i = 0; i < input_params.size(); ++i) {
            if (i) out << ", ";
            out << parameter_declaration(input_params[i], true);
        }
        out << ") {\n";
        if (guarded) out << "    try {\n";
        const std::string indent = guarded ? "        " : "    ";
        for (const auto& receiver : receivers) out << indent << receiver.owner << " " << receiver.name << ";\n";
        for (const auto& call : calls) {
            if (call.returns) out << indent << "return " << call.expression << ";\n";
            else if (call.binding.empty()) out << indent << call.expression << ";\n";
            else out << indent << "auto " << call.binding << " = " << call.expression << ";\n";
        }
        if (guarded) out << "    } catch (...) {\n        throw;\n    }\n";
        out << "}\n";
    } else if (lang == CodeLanguage::PYTHON) {
        std::map<std::string, std::vector<std::string>> imports;
        for (const auto& symbol : chosen) {
            const auto module = std::filesystem::path(symbol.file_path).stem().string();
            if (!module.empty()) imports[module].push_back(symbol.is_method ? symbol.owner : symbol.name);
        }
        for (const auto& [module, names] : imports) {
            out << "from " << module << " import ";
            for (size_t i = 0; i < names.size(); ++i) out << (i ? ", " : "") << names[i];
            out << "\n";
        }
        out << "\ndef " << function_name << "(";
        for (size_t i = 0; i < input_params.size(); ++i) {
            if (i) out << ", ";
            out << input_params[i].first;
        }
        out << "):\n";
        if (guarded) out << "    try:\n";
        const std::string indent = guarded ? "        " : "    ";
        for (const auto& receiver : receivers) out << indent << receiver.name << " = " << receiver.owner << "()\n";
        for (const auto& call : calls) {
            if (call.returns) out << indent << "return " << call.expression << "\n";
            else if (call.binding.empty()) out << indent << call.expression << "\n";
            else out << indent << call.binding << " = " << call.expression << "\n";
        }
        if (guarded) out << "    except Exception:\n        raise\n";
        if (calls.empty()) out << "    pass\n";
    } else {
        for (const auto& symbol : chosen) {
            if (symbol.is_method) {
                out << "import { " << symbol.owner << " } from './" <<
                    std::filesystem::path(symbol.file_path).filename().string() << "';\n";
            } else {
                out << "import { " << symbol.name << " } from './" <<
                    std::filesystem::path(symbol.file_path).filename().string() << "';\n";
            }
        }
        out << "function " << function_name << "(";
        for (size_t i = 0; i < input_params.size(); ++i) {
            if (i) out << ", ";
            out << input_params[i].first;
        }
        out << ") {\n";
        for (const auto& receiver : receivers) out << "  const " << receiver.name << " = new " << receiver.owner << "();\n";
        for (const auto& call : calls) {
            if (call.returns) out << "  return " << call.expression << ";\n";
            else if (call.binding.empty()) out << "  " << call.expression << ";\n";
            else out << "  const " << call.binding << " = " << call.expression << ";\n";
        }
        out << "}\n";
    }
    res.code = out.str();
    res.syntax_valid = TreeSitterCodeParser::validate_syntax(res.code, lang);
    if (!res.syntax_valid) { res.code.clear(); res.error = "A validação AST rejeitou a composição."; }
    return res;
}
CodeSynthesisResult CodeSynthesizer::synthesize_tested(const std::string& task_description,
                                                       CodeLanguage lang,
                                                       const CodeGraph& graph,
                                                       const Codebook& codebook,
                                                       const std::vector<CodeTestCase>& tests,
                                                       const std::string& workspace) const {
    CodeSynthesisResult baseline = synthesize(task_description, lang, graph, codebook);
    std::vector<std::string> candidates;
    if (baseline.syntax_valid) candidates.push_back(baseline.code);

    CodeSynthesisResult best = baseline;
    best.iterations = 0;
    best.generated_tests.clear();
    for (const auto& test : tests)
        best.generated_tests.push_back("f(" + test.input + ") == " + test.expected);
    if (tests.empty()) return best;

    std::error_code ec;
    const auto temporary = workspace.empty()
        ? std::filesystem::temp_directory_path(ec) : std::filesystem::path(workspace);
    if (ec) { best.error = "Sem diretório para executar os testes."; return best; }
    const auto candidate_path = temporary / "__cognicore_candidate.py";
    const auto driver_path = temporary / "__cognicore_test_driver.py";

    for (size_t i = 0; i < candidates.size(); ++i) {
        ++best.iterations;
        const auto& candidate = candidates[i];
        if (!TreeSitterCodeParser::validate_syntax(candidate, lang)) continue;
        if (lang != CodeLanguage::PYTHON) {
            best.code = candidate;
            best.syntax_valid = true;
            best.repair_notes = "Testes executáveis apenas para Python; candidato validado por Tree-sitter.";
            return best;
        }
        {
            std::ofstream out(candidate_path);
            if (!out) { best.error = "Falha ao escrever candidato."; return best; }
            out << candidate;
        }
        {
            std::ofstream out(driver_path);
            if (!out) { best.error = "Falha ao escrever testes."; return best; }
            out << "from __cognicore_candidate import synthesized_pipeline\n";
            for (const auto& test : tests)
                out << "assert synthesized_pipeline(" << test.input << ") == (" << test.expected << ")\n";
            out << "print('ok')\n";
        }
        std::ostringstream command;
        command << "cd \"" << temporary.string() << "\" && python3 \"" << driver_path.filename().string()
                << "\" >/dev/null 2>&1";
        if (std::system(command.str().c_str()) == 0) {
            best.code = candidate;
            best.syntax_valid = true;
            best.error.clear();
            best.repair_notes = "CEGIS: candidato #" + std::to_string(i + 1) + " passou em todos os testes.";
            std::filesystem::remove(candidate_path, ec);
            std::filesystem::remove(driver_path, ec);
            return best;
        }
        best.repair_notes += "Candidato #" + std::to_string(i + 1) + " falhou nos testes. ";
    }
    best.syntax_valid = false;
    best.code.clear();
    if (best.error.empty()) best.error = "Nenhum candidato passou nos testes.";
    std::filesystem::remove(candidate_path, ec);
    std::filesystem::remove(driver_path, ec);
    return best;
}

} // namespace cognicore
