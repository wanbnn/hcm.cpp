#include "core/code_ast.hpp"

#include <tree_sitter/api.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <set>

extern "C" {
const TSLanguage* tree_sitter_cpp(void);
const TSLanguage* tree_sitter_python(void);
const TSLanguage* tree_sitter_javascript(void);
const TSLanguage* tree_sitter_typescript(void);
}

namespace fs = std::filesystem;

namespace cognicore {

namespace {

const TSLanguage* get_ts_lang(CodeLanguage lang) {
    switch (lang) {
        case CodeLanguage::CPP:        return tree_sitter_cpp();
        case CodeLanguage::PYTHON:     return tree_sitter_python();
        case CodeLanguage::JAVASCRIPT: return tree_sitter_javascript();
        case CodeLanguage::TYPESCRIPT: return tree_sitter_typescript();
        default:                       return nullptr;
    }
}

std::string node_text(TSNode node, const std::string& src) {
    if (ts_node_is_null(node)) return "";
    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);
    if (start >= src.size() || end > src.size() || start >= end) return "";
    return src.substr(start, end - start);
}

bool has_missing_or_error(TSNode node) {
    if (ts_node_is_error(node) || ts_node_is_missing(node)) return true;
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        if (has_missing_or_error(ts_node_child(node, i))) return true;
    }
    return false;
}

std::string canonical_ast(TSNode node, size_t depth = 0) {
    if (depth > 64) return "...";
    std::string out = ts_node_type(node);
    const auto count = ts_node_named_child_count(node);
    if (count) {
        out += "(";
        for (uint32_t i = 0; i < count; ++i) {
            if (i) out += ",";
            out += canonical_ast(ts_node_named_child(node, i), depth + 1);
        }
        out += ")";
    }
    return out;
}
std::string documentation(TSNode node, TSNode body, const std::string& src) {
    const auto previous = ts_node_prev_named_sibling(node);
    if (!ts_node_is_null(previous) && std::string(ts_node_type(previous)) == "comment") return node_text(previous, src);
    if (!ts_node_is_null(body) && ts_node_named_child_count(body)) {
        const auto first = ts_node_named_child(body, 0);
        const auto text = node_text(first, src);
        if (text.starts_with("\"\"\"") || text.starts_with("'''")) return text;
    }
    return "";
}

std::string enclosing_class(TSNode node, const std::string& src) {
    for (auto parent = ts_node_parent(node); !ts_node_is_null(parent); parent = ts_node_parent(parent)) {
        const std::string kind = ts_node_type(parent);
        if (kind == "class_specifier" || kind == "struct_specifier" || kind == "class_definition" || kind == "class_declaration") {
            auto name = ts_node_child_by_field_name(parent, "name", 4);
            if (!ts_node_is_null(name)) return node_text(name, src);
        }
    }
    return "";
}

void collect_calls(TSNode node, const std::string& src, std::vector<std::string>& out_calls) {
    const char* type = ts_node_type(node);
    if (std::string(type) == "call_expression" || std::string(type) == "call") {
        TSNode fn = ts_node_child_by_field_name(node, "function", 8);
        if (!ts_node_is_null(fn)) {
            std::string fn_text = node_text(fn, src);
            // Strip leading object/namespace prefix if needed, also keep full
            if (!fn_text.empty()) out_calls.push_back(fn_text);
        }
    }
    uint32_t count = ts_node_child_count(node);
    for (uint32_t i = 0; i < count; ++i) {
        collect_calls(ts_node_child(node, i), src, out_calls);
    }
}

std::string extract_cpp_func_name(TSNode decl, const std::string& src) {
    if (ts_node_is_null(decl)) return "";
    const char* t = ts_node_type(decl);
    std::string ts(t);
    if (ts == "function_declarator") {
        TSNode d = ts_node_child_by_field_name(decl, "declarator", 10);
        if (!ts_node_is_null(d)) return extract_cpp_func_name(d, src);
    } else if (ts == "pointer_declarator" || ts == "reference_declarator") {
        uint32_t n = ts_node_named_child_count(decl);
        for (uint32_t i = 0; i < n; ++i) {
            std::string r = extract_cpp_func_name(ts_node_named_child(decl, i), src);
            if (!r.empty()) return r;
        }
    } else if (ts == "identifier" || ts == "field_identifier" || ts == "qualified_identifier") {
        return node_text(decl, src);
    }
    return "";
}

void extract_cpp_params(TSNode decl,
                        const std::string& src,
                        std::vector<std::string>& p_types,
                        std::vector<std::string>& p_names) {
    const char* t = ts_node_type(decl);
    std::string ts(t);
    if (ts == "function_declarator") {
        TSNode plist = ts_node_child_by_field_name(decl, "parameters", 10);
        if (!ts_node_is_null(plist)) {
            uint32_t n = ts_node_named_child_count(plist);
            for (uint32_t i = 0; i < n; ++i) {
                TSNode param = ts_node_named_child(plist, i);
                if (std::string(ts_node_type(param)) == "parameter_declaration" || std::string(ts_node_type(param)) == "optional_parameter_declaration") {
                    TSNode pt = ts_node_child_by_field_name(param, "type", 4);
                    TSNode pd = ts_node_child_by_field_name(param, "declarator", 10);
                    std::string type_s = ts_node_is_null(pt) ? "auto" : node_text(pt, src);
                    std::string name_s = ts_node_is_null(pd) ? ("arg" + std::to_string(i)) : node_text(pd, src);
                    // Clean pointer/ref from name into type if needed
                    while (!name_s.empty() && (name_s[0] == '&' || name_s[0] == '*')) {
                        type_s += name_s[0];
                        name_s.erase(name_s.begin());
                    }
                    // Type qualifiers belong to the type, not the identifier.
                    if (node_text(param, src).starts_with("const ") && !type_s.starts_with("const ")) type_s = "const " + type_s;
                    p_types.push_back(type_s);
                    p_names.push_back(name_s);
                }
            }
        }
    } else {
        uint32_t n = ts_node_named_child_count(decl);
        for (uint32_t i = 0; i < n; ++i) {
            extract_cpp_params(ts_node_named_child(decl, i), src, p_types, p_names);
        }
    }
}

std::string short_symbol_name(const std::string& s) {
    size_t pos = s.rfind("::");
    if (pos != std::string::npos) return s.substr(pos + 2);
    pos = s.rfind('.');
    if (pos != std::string::npos) return s.substr(pos + 1);
    pos = s.rfind("->");
    if (pos != std::string::npos) return s.substr(pos + 2);
    return s;
}

} // namespace

const char* code_lang_to_string(CodeLanguage lang) noexcept {
    switch (lang) {
        case CodeLanguage::CPP:        return "cpp";
        case CodeLanguage::PYTHON:     return "py";
        case CodeLanguage::JAVASCRIPT: return "js";
        case CodeLanguage::TYPESCRIPT: return "ts";
        default:                       return "unknown";
    }
}

CodeLanguage string_to_code_lang(const std::string& s) noexcept {
    if (s == "cpp" || s == "c++" || s == "cc" || s == "h" || s == "hpp") return CodeLanguage::CPP;
    if (s == "py" || s == "python") return CodeLanguage::PYTHON;
    if (s == "ts" || s == "typescript") return CodeLanguage::TYPESCRIPT;
    if (s == "js" || s == "javascript") return CodeLanguage::JAVASCRIPT;
    return CodeLanguage::UNKNOWN;
}

HyperVector CodeGraph::encode_code_vsa(CodeLanguage lang,
                                       const std::string& name,
                                       const std::string& ret_type,
                                       const std::vector<std::string>& param_types,
                                       const std::vector<std::string>& calls) {
    std::vector<HyperVector> parts;
    parts.push_back(HyperVector::from_token("ROLE_LANG").bind(HyperVector::from_token(code_lang_to_string(lang))));
    if (!name.empty()) {
        parts.push_back(HyperVector::from_token("ROLE_NAME").bind(HyperVector::from_token(short_symbol_name(name))));
        parts.push_back(HyperVector::from_token("ROLE_SHORT").bind(HyperVector::from_token(short_symbol_name(name))));
    }
    if (!ret_type.empty()) {
        parts.push_back(HyperVector::from_token("ROLE_RET").bind(HyperVector::from_token(ret_type)));
    }
    for (const auto& pt : param_types) {
        parts.push_back(HyperVector::from_token("ROLE_PARAM").bind(HyperVector::from_token(pt)));
    }
    for (const auto& c : calls) {
        parts.push_back(HyperVector::from_token("ROLE_CALL").bind(HyperVector::from_token(short_symbol_name(c))));
    }
    return HyperVector::bundle(parts);
}

uint32_t CodeGraph::add_or_update_symbol(CodeSymbol sym) {
    auto it = name_to_id_.find(sym.name);
    if (it != name_to_id_.end()) {
        const uint32_t id = it->second;
        sym.id = id;
        // Preserve non-empty fields if updating declaration with definition
        if (sym.body_snippet.empty()) sym.body_snippet = symbols_[id].body_snippet;
        if (sym.return_type.empty()) sym.return_type = symbols_[id].return_type;
        if (sym.param_types.empty()) {
            sym.param_types = symbols_[id].param_types;
            sym.param_names = symbols_[id].param_names;
        }
        symbols_[id] = std::move(sym);
        return id;
    }
    const uint32_t id = static_cast<uint32_t>(symbols_.size());
    sym.id = id;
    name_to_id_.emplace(sym.name, id);
    symbols_.push_back(std::move(sym));
    return id;
}

uint32_t CodeGraph::find_symbol(const std::string& name) const {
    auto it = name_to_id_.find(name);
    return (it != name_to_id_.end()) ? it->second : UINT32_MAX;
}

std::vector<uint32_t> CodeGraph::find_symbols_by_suffix(const std::string& short_name) const {
    std::vector<uint32_t> out;
    for (const auto& s : symbols_) {
        if (s.name == short_name || short_symbol_name(s.name) == short_name) {
            out.push_back(s.id);
        }
    }
    return out;
}

void CodeGraph::add_edge(const std::string& src_name, const std::string& dst_name, CodeEdgeType type) {
    uint32_t s = find_symbol(src_name);
    if (s == UINT32_MAX) {
        CodeSymbol tmp;
        tmp.name = src_name;
        s = add_or_update_symbol(std::move(tmp));
    }
    uint32_t d = find_symbol(dst_name);
    if (d == UINT32_MAX) {
        CodeSymbol tmp;
        tmp.name = dst_name;
        d = add_or_update_symbol(std::move(tmp));
    }
    for (const auto& e : edges_) if (e.src == s && e.dst == d && e.type == type) return;
    edges_.push_back(CodeEdge{s, d, type});
}

std::vector<CodeSymbol> CodeGraph::find_callers(const std::string& target_name) const {
    std::vector<CodeSymbol> callers;
    const auto target_short = short_symbol_name(target_name);
    std::vector<bool> seen(symbols_.size(), false);
    auto binding_type = [&](const std::string& scope, const std::string& name) {
        auto id = find_symbol("@binding:" + scope + "::" + name);
        if (id == UINT32_MAX) return std::string{};
        auto type = symbols_[id].return_type;
        if (type.starts_with("const ")) type.erase(0, 6);
        while (!type.empty() && (type.back() == '&' || type.back() == '*' || type.back() == ' ')) type.pop_back();
        return type;
    };
    for (const auto& edge : edges_) {
        if (edge.type != CodeEdgeType::CALLS) continue;
        const auto& source = symbols_[edge.src].name;
        auto called = symbols_[edge.dst].name;
        const auto owner_at = source.rfind("::");
        const auto owner = owner_at == std::string::npos ? "" : source.substr(0, owner_at);
        auto normalized = called;
        for (size_t at; (at = normalized.find("->")) != std::string::npos;) normalized.replace(at, 2, ".");
        const auto method_at = normalized.rfind('.');
        if (method_at != std::string::npos) {
            auto receiver = normalized.substr(0, method_at);
            std::istringstream path(receiver);
            std::string part, type;
            bool first = true;
            while (std::getline(path, part, '.')) {
                if (first) {
                    if (part == "this") type = owner;
                    else { type = binding_type(source, part); if (type.empty()) type = binding_type(owner, part); }
                    first = false;
                } else if (!type.empty()) type = binding_type(type, part);
            }
            if (!type.empty()) called = type + "::" + normalized.substr(method_at + 1);
        } else if (called.find("::") == std::string::npos && !owner.empty() && find_symbol(owner + "::" + called) != UINT32_MAX) {
            called = owner + "::" + called;
        }
        const bool match = target_name == target_short ? short_symbol_name(called) == target_short : called == target_name;
        if (match && !seen[edge.src]) { seen[edge.src] = true; callers.push_back(symbols_[edge.src]); }
    }
    return callers;
}

std::vector<std::string> CodeGraph::get_param_types(const std::string& name) const {
    uint32_t id = find_symbol(name);
    if (id != UINT32_MAX) return symbols_[id].param_types;
    auto ids = find_symbols_by_suffix(name);
    if (!ids.empty()) return symbols_[ids[0]].param_types;
    return {};
}

std::string CodeGraph::get_return_type(const std::string& name) const {
    uint32_t id = find_symbol(name);
    if (id != UINT32_MAX) return symbols_[id].return_type;
    auto ids = find_symbols_by_suffix(name);
    if (!ids.empty()) return symbols_[ids[0]].return_type;
    return "";
}

std::vector<CodeSymbol> CodeGraph::query_vsa(const HyperVector& query_hv, size_t top_k) const {
    if (symbols_.empty() || top_k == 0) return {};
    std::vector<std::pair<double, uint32_t>> scored;
    scored.reserve(symbols_.size());
    for (const auto& s : symbols_) {
        if (s.body_snippet.empty()) continue;
        const auto signature = encode_code_vsa(s.lang, s.name, s.return_type, s.param_types);
        scored.emplace_back(0.8 * query_hv.similarity(signature) + 0.2 * query_hv.similarity(s.ast_vsa), s.id);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<CodeSymbol> out;
    const size_t k = std::min(top_k, scored.size());
    for (size_t i = 0; i < k; ++i) {
        out.push_back(symbols_[scored[i].second]);
    }
    return out;
}

void CodeGraph::add_pattern(AstPattern pat) {
    patterns_.push_back(std::move(pat));
}

size_t CodeGraph::memory_bytes() const noexcept {
    size_t b = symbols_.size() * sizeof(CodeSymbol) + edges_.size() * sizeof(CodeEdge);
    for (const auto& s : symbols_) {
        b += s.name.capacity() + s.return_type.capacity() + s.body_snippet.capacity() + s.file_path.capacity() + s.documentation.capacity() + s.canonical_ast.capacity();
        for (const auto& type : s.param_types) b += type.capacity() + sizeof(std::string);
        for (const auto& name : s.param_names) b += name.capacity() + sizeof(std::string);
        for (const auto& inc : s.includes_or_imports) b += inc.capacity() + sizeof(std::string);
    }
    return b;
}

void CodeGraph::load_raw(std::vector<CodeSymbol> symbols,
                         std::vector<CodeEdge> edges,
                         std::vector<AstPattern> patterns) {
    symbols_ = std::move(symbols);
    edges_ = std::move(edges);
    patterns_ = std::move(patterns);
    name_to_id_.clear();
    for (size_t i = 0; i < symbols_.size(); ++i) {
        name_to_id_[symbols_[i].name] = static_cast<uint32_t>(i);
    }
}

CodeLanguage TreeSitterCodeParser::detect_language(const std::string& file_path) {
    auto ext = fs::path(file_path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".hpp" || ext == ".h") {
        return CodeLanguage::CPP;
    }
    if (ext == ".py") return CodeLanguage::PYTHON;
    if (ext == ".ts") return CodeLanguage::TYPESCRIPT;
    if (ext == ".js" || ext == ".mjs") return CodeLanguage::JAVASCRIPT;
    return CodeLanguage::UNKNOWN;
}

bool TreeSitterCodeParser::validate_syntax(const std::string& source_code, CodeLanguage lang) {
    const TSLanguage* ts_lang = get_ts_lang(lang);
    if (!ts_lang) return false;
    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, ts_lang);
    TSTree* tree = ts_parser_parse_string(parser, nullptr, source_code.c_str(),
                                          static_cast<uint32_t>(source_code.size()));
    bool valid = false;
    if (tree && !source_code.empty()) {
        TSNode root = ts_tree_root_node(tree);
        valid = !ts_node_has_error(root) && !has_missing_or_error(root);
        ts_tree_delete(tree);
    }
    ts_parser_delete(parser);
    return valid;
}

size_t TreeSitterCodeParser::ingest_source(const std::string& source_code,
                                           const std::string& file_path,
                                           CodeLanguage lang,
                                           CodeGraph& graph,
                                           KanervaSDM* sdm, bool use_gpu) {
    const TSLanguage* ts_lang = get_ts_lang(lang);
    if (!ts_lang) return 0;

    TSParser* parser = ts_parser_new();
    ts_parser_set_language(parser, ts_lang);
    TSTree* tree = ts_parser_parse_string(parser, nullptr, source_code.c_str(),
                                          static_cast<uint32_t>(source_code.size()));
    if (!tree) {
        ts_parser_delete(parser);
        return 0;
    }

    TSNode root = ts_tree_root_node(tree);
    if (ts_node_has_error(root) || has_missing_or_error(root)) {
        ts_tree_delete(tree); ts_parser_delete(parser); return 0;
    }
    std::vector<std::string> file_includes;
    size_t added = 0;

    auto walk = [&](auto& self, TSNode node) -> void {
        const std::string ntype = ts_node_type(node);

        if (lang == CodeLanguage::CPP && ntype == "declaration") {
            TSNode parent = ts_node_parent(node);
            while (!ts_node_is_null(parent) && std::string(ts_node_type(parent)) != "function_definition") parent = ts_node_parent(parent);
            if (!ts_node_is_null(parent)) {
                const auto scope = extract_cpp_func_name(ts_node_child_by_field_name(parent, "declarator", 10), source_code);
                const auto type = ts_node_child_by_field_name(node, "type", 4);
                auto declarator = ts_node_child_by_field_name(node, "declarator", 10);
                if (!ts_node_is_null(declarator) && std::string(ts_node_type(declarator)) == "init_declarator") declarator = ts_node_child_by_field_name(declarator, "declarator", 10);
                const auto name = extract_cpp_func_name(declarator, source_code);
                if (!scope.empty() && !name.empty() && !ts_node_is_null(type)) {
                    CodeSymbol binding; binding.name = "@binding:" + scope + "::" + name;
                    binding.return_type = node_text(type, source_code); graph.add_or_update_symbol(std::move(binding));
                }
            }
        }
        if (lang == CodeLanguage::CPP && ntype == "field_declaration") {
            const auto owner = enclosing_class(node, source_code);
            const auto type = ts_node_child_by_field_name(node, "type", 4);
            const auto declarator = ts_node_child_by_field_name(node, "declarator", 10);
            const auto name = extract_cpp_func_name(declarator, source_code);
            if (!owner.empty() && !name.empty() && !ts_node_is_null(type)) {
                CodeSymbol binding;
                binding.name = "@binding:" + owner + "::" + name;
                binding.return_type = node_text(type, source_code);
                graph.add_or_update_symbol(std::move(binding));
            }
        }
        if (ntype == "class_specifier" || ntype == "struct_specifier" || ntype == "class_definition") {
            const auto name = ts_node_child_by_field_name(node, "name", 4);
            for (uint32_t i = 0; i < ts_node_named_child_count(node); ++i) {
                const auto bases = ts_node_named_child(node, i);
                const std::string kind = ts_node_type(bases);
                if (kind != "base_class_clause" && kind != "argument_list") continue;
                for (uint32_t j = 0; j < ts_node_named_child_count(bases); ++j) {
                    const auto base = ts_node_named_child(bases, j);
                    if (std::string(ts_node_type(base)) == "access_specifier") continue;
                    graph.add_edge(node_text(name, source_code), node_text(base, source_code), CodeEdgeType::IMPLEMENTS);
                }
            }
        }
        if (ntype == "for_statement" || ntype == "for_range_loop" || ntype == "with_statement" ||
            ntype == "try_statement") {
            AstPattern pattern;
            pattern.id_tag = (ntype == "try_statement" ? "TRY_CATCH" : ntype == "with_statement" ? "SAFE_RESOURCE" : "LOOP_COLLECTION");
            pattern.lang = lang;
            pattern.template_code = node_text(node, source_code);
            pattern.required_includes = file_includes;
            pattern.pattern_vsa = HyperVector::from_token("ROLE_AST").bind(HyperVector::from_token(ntype));
            graph.add_pattern(std::move(pattern));
        }
        if (lang == CodeLanguage::CPP && ntype == "preproc_include") {
            file_includes.push_back(node_text(node, source_code));
        } else if ((lang == CodeLanguage::PYTHON || (lang == CodeLanguage::JAVASCRIPT || lang == CodeLanguage::TYPESCRIPT)) &&
                   (ntype == "import_statement" || ntype == "import_from_statement")) {
            file_includes.push_back(node_text(node, source_code));
        }

        if (lang == CodeLanguage::CPP && ntype == "function_definition") {
            TSNode ret_node = ts_node_child_by_field_name(node, "type", 4);
            TSNode decl_node = ts_node_child_by_field_name(node, "declarator", 10);
            TSNode body_node = ts_node_child_by_field_name(node, "body", 4);

            CodeSymbol sym;
            sym.lang = lang;
            sym.file_path = file_path;
            sym.return_type = ts_node_is_null(ret_node) ? "void" : node_text(ret_node, source_code);
            sym.name = ts_node_is_null(decl_node) ? "" : extract_cpp_func_name(decl_node, source_code);
            if (!ts_node_is_null(decl_node)) {
                extract_cpp_params(decl_node, source_code, sym.param_types, sym.param_names);
            }
            const auto owner_class = enclosing_class(node, source_code);
            if (!owner_class.empty() && sym.name.find("::") == std::string::npos)
                sym.name = owner_class + (lang == CodeLanguage::CPP ? "::" : ".") + sym.name;
            sym.includes_or_imports = file_includes;
            sym.body_snippet = node_text(node, source_code);
            sym.canonical_ast = canonical_ast(node);
            sym.documentation = documentation(node, body_node, source_code);

            std::vector<std::string> calls;
            if (!ts_node_is_null(body_node)) collect_calls(body_node, source_code, calls);
            sym.ast_vsa = HyperVector::bundle({CodeGraph::encode_code_vsa(lang, sym.name, sym.return_type, sym.param_types, calls),
                HyperVector::from_token("AST").bind(HyperVector::from_token(sym.canonical_ast))});

            if (!sym.name.empty()) {
                const std::string sname = sym.name;
                const HyperVector hv = sym.ast_vsa;
                const auto ret = sym.return_type;
                const auto params = sym.param_types;
                const auto imports = sym.includes_or_imports;
                graph.add_or_update_symbol(std::move(sym));
                const auto& function = graph.symbols()[graph.find_symbol(sname)];
                const auto names = function.param_names;
                for (size_t i = 0; i < std::min(names.size(), params.size()); ++i) {
                    CodeSymbol binding; binding.name = "@binding:" + sname + "::" + names[i];
                    binding.return_type = params[i]; graph.add_or_update_symbol(std::move(binding));
                }
                graph.add_edge(sname, "type:" + ret, CodeEdgeType::RETURNS_TYPE);
                for (const auto& pt : params) graph.add_edge(sname, "type:" + pt, CodeEdgeType::TAKES_PARAM);
                for (const auto& inc : imports) graph.add_edge(sname, "import:" + inc, CodeEdgeType::DEPENDS_ON);
                const auto owner = sname.rfind(lang == CodeLanguage::CPP ? "::" : ".");
                if (owner != std::string::npos) graph.add_edge(sname, sname.substr(0, owner), CodeEdgeType::MEMBER_OF);
                for (const auto& c : calls) {
                    graph.add_edge(sname, c, CodeEdgeType::CALLS);
                }
                if (sdm) {
                    sdm->write(HyperVector::from_token(short_symbol_name(sname)), hv, use_gpu);
                }
                ++added;
            }
        } else if (lang == CodeLanguage::PYTHON && ntype == "function_definition") {
            TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
            TSNode params_node = ts_node_child_by_field_name(node, "parameters", 10);
            TSNode ret_node = ts_node_child_by_field_name(node, "return_type", 11);
            TSNode body_node = ts_node_child_by_field_name(node, "body", 4);

            CodeSymbol sym;
            sym.lang = lang;
            sym.file_path = file_path;
            sym.name = ts_node_is_null(name_node) ? "" : node_text(name_node, source_code);
            sym.return_type = ts_node_is_null(ret_node) ? "Any" : node_text(ret_node, source_code);
            if (!ts_node_is_null(params_node)) {
                uint32_t np = ts_node_named_child_count(params_node);
                for (uint32_t i = 0; i < np; ++i) {
                    const auto parameter = ts_node_named_child(params_node, i);
                    std::string pn = node_text(parameter, source_code);
                    const auto colon = pn.find(':');
                    std::string type = colon == std::string::npos ? "Any" : pn.substr(colon + 1);
                    pn = pn.substr(0, colon);
                    while (!pn.empty() && pn.back() == ' ') pn.pop_back();
                    while (!type.empty() && type.front() == ' ') type.erase(type.begin());
                    if (pn != "self") {
                        sym.param_names.push_back(pn);
                        sym.param_types.push_back(type);
                    }
                }
            }
            const auto owner_class = enclosing_class(node, source_code);
            if (!owner_class.empty() && sym.name.find("::") == std::string::npos)
                sym.name = owner_class + (lang == CodeLanguage::CPP ? "::" : ".") + sym.name;
            sym.includes_or_imports = file_includes;
            sym.body_snippet = node_text(node, source_code);
            sym.canonical_ast = canonical_ast(node);
            sym.documentation = documentation(node, body_node, source_code);

            std::vector<std::string> calls;
            if (!ts_node_is_null(body_node)) collect_calls(body_node, source_code, calls);
            sym.ast_vsa = HyperVector::bundle({CodeGraph::encode_code_vsa(lang, sym.name, sym.return_type, sym.param_types, calls),
                HyperVector::from_token("AST").bind(HyperVector::from_token(sym.canonical_ast))});

            if (!sym.name.empty()) {
                const std::string sname = sym.name;
                const HyperVector hv = sym.ast_vsa;
                const auto ret = sym.return_type;
                const auto params = sym.param_types;
                const auto imports = sym.includes_or_imports;
                graph.add_or_update_symbol(std::move(sym));
                const auto& function = graph.symbols()[graph.find_symbol(sname)];
                const auto names = function.param_names;
                for (size_t i = 0; i < std::min(names.size(), params.size()); ++i) {
                    CodeSymbol binding; binding.name = "@binding:" + sname + "::" + names[i];
                    binding.return_type = params[i]; graph.add_or_update_symbol(std::move(binding));
                }
                graph.add_edge(sname, "type:" + ret, CodeEdgeType::RETURNS_TYPE);
                for (const auto& pt : params) graph.add_edge(sname, "type:" + pt, CodeEdgeType::TAKES_PARAM);
                for (const auto& inc : imports) graph.add_edge(sname, "import:" + inc, CodeEdgeType::DEPENDS_ON);
                const auto owner = sname.rfind(lang == CodeLanguage::CPP ? "::" : ".");
                if (owner != std::string::npos) graph.add_edge(sname, sname.substr(0, owner), CodeEdgeType::MEMBER_OF);
                for (const auto& c : calls) {
                    graph.add_edge(sname, c, CodeEdgeType::CALLS);
                }
                if (sdm) {
                    sdm->write(HyperVector::from_token(sname), hv, use_gpu);
                }
                ++added;
            }
        } else if ((lang == CodeLanguage::JAVASCRIPT || lang == CodeLanguage::TYPESCRIPT) &&
                   (ntype == "function_declaration" || ntype == "method_definition")) {
            TSNode name_node = ts_node_child_by_field_name(node, "name", 4);
            TSNode params_node = ts_node_child_by_field_name(node, "parameters", 10);
            TSNode body_node = ts_node_child_by_field_name(node, "body", 4);

            CodeSymbol sym;
            sym.lang = lang;
            sym.file_path = file_path;
            sym.name = ts_node_is_null(name_node) ? "" : node_text(name_node, source_code);
            const auto return_node = ts_node_child_by_field_name(node, "return_type", 11);
            sym.return_type = ts_node_is_null(return_node) ? "any" : node_text(return_node, source_code);
            while (!sym.return_type.empty() && (sym.return_type.front() == ':' || sym.return_type.front() == ' ')) sym.return_type.erase(sym.return_type.begin());
            if (!ts_node_is_null(params_node)) {
                uint32_t np = ts_node_named_child_count(params_node);
                for (uint32_t i = 0; i < np; ++i) {
                    const auto param = ts_node_named_child(params_node, i);
                    const auto name = ts_node_child_by_field_name(param, "pattern", 7);
                    const auto type = ts_node_child_by_field_name(param, "type", 4);
                    sym.param_names.push_back(node_text(ts_node_is_null(name) ? param : name, source_code));
                    auto type_name = ts_node_is_null(type) ? "any" : node_text(type, source_code);
                    while (!type_name.empty() && (type_name.front() == ':' || type_name.front() == ' ')) type_name.erase(type_name.begin());
                    sym.param_types.push_back(type_name);
                }
            }
            const auto owner_class = enclosing_class(node, source_code);
            if (!owner_class.empty() && sym.name.find("::") == std::string::npos)
                sym.name = owner_class + (lang == CodeLanguage::CPP ? "::" : ".") + sym.name;
            sym.includes_or_imports = file_includes;
            sym.body_snippet = node_text(node, source_code);
            sym.canonical_ast = canonical_ast(node);
            sym.documentation = documentation(node, body_node, source_code);

            std::vector<std::string> calls;
            if (!ts_node_is_null(body_node)) collect_calls(body_node, source_code, calls);
            sym.ast_vsa = HyperVector::bundle({CodeGraph::encode_code_vsa(lang, sym.name, sym.return_type, sym.param_types, calls),
                HyperVector::from_token("AST").bind(HyperVector::from_token(sym.canonical_ast))});

            if (!sym.name.empty()) {
                const std::string sname = sym.name;
                const HyperVector hv = sym.ast_vsa;
                const auto ret = sym.return_type;
                const auto params = sym.param_types;
                const auto imports = sym.includes_or_imports;
                graph.add_or_update_symbol(std::move(sym));
                const auto& function = graph.symbols()[graph.find_symbol(sname)];
                const auto names = function.param_names;
                for (size_t i = 0; i < std::min(names.size(), params.size()); ++i) {
                    CodeSymbol binding; binding.name = "@binding:" + sname + "::" + names[i];
                    binding.return_type = params[i]; graph.add_or_update_symbol(std::move(binding));
                }
                graph.add_edge(sname, "type:" + ret, CodeEdgeType::RETURNS_TYPE);
                for (const auto& pt : params) graph.add_edge(sname, "type:" + pt, CodeEdgeType::TAKES_PARAM);
                for (const auto& inc : imports) graph.add_edge(sname, "import:" + inc, CodeEdgeType::DEPENDS_ON);
                const auto owner = sname.rfind(lang == CodeLanguage::CPP ? "::" : ".");
                if (owner != std::string::npos) graph.add_edge(sname, sname.substr(0, owner), CodeEdgeType::MEMBER_OF);
                for (const auto& c : calls) {
                    graph.add_edge(sname, c, CodeEdgeType::CALLS);
                }
                if (sdm) {
                    sdm->write(HyperVector::from_token(sname), hv, use_gpu);
                }
                ++added;
            }
        }

        uint32_t nc = ts_node_child_count(node);
        for (uint32_t i = 0; i < nc; ++i) {
            self(self, ts_node_child(node, i));
        }
    };

    walk(walk, root);
    ts_tree_delete(tree);
    ts_parser_delete(parser);
    return added;
}

size_t TreeSitterCodeParser::ingest_directory(const std::string& repo_dir,
                                              CodeGraph& graph,
                                              KanervaSDM* sdm, bool use_gpu) {
    size_t total = 0;
    std::error_code ec;
    fs::recursive_directory_iterator it(repo_dir, fs::directory_options::skip_permission_denied, ec), end;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto& entry = *it;
        if (entry.is_symlink()) { if (entry.is_directory()) it.disable_recursion_pending(); continue; }
        if (entry.is_directory()) {
            const auto name = entry.path().filename().string();
            if (name.starts_with("build") || name == ".git" || name == ".cognicore" || name == "node_modules" || name == ".venv" || name == "__pycache__")
                it.disable_recursion_pending();
            continue;
        }
        if (!entry.is_regular_file() || entry.file_size() > 4 * 1024 * 1024) continue;
        const std::string p = entry.path().string();
        CodeLanguage lang = detect_language(p);
        if (lang == CodeLanguage::UNKNOWN) continue;

        std::ifstream ifs(p);
        if (!ifs) continue;
        std::ostringstream ss;
        ss << ifs.rdbuf();
        total += ingest_source(ss.str(), p, lang, graph, sdm, use_gpu);
    }
    return total;
}

} // namespace cognicore
