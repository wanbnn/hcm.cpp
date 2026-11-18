#pragma once

#include "core/sdm.hpp"
#include "core/vsa.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cognicore {

enum class CodeLanguage : uint8_t {
    CPP        = 0,
    PYTHON     = 1,
    JAVASCRIPT = 2,
    TYPESCRIPT = 3,
    UNKNOWN    = 4
};

const char* code_lang_to_string(CodeLanguage lang) noexcept;
CodeLanguage string_to_code_lang(const std::string& s) noexcept;

enum class CodeEdgeType : uint8_t {
    CALLS        = 0,
    RETURNS_TYPE = 1,
    TAKES_PARAM  = 2,
    MEMBER_OF    = 3,
    IMPLEMENTS   = 4,
    DEPENDS_ON   = 5
};

struct CodeSymbol {
    uint32_t id = 0;
    std::string name;         // e.g. "KanervaSDM::read" or "read"
    std::string return_type;  // e.g. "HyperVector"
    std::vector<std::string> param_types;
    std::vector<std::string> param_names;
    std::vector<std::string> includes_or_imports;
    std::string documentation;
    std::string canonical_ast;
    std::string body_snippet;
    std::string file_path;
    CodeLanguage lang = CodeLanguage::CPP;
    HyperVector ast_vsa;      // Compositional VSA encoding of signature + AST roles
};

struct CodeEdge {
    uint32_t src = 0;
    uint32_t dst = 0;
    CodeEdgeType type = CodeEdgeType::CALLS;
};

struct AstPattern {
    std::string id_tag;       // e.g. "READ_FILE", "LOOP_COLLECTION", "TRY_CATCH", "CALL_API"
    CodeLanguage lang = CodeLanguage::CPP;
    std::string template_code;
    std::vector<std::string> required_includes;
    HyperVector pattern_vsa;
};

class CodeGraph {
public:
    uint32_t add_or_update_symbol(CodeSymbol sym);
    uint32_t find_symbol(const std::string& name) const;
    std::vector<uint32_t> find_symbols_by_suffix(const std::string& short_name) const;
    void add_edge(const std::string& src_name, const std::string& dst_name, CodeEdgeType type);

    /// Finds all symbols that call `target_name` (supports short name like "read" or qualified "KanervaSDM::read").
    std::vector<CodeSymbol> find_callers(const std::string& target_name) const;

    /// Finds return type & parameter types of a symbol.
    std::vector<std::string> get_param_types(const std::string& name) const;
    std::string get_return_type(const std::string& name) const;

    /// Encodes a symbol/query into compositional VSA: Bind(LANG, l) ^ Bind(RET, r) ^ Bind(NAME, n) ...
    static HyperVector encode_code_vsa(CodeLanguage lang,
                                       const std::string& name,
                                       const std::string& ret_type,
                                       const std::vector<std::string>& param_types,
                                       const std::vector<std::string>& calls = {});

    /// Queries CodeGraph + SDM associatively for symbols matching a natural-language or code query.
    std::vector<CodeSymbol> query_vsa(const HyperVector& query_hv, size_t top_k = 5) const;

    void add_pattern(AstPattern pat);
    const std::vector<AstPattern>& patterns() const { return patterns_; }
    const std::vector<CodeSymbol>& symbols() const { return symbols_; }
    const std::vector<CodeEdge>& edges() const { return edges_; }

    size_t num_symbols() const noexcept { return symbols_.size(); }
    size_t num_edges() const noexcept { return edges_.size(); }
    size_t memory_bytes() const noexcept;

    void load_raw(std::vector<CodeSymbol> symbols,
                  std::vector<CodeEdge> edges,
                  std::vector<AstPattern> patterns);

private:
    std::unordered_map<std::string, uint32_t> name_to_id_;
    std::vector<CodeSymbol> symbols_;
    std::vector<CodeEdge> edges_;
    std::vector<AstPattern> patterns_;
};

class TreeSitterCodeParser {
public:
    /// Detects language from file extension (.cpp/.hpp/.cc/.h, .py, .js/.ts).
    static CodeLanguage detect_language(const std::string& file_path);

    /// Validates syntax using tree-sitter: returns true iff root tree has 0 ERROR/MISSING nodes.
    static bool validate_syntax(const std::string& source_code, CodeLanguage lang);

    /// Parses source code into CodeGraph + stores compositional VSA vectors in SDM.
    static size_t ingest_source(const std::string& source_code,
                                const std::string& file_path,
                                CodeLanguage lang,
                                CodeGraph& graph,
                                KanervaSDM* sdm = nullptr, bool use_gpu = false);

    /// Recursively ingests a repository directory (.cpp, .hpp, .h, .py, .js).
    static size_t ingest_directory(const std::string& repo_dir,
                                   CodeGraph& graph,
                                   KanervaSDM* sdm = nullptr, bool use_gpu = false);
};

} // namespace cognicore
