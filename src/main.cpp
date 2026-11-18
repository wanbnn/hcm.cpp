#include "core/model_bundle.hpp"
#include "core/assistant.hpp"
#include "core/document_planner.hpp"
#include "core/lexicalized_realizer.hpp"
#include "core/morphological_realizer.hpp"
#include "core/symbolic_math.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cognicore;

namespace {

void print_usage() {
    std::cout << "Uso:\n"
              << "  cognicore ingest-books --input <dir_ou_txt> --model <dir_modelo> [--parser rules|conllu] [--style-only] [--use-gpu]\n"
              << "  cognicore ingest-corpus --input <dir_ou_arquivo> --model <dir_modelo> [--format auto|text|jsonl] [--checkpoint-every N] [--limit-files N] [--limit-records N] [--style-only] [--use-gpu]\n"
              << "  cognicore train-dialogue --input <conversas.jsonl> --model <dir_modelo> [--epochs 20] [--use-gpu]\n"
              << "  cognicore evaluate-dialogue --input <avaliacao.jsonl> --model <dir_modelo> [--out <metricas.json>]\n"
              << "  cognicore consolidate --model <dir_modelo>\n"
              << "  cognicore stats --model <dir_modelo>\n"
              << "  cognicore ingest-code --repo <dir> --model <dir_modelo> [--use-gpu]\n"
              << "  cognicore code --model <dir_modelo> --workspace <dir> --lang cpp|py|js --task <descrição>\n"
              << "  cognicore train-realizer --model <dir_modelo> [--use-gpu]\n"
              << "  cognicore evaluate-fluency --model <dir_modelo> --input <texto> [--out <relatorio.json>]\n"
              << "  cognicore write --model <dir_modelo> --topic <assunto> [--sections N] [--paragraphs N] [--realizer strict|lexical|fluent] [--out <arquivo.md>]\n"
              << "  cognicore math --expr \"2x + 3 = 11\" [--model <dir_modelo>]\n"
              << "  cognicore chat --model <dir_modelo> [--workspace <dir>] [--realizer strict|lexical|fluent] [--lang cpp|py|js] [--debug]\n";
}

std::string get_arg(int argc, char** argv, const std::string& flag, const std::string& def = "") {
    for (int i = 2; i + 1 < argc; ++i) {
        if (argv[i] == flag) return argv[i + 1];
    }
    return def;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 2; i < argc; ++i) {
        if (argv[i] == flag) return true;
    }
    return false;
}

std::string read_file_text(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// Simple JSONL field extractor for {"text": "...", "intent": "..."}
std::string extract_json_str(const std::string& line, const std::string& key) {
    const std::string pat = "\"" + key + "\"";
    size_t kpos = line.find(pat);
    if (kpos == std::string::npos) return "";
    size_t colon = line.find(':', kpos + pat.size());
    if (colon == std::string::npos) return "";
    size_t q1 = line.find('"', colon + 1);
    if (q1 == std::string::npos) throw std::invalid_argument("JSONL: expected a string");
    std::string out;
    auto utf8 = [&](uint32_t cp) {
        if (cp <= 0x7f) out += char(cp);
        else if (cp <= 0x7ff) { out += char(0xc0 | (cp >> 6)); out += char(0x80 | (cp & 63)); }
        else if (cp <= 0xffff) { out += char(0xe0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63)); }
        else { out += char(0xf0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 63)); out += char(0x80 | ((cp >> 6) & 63)); out += char(0x80 | (cp & 63)); }
    };
    auto hex4 = [&](size_t& at) {
        uint32_t value = 0;
        for (int n = 0; n < 4; ++n) {
            if (++at >= line.size()) throw std::invalid_argument("JSONL: truncated unicode escape");
            char c = line[at];
            int digit = c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : c >= 'A' && c <= 'F' ? c-'A'+10 : -1;
            if (digit < 0) throw std::invalid_argument("JSONL: invalid unicode escape");
            value = (value << 4) | uint32_t(digit);
        }
        return value;
    };
    for (size_t at = q1 + 1; at < line.size(); ++at) {
        char c = line[at];
        if (c == '"') return out;
        if (c != '\\') { out += c; continue; }
        if (++at >= line.size()) break;
        switch (line[at]) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                auto cp = hex4(at);
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    if (at + 2 >= line.size() || line[at+1] != '\\' || line[at+2] != 'u') throw std::invalid_argument("JSONL: missing low surrogate");
                    at += 2; auto low = hex4(at);
                    if (low < 0xdc00 || low > 0xdfff) throw std::invalid_argument("JSONL: invalid low surrogate");
                    cp = 0x10000 + ((cp-0xd800) << 10) + (low-0xdc00);
                } else if (cp >= 0xdc00 && cp <= 0xdfff) throw std::invalid_argument("JSONL: unexpected low surrogate");
                utf8(cp); break;
            }
            default: throw std::invalid_argument("JSONL: invalid escape");
        }
    }
    throw std::invalid_argument("JSONL: unterminated string");
}

void load_rules_if_present(TextParser& parser) {
    if (fs::exists("data/linguistic_rules.json")) {
        parser.load_rules_json("data/linguistic_rules.json");
    } else if (fs::exists("../data/linguistic_rules.json")) {
        parser.load_rules_json("../data/linguistic_rules.json");
    }
}

int cmd_ingest_books(int argc, char** argv) {
    const std::string input = get_arg(argc, argv, "--input");
    const std::string model_dir = get_arg(argc, argv, "--model");
    const bool use_gpu = has_flag(argc, argv, "--use-gpu");
    const std::string parser_mode = get_arg(argc, argv, "--parser", "rules");
    const bool style_only = has_flag(argc, argv, "--style-only");
    const size_t max_vocab = std::stoul(get_arg(argc, argv, "--max-vocab", "0"));
    const size_t context_stride = std::stoul(get_arg(argc, argv, "--context-stride", "1"));
    if (!context_stride) throw std::invalid_argument("--context-stride deve ser positivo");
    if (parser_mode != "rules" && parser_mode != "conllu") {
        std::cerr << "Parser indisponível: use rules ou conllu (pré-processado).\n";
        return 1;
    }
    if (input.empty() || model_dir.empty()) {
        print_usage();
        return 1;
    }

    ModelBundle bundle;
    if (fs::exists(model_dir + "/codebook.bin")) {
        if (!bundle.load(model_dir, false)) { std::cerr << "Erro ao carregar modelo existente.\n"; return 1; }
    }
    load_rules_if_present(bundle.parser);

    std::vector<std::string> files;
    if (fs::is_directory(input)) {
        for (const auto& entry : fs::recursive_directory_iterator(input)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".txt" || entry.path().extension() == ".conllu")) {
                files.push_back(entry.path().string());
            }
        }
    } else {
        files.push_back(input);
    }

    std::sort(files.begin(), files.end());
    size_t total_triples = 0;
    for (const auto& fpath : files) {
        const std::string text = read_file_text(fpath);
        if (text.empty()) continue;
        bundle.lexicon.observe(text, bundle.parser, "technical");
        bundle.coherence.observe(text, bundle.parser);

        auto doc = (parser_mode == "conllu" || fs::path(fpath).extension() == ".conllu")
            ? bundle.parser.parse_conllu(text) : bundle.parser.parse_text(text);
        for (const auto& s_toks : doc.sentences_tokens) {
            bundle.realizer.observe_sentence_tokens(s_toks);
            for (size_t i = 0; i < s_toks.size(); i += context_stride) {
                if (max_vocab && bundle.codebook.size() >= max_vocab && !bundle.codebook.contains(s_toks[i])) continue;
                std::vector<std::string> ctx;
                for (size_t j = (i >= 2 ? i - 2 : 0); j <= std::min(s_toks.size() - 1, i + 2); ++j) {
                    if (i != j && (!max_vocab || bundle.codebook.size() < max_vocab || bundle.codebook.contains(s_toks[j]))) ctx.push_back(s_toks[j]);
                }
                bundle.codebook.observe_context(s_toks[i], ctx);
            }
        }
        if (fs::path(fpath).extension() != ".conllu") bundle.fluent_realizer.observe(text, bundle.parser, &doc);
        else for (const auto& tokens : doc.sentences_tokens) {
            std::string sentence; for (const auto& t : tokens) { if (!sentence.empty()) sentence += " "; sentence += t; }
            bundle.fluent_realizer.observe(sentence, bundle.parser, &doc);
        }
        if (!style_only) for (const auto& tr : doc.triples) {
            bundle.graph.add_fact(tr.subject, tr.relation, tr.object, tr.type, tr.confidence, false,
                                  tr.negated, tr.tense, tr.modality, tr.subj_gender, tr.subj_number,
                                  tr.obj_gender, tr.obj_number);
            HyperVector addr = HyperVector::from_token(tr.subject);
            HyperVector data = HyperVector::from_token(tr.object);
            bundle.sdm.write(addr, data, use_gpu);
            ++total_triples;
        }
        if (!style_only) for (auto& pair : realizer_document_pairs(doc)) bundle.realizer_pairs.push_back(std::move(pair));
        for (const auto& cf : doc.frames) {
            bundle.realizer.add_frame(cf);
        }
    }

    bundle.codebook.finalize_semantics();
    if (!bundle.save(model_dir)) {
        std::cerr << "Erro ao salvar modelo em " << model_dir << "\n";
        return 1;
    }

    std::cout << "[ingest-books] Arquivos: " << files.size()
              << " | Vocabulario VSA: " << bundle.codebook.size()
              << " | Triplas extraidas: " << total_triples
              << " | Nos no grafo: " << bundle.graph.num_nodes()
              << " | Frames: " << bundle.realizer.num_frames()
              << " | Uso: " << (style_only ? "somente estilo/associação" : "fatos e estilo") << "\n";
    return 0;
}

int cmd_ingest_corpus(int argc, char** argv) {
    const auto input = get_arg(argc, argv, "--input");
    const auto model_dir = get_arg(argc, argv, "--model");
    if (input.empty() || model_dir.empty()) { print_usage(); return 1; }
    const bool use_gpu = has_flag(argc, argv, "--use-gpu");
    const bool style_only = has_flag(argc, argv, "--style-only");
    const auto format = get_arg(argc, argv, "--format", "auto");
    const size_t checkpoint = std::stoul(get_arg(argc, argv, "--checkpoint-every", "10"));
    const size_t limit_files = std::stoul(get_arg(argc, argv, "--limit-files", "0"));
    const size_t limit_records = std::stoul(get_arg(argc, argv, "--limit-records", "0"));
    const size_t max_vocab = std::stoul(get_arg(argc, argv, "--max-vocab", "0"));
    const size_t context_stride = std::stoul(get_arg(argc, argv, "--context-stride", "1"));
    if (format != "auto" && format != "text" && format != "jsonl") {
        std::cerr << "Formato inválido: use auto, text ou jsonl.\n";
        return 1;
    }

    ModelBundle bundle;
    if (fs::exists(model_dir + "/codebook.bin") && !bundle.load(model_dir, false)) {
        std::cerr << "Erro ao carregar modelo existente.\n";
        return 1;
    }
    load_rules_if_present(bundle.parser);
    fs::create_directories(model_dir);
    const auto manifest_path = fs::path(model_dir) / "ingestion_manifest.jsonl";
    std::unordered_map<std::string, std::string> processed;
    if (fs::exists(manifest_path)) {
        std::ifstream manifest(manifest_path);
        std::string line;
        while (std::getline(manifest, line)) {
            const auto path = extract_json_str(line, "path");
            const auto size = extract_json_str(line, "size");
            if (!path.empty()) processed[path] = size;
        }
    }
    std::ofstream manifest(manifest_path, std::ios::app);

    std::vector<std::string> files;
    if (fs::is_directory(input)) {
        for (const auto& entry : fs::recursive_directory_iterator(input)) {
            if (!entry.is_regular_file()) continue;
            const auto ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".jsonl" || ext == ".md" || ext == ".conllu")
                files.push_back(entry.path().string());
        }
    } else files.push_back(input);
    std::sort(files.begin(), files.end());
    if (limit_files && files.size() > limit_files) files.resize(limit_files);

    size_t records = 0, new_files = 0;
    auto observe = [&](const std::string& text) {
        if (text.empty()) return;
        auto document = bundle.parser.parse_text(text);
        for (const auto& toks : document.sentences_tokens) {
            bundle.realizer.observe_sentence_tokens(toks);
            for (size_t i = 0; i < toks.size(); i += std::max<size_t>(1, context_stride)) {
                if (max_vocab && bundle.codebook.size() >= max_vocab && !bundle.codebook.contains(toks[i])) continue;
                std::vector<std::string> context;
                for (size_t j = (i >= 2 ? i - 2 : 0); j <= std::min(toks.size() - 1, i + 2); ++j)
                    if (i != j && (!max_vocab || bundle.codebook.size() < max_vocab || bundle.codebook.contains(toks[j])))
                        context.push_back(toks[j]);
                bundle.codebook.observe_context(toks[i], context);
            }
        }
        bundle.lexicon.observe(text, bundle.parser, "technical");
        bundle.coherence.observe(text, bundle.parser);
        bundle.fluent_realizer.observe(text, bundle.parser, &document);
        if (!style_only) {
            for (const auto& triple : document.triples) {
                bundle.graph.add_fact(triple.subject, triple.relation, triple.object, triple.type,
                                      triple.confidence, false, triple.negated, triple.tense,
                                      triple.modality, triple.subj_gender, triple.subj_number,
                                      triple.obj_gender, triple.obj_number);
                bundle.sdm.write(HyperVector::from_token(triple.subject),
                                 HyperVector::from_token(triple.object), use_gpu);
            }
            for (auto& pair : realizer_document_pairs(document)) bundle.realizer_pairs.push_back(std::move(pair));
            for (const auto& frame : document.frames) bundle.realizer.add_frame(frame);
        }
    };

    for (const auto& file : files) {
        std::error_code ec;
        const auto size = fs::file_size(file, ec);
        if (ec) continue;
        const auto size_text = std::to_string(size);
        if (processed[file] == size_text) continue;
        std::ifstream in(file);
        if (!in) continue;
        const auto ext = fs::path(file).extension().string();
        const bool jsonl = format == "jsonl" || (format == "auto" && ext == ".jsonl");
        if (jsonl) {
            std::string line;
            while (std::getline(in, line)) {
                auto text = extract_json_str(line, "text");
                if (text.empty()) text = extract_json_str(line, "content");
                if (text.empty()) continue;
                observe(text);
                ++records;
                if (limit_records && records >= limit_records) break;
            }
        } else {
            std::ostringstream buffer;
            buffer << in.rdbuf();
            observe(buffer.str());
        }
        if (limit_records && records >= limit_records) break;
        manifest << "{\"path\":\"" << json_escape(file) << "\",\"size\":\"" << size_text << "\"}\n" << std::flush;
        processed[file] = size_text;
        ++new_files;
        if (checkpoint && new_files % checkpoint == 0) {
            bundle.codebook.finalize_semantics();
            if (!bundle.save(model_dir)) { std::cerr << "Erro ao salvar checkpoint.\n"; return 1; }
        }
    }
    bundle.codebook.finalize_semantics();
    if (!bundle.save(model_dir)) { std::cerr << "Erro ao salvar modelo.\n"; return 1; }
    std::cout << "[ingest-corpus] arquivos novos=" << new_files << " registros=" << records
              << " vocabulario=" << bundle.codebook.size() << " grafo=" << bundle.graph.num_nodes()
              << "/" << bundle.graph.num_edges() << "\n";
    return 0;
}

int cmd_train_dialogue(int argc, char** argv) {
    const std::string input = get_arg(argc, argv, "--input");
    const std::string model_dir = get_arg(argc, argv, "--model");
    const size_t epochs = std::stoul(get_arg(argc, argv, "--epochs", "20"));
    const bool use_gpu = has_flag(argc, argv, "--use-gpu");
    if (input.empty() || model_dir.empty()) {
        print_usage();
        return 1;
    }

    ModelBundle bundle;
    if (fs::exists(model_dir + "/codebook.bin")) {
        if (!bundle.load(model_dir, false)) { std::cerr << "Erro ao carregar modelo existente.\n"; return 1; }
    }
    auto config = bundle.tsetlin.config();
    const auto clauses = get_arg(argc, argv, "--clauses");
    if (!clauses.empty() || config.num_classes < static_cast<size_t>(DialogIntent::COUNT)) {
        config.num_classes = static_cast<size_t>(DialogIntent::COUNT);
        if (!clauses.empty()) config.clauses_per_class = std::stoul(clauses);
        if (!config.clauses_per_class || config.clauses_per_class > 512 || config.clauses_per_class % 2)
            throw std::invalid_argument("--clauses deve ser par entre 2 e 512");
        bundle.tsetlin = TsetlinMachine(config);
        bundle.has_sparse_tsetlin = false;
    }
    load_rules_if_present(bundle.parser);

    std::ifstream ifs(input);
    if (!ifs) {
        std::cerr << "Nao foi possivel abrir " << input << "\n";
        return 1;
    }

    std::vector<std::vector<uint64_t>> X;
    std::vector<size_t> y;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::string text = extract_json_str(line, "text");
        std::string intent_str = extract_json_str(line, "intent");
        if (text.empty()) continue;
        if (intent_str != intent_to_string(string_to_intent(intent_str)))
            throw std::invalid_argument("Intenção desconhecida: " + intent_str);

        auto toks = bundle.parser.tokenize(text);
        bundle.realizer.observe_sentence_tokens(toks);
        std::string tone = "technical";
        if (intent_str == "EXPLAIN_CAUSE" || intent_str == "HYPOTHESIZE") tone = "didactic";
        else if (intent_str == "GREET" || intent_str == "THANK" || intent_str == "SMALLTALK") tone = "casual";
        bundle.lexicon.observe(text, bundle.parser, tone);
        bundle.coherence.observe(text, bundle.parser);
        bundle.fluent_realizer.observe(text, bundle.parser);
        for (auto& pair : realizer_document_pairs(bundle.parser.parse_text(text)))
            bundle.realizer_pairs.push_back(std::move(pair));
        X.push_back(bundle.encode_dialogue_features(text));
        y.push_back(static_cast<size_t>(string_to_intent(intent_str)));
    }

    if (X.empty()) {
        std::cerr << "Nenhum exemplo valido encontrado em " << input << "\n";
        return 1;
    }

    const size_t unique_examples = X.size();
    if (has_flag(argc, argv, "--balance")) {
        std::vector<std::vector<size_t>> by_class(static_cast<size_t>(DialogIntent::COUNT));
        for (size_t i = 0; i < y.size(); ++i) by_class[y[i]].push_back(i);
        size_t largest = 0;
        for (const auto& values : by_class) largest = std::max(largest, values.size());
        const auto original_x = X; const auto original_y = y;
        for (const auto& values : by_class) if (!values.empty())
            for (size_t i = values.size(); i < largest; ++i) {
                const size_t index = values[i % values.size()];
                X.push_back(original_x[index]); y.push_back(original_y[index]);
            }
    }
    bundle.tsetlin.fit(X, y, epochs, use_gpu);
    bundle.has_sparse_tsetlin = false;
    TsetlinConfig context_config;
    context_config.num_features = 16;
    context_config.num_classes = static_cast<size_t>(DialogIntent::COUNT);
    context_config.clauses_per_class = 64;
    context_config.n_states = 30;
    context_config.margin_T = 20;
    TsetlinMachine context(context_config);
    std::vector<std::vector<uint64_t>> context_x;
    context_x.reserve(X.size());
    for (const auto& row : X) context_x.push_back({row[VSA_WORDS]});
    context.fit(context_x, y, std::max<size_t>(epochs, 12), false);
    bundle.context_policy = context.export_sparse();
    bundle.has_context_policy = true;

    if (!bundle.save(model_dir)) {
        std::cerr << "Erro ao salvar modelo em " << model_dir << "\n";
        return 1;
    }

    std::cout << "[train-dialogue] Exemplos únicos: " << unique_examples << " | Após balanceamento: " << X.size()
              << " | Epocas: " << epochs
              << " | GPU: " << (use_gpu && backend::is_gpu_available() ? "HIP/ROCm" : "CPU") << "\n";
    return 0;
}

int cmd_evaluate_dialogue(int argc, char** argv) {
    ModelBundle bundle;
    const auto model = get_arg(argc, argv, "--model");
    std::ifstream input(get_arg(argc, argv, "--input"));
    if (model.empty() || !input || !bundle.load(model)) return 1;
    const size_t classes = static_cast<size_t>(DialogIntent::COUNT);
    std::vector<std::vector<size_t>> matrix(classes, std::vector<size_t>(classes));
    size_t total = 0, correct = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto text = extract_json_str(line, "text"), label = extract_json_str(line, "intent");
        if (text.empty() || label != intent_to_string(string_to_intent(label))) return 1;
        const size_t actual = static_cast<size_t>(string_to_intent(label));
        const size_t predicted = static_cast<size_t>(bundle.classify_dialogue(text));
        if (predicted >= classes) return 1;
        ++matrix[actual][predicted]; ++total; correct += actual == predicted;
    }
    if (!total) return 1;
    std::ostringstream report;
    report << "{\"samples\":" << total << ",\"correct\":" << correct
           << ",\"accuracy\":" << double(correct) / double(total) << ",\"confusion_matrix\":{";
    for (size_t i = 0; i < classes; ++i) {
        if (i) report << ',';
        report << '\"' << intent_to_string(static_cast<DialogIntent>(i)) << "\":[";
        for (size_t j = 0; j < classes; ++j) report << (j ? "," : "") << matrix[i][j];
        report << ']';
    }
    report << "}}\n";
    const auto output = get_arg(argc, argv, "--out");
    if (!output.empty()) { std::ofstream file(output); if (!file) return 1; file << report.str(); }
    std::cout << report.str();
    return 0;
}

int cmd_consolidate(int argc, char** argv) {
    const std::string model_dir = get_arg(argc, argv, "--model");
    if (model_dir.empty()) {
        print_usage();
        return 1;
    }

    ModelBundle bundle;
    if (!bundle.load(model_dir, false)) {
        std::cerr << "Erro ao carregar modelo de " << model_dir << "\n";
        return 1;
    }

    const size_t pruned = bundle.graph.consolidate();
    bundle.realizer.consolidate_frames();
    bundle.sdm.binarize_for_inference();
    bundle.sparse_tsetlin = bundle.tsetlin.export_sparse();
    bundle.has_sparse_tsetlin = true;

    if (!bundle.save(model_dir)) {
        std::cerr << "Erro ao salvar modelo consolidado.\n";
        return 1;
    }

    std::cout << "[consolidate] Arestas podadas/fundidas: " << pruned
              << " | SDM binarizada: sim"
              << " | Tsetlin esparso (literais ativos): "
              << bundle.sparse_tsetlin.active_literal_count() << "\n";
    return 0;
}

int cmd_stats(int argc, char** argv) {
    const std::string model_dir = get_arg(argc, argv, "--model");
    if (model_dir.empty()) {
        print_usage();
        return 1;
    }

    ModelBundle bundle;
    if (!bundle.load(model_dir, true)) {
        std::cerr << "Erro ao carregar modelo de " << model_dir << "\n";
        return 1;
    }

    const size_t disk_bytes = ModelBundle::disk_size_bytes(model_dir);
    const size_t ram_bytes = bundle.estimated_ram_bytes();
    const size_t active_clauses = bundle.has_sparse_tsetlin
        ? bundle.sparse_tsetlin.non_empty_clause_count()
        : bundle.tsetlin.total_clauses();

    std::cout << "=== CogniCore Stats (" << model_dir << ") ===\n"
              << "  VSA bits               : " << VSA_BITS << "\n"
              << "  Vocabulario Codebook   : " << bundle.codebook.size() << " tokens\n"
              << "  SDM ocupacao           : " << bundle.sdm.occupied_locations()
              << " / " << bundle.sdm.num_locations()
              << " (writes=" << bundle.sdm.write_count()
              << ", binarized=" << (bundle.sdm.is_binarized() ? "sim" : "nao") << ")\n"
              << "  Tsetlin clausulas      : " << active_clauses
              << " (modo=" << (bundle.has_sparse_tsetlin ? "esparso" : "denso") << ")\n"
              << "  Grafo Causal           : " << bundle.graph.num_nodes() << " nos, "
              << bundle.graph.num_edges() << " arestas; mmap="
              << ((bundle.mapped_graph && bundle.mapped_graph->valid()) ? "sim" : "nao") << "\n"
              << "  Realizador linguistico : " << bundle.realizer.num_frames() << " frames, "
              << bundle.realizer.num_ngrams() << " n-gramas\n"
              << "  Fluencia DOP/PPM       : " << bundle.fluent_realizer.num_fragments() << " producoes, "
              << bundle.fluent_realizer.ppm_contexts() << " contextos; treinado=" << bundle.fluent_realizer.ready()
              << "; mmap=" << bundle.fluent_realizer.mapped() << "\n"
              << "  Contexto discursivo    : " << bundle.context_memory.entities().size()
              << " entidades, turno " << bundle.context_memory.turn() << "\n"
              << "  Lexico do corpus       : " << bundle.lexicon.entries() << " entradas, "
              << bundle.lexicon.collocations() << " colocacoes\n"
              << "  Coerencia (KN)         : " << bundle.coherence.unigrams() << " unigramas, "
              << bundle.coherence.bigrams() << " bigramas, " << bundle.coherence.trigrams() << " trigramas\n"
              << "  Tamanho em disco       : " << (disk_bytes / 1024.0) << " KB\n"
              << "  RAM estimada           : " << (ram_bytes / 1024.0) << " KB\n";
    return 0;
}

int cmd_evaluate_fluency(int argc, char** argv) {
    const auto model_path = get_arg(argc, argv, "--model");
    const auto input = get_arg(argc, argv, "--input");
    if (model_path.empty() || input.empty()) { print_usage(); return 1; }
    ModelBundle bundle;
    if (!bundle.load(model_path, true)) { std::cerr << "Erro ao carregar modelo.\n"; return 1; }
    std::ifstream input_stream(input);
    if (!input_stream) { std::cerr << "Arquivo de avaliação ausente.\n"; return 1; }
    std::ostringstream buffer;
    buffer << input_stream.rdbuf();
    const auto text = buffer.str();
    if (text.empty()) { std::cerr << "Arquivo de avaliação vazio.\n"; return 1; }
    const auto evaluation = bundle.fluent_realizer.evaluate_text(text);
    std::ostringstream report;
    report << "{"
           << "\"model\":\"" << json_escape(model_path) << "\","
           << "\"input\":\"" << json_escape(input) << "\","
           << "\"estimator\":\"PPM-C ordem 5 sem exclusão (calibração de relatório)\","
           << "\"bits_per_byte\":" << std::setprecision(8) << evaluation.bits_per_byte << ","
           << "\"perplexity\":" << evaluation.perplexity << ","
           << "\"bytes\":" << evaluation.bytes << ","
           << "\"ppm_contexts\":" << bundle.fluent_realizer.ppm_contexts() << ","
           << "\"fragments\":" << bundle.fluent_realizer.num_fragments() << ","
           << "\"trained\":" << (bundle.fluent_realizer.ready() ? "true" : "false") << ","
           << "\"mapped\":" << (bundle.fluent_realizer.mapped() ? "true" : "false")
           << "}\n";
    const auto output = get_arg(argc, argv, "--out");
    if (!output.empty()) { std::ofstream file(output); if (!file) return 1; file << report.str(); }
    std::cout << report.str();
    return 0;
}

int cmd_write(int argc, char** argv) {
    const auto model_path = get_arg(argc, argv, "--model");
    const auto topic = get_arg(argc, argv, "--topic");
    if (model_path.empty() || topic.empty()) { print_usage(); return 1; }
    ModelBundle bundle;
    if (!bundle.load(model_path, true)) { std::cerr << "Erro ao carregar modelo.\n"; return 1; }
    MorphologicalRealizer strict;
    LexicalizedGrammarRealizer lexical(bundle.language_rules);
    const auto mode = get_arg(argc, argv, "--realizer", "strict");
    const ISurfaceRealizer* mouth = &strict;
    if (mode == "lexical") mouth = &lexical;
    else if (mode == "fluent" && bundle.fluent_realizer.ready()) mouth = &bundle.fluent_realizer;
    else if (mode != "strict" && mode != "fluent" && mode != "lexical") { std::cerr << "Realizador inválido.\n"; return 1; }
    const size_t sections = std::stoul(get_arg(argc, argv, "--sections", "5"));
    const size_t paragraphs = std::stoul(get_arg(argc, argv, "--paragraphs", "2"));
    DocumentPlanner planner;
    const auto plan = planner.plan(topic, bundle.graph, bundle.codebook, bundle.language_rules,
                                   sections, paragraphs);
    const auto text = planner.realize(plan, *mouth, bundle.language_rules);
    if (text.empty()) { std::cerr << "Não há fatos suficientes sobre o tópico.\n"; return 1; }
    const auto output = get_arg(argc, argv, "--out");
    if (!output.empty()) { std::ofstream file(output); if (!file) return 1; file << text; }
    std::cout << text;
    return 0;
}

int cmd_math(int argc, char** argv) {
    const auto expression = get_arg(argc, argv, "--expr");
    if (expression.empty()) { print_usage(); return 1; }
    LanguageRules rules;
    const auto model = get_arg(argc, argv, "--model");
    if (!model.empty()) {
        ModelBundle bundle;
        if (!bundle.load(model, true)) return 1;
        rules = bundle.language_rules;
    } else rules.load_default();
    SymbolicMath math(rules);
    if (const auto equation = math.solve(expression)) {
        std::cout << equation->variable << " = " << equation->value.to_string() << "\n";
        std::cout << equation->steps << "\n";
        return 0;
    }
    if (const auto evaluated = math.evaluate(expression)) {
        std::cout << evaluated->value.to_string() << "\n";
        std::cout << evaluated->steps << "\n";
        return 0;
    }
    std::cerr << "Expressão não reconhecida.\n";
    return 1;
}

int cmd_chat(int argc, char** argv) {
    const std::string model_dir = get_arg(argc, argv, "--model");
    const bool debug = has_flag(argc, argv, "--debug");
    if (model_dir.empty()) {
        print_usage();
        return 1;
    }

    const auto workspace = get_arg(argc, argv, "--workspace");
    if (!workspace.empty() && !fs::is_directory(workspace)) { std::cerr << "Workspace inexistente.\n"; return 1; }
    const auto session_dir = workspace.empty() ? model_dir : (fs::path(workspace) / ".cognicore").string();
    ModelBundle bundle;
    if (!bundle.load(fs::exists(session_dir + "/codebook.bin") ? session_dir : model_dir, true)) {
        std::cerr << "Erro ao carregar modelo de " << model_dir << "\n";
        return 1;
    }
    load_rules_if_present(bundle.parser);

    if (!workspace.empty()) {
        bundle.code_graph = CodeGraph{};
        TreeSitterCodeParser::ingest_directory(workspace, bundle.code_graph, &bundle.sdm);
        fs::create_directories(session_dir);
    }
    const auto realizer = get_arg(argc, argv, "--realizer", "fluent");
    MorphologicalRealizer morphology;
    LexicalizedGrammarRealizer lexical(bundle.language_rules);
    const ISurfaceRealizer* mouth = nullptr;
    if (realizer == "strict") mouth = &morphology;
    else if (realizer == "lexical") mouth = &lexical;
    else if (realizer == "fluent") {
        if (!bundle.fluent_realizer.ready()) {
            std::cerr << "Realizador fluent ainda não treinado; usando strict. Execute train-realizer.\n";
            mouth = &morphology;
        } else mouth = &bundle.fluent_realizer;
    } else { std::cerr << "Realizador inválido: use strict, lexical ou fluent.\n"; return 1; }
    AssistantSession session(bundle);
    const auto language = string_to_code_lang(get_arg(argc, argv, "--lang", "cpp"));
    if (language == CodeLanguage::UNKNOWN) { std::cerr << "Linguagem inválida.\n"; return 1; }
    std::ofstream history(session_dir + "/conversation.jsonl", std::ios::app);
    std::cout << "CogniCore CPU-only chat pronto. Digite 'sair' para encerrar.\n";
    std::string line;
    while (true) {
        std::cout << "\nVoce> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line == "sair" || line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        auto reply = session.respond(line, language, mouth);
        auto trace = reply.trace;
        trace.response = reply.text;
        if (history) history << "{\"user\":\"" << json_escape(line) << "\",\"assistant\":\"" << json_escape(reply.text)
                             << "\",\"route\":\"" << route_name(reply.route) << "\"}\n" << std::flush;
        if (!bundle.save(session_dir)) { std::cerr << "Erro ao persistir sessão.\n"; return 1; }
        bundle.refresh_mapped_graph(session_dir);
        if (debug) {
            std::cout << "[DEBUG] Rota: " << route_name(reply.route) << "\n" << trace.debug_report();
        }
        std::cout << "CogniCore> " << trace.response << "\n";
    }
    return 0;
}

int cmd_ingest_code(int argc, char** argv) {
    const auto repo = get_arg(argc, argv, "--repo");
    const auto model = get_arg(argc, argv, "--model");
    if (!fs::is_directory(repo) || model.empty()) { print_usage(); return 1; }
    ModelBundle bundle;
    if (fs::exists(model + "/codebook.bin") && !bundle.load(model, false)) return 1;
    const auto count = TreeSitterCodeParser::ingest_directory(repo, bundle.code_graph, &bundle.sdm, has_flag(argc, argv, "--use-gpu"));
    if (!bundle.save(model)) return 1;
    std::cout << "[ingest-code] Funções: " << count << " | Arestas: " << bundle.code_graph.num_edges() << "\n";
    return 0;
}
int cmd_code(int argc, char** argv) {
    const auto repo = get_arg(argc, argv, "--workspace");
    const auto model = get_arg(argc, argv, "--model");
    const auto task = get_arg(argc, argv, "--task");
    const auto lang = string_to_code_lang(get_arg(argc, argv, "--lang"));
    if (!fs::is_directory(repo) || model.empty() || task.empty() || lang == CodeLanguage::UNKNOWN) { print_usage(); return 1; }
    ModelBundle bundle;
    if (!bundle.load(model, true)) return 1;
    bundle.code_graph = CodeGraph{};
    TreeSitterCodeParser::ingest_directory(repo, bundle.code_graph, &bundle.sdm, has_flag(argc, argv, "--use-gpu"));
    const auto result = CodeSynthesizer{}.synthesize(task, lang, bundle.code_graph, bundle.codebook);
    if (!result.syntax_valid) { std::cerr << result.error << "\n"; return 1; }
    std::cout << result.code;
    return 0;
}
int cmd_train_realizer(int argc, char** argv) {
    ModelBundle bundle;
    const auto model = get_arg(argc, argv, "--model");
    if (model.empty() || !bundle.load(model)) return 1;
    const bool gpu = has_flag(argc, argv, "--use-gpu");
    if (gpu && !backend::is_gpu_available()) { std::cerr << "GPU indisponível.\n"; return 1; }
    bundle.fluent_realizer.train(gpu);
    if (!bundle.save(model)) return 1;
    std::cout << "Realizador DOP/PPM/Tsetlin treinado: " << bundle.fluent_realizer.num_fragments()
              << " fragmentos, " << bundle.fluent_realizer.ppm_contexts() << " contextos PPM.\n";
    return 0;
}

} // namespace

int run_cli(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "ingest-books")   return cmd_ingest_books(argc, argv);
    if (cmd == "ingest-corpus")  return cmd_ingest_corpus(argc, argv);
    if (cmd == "train-dialogue") return cmd_train_dialogue(argc, argv);
    if (cmd == "evaluate-dialogue") return cmd_evaluate_dialogue(argc, argv);
    if (cmd == "consolidate")    return cmd_consolidate(argc, argv);
    if (cmd == "stats")          return cmd_stats(argc, argv);
    if (cmd == "ingest-code") return cmd_ingest_code(argc, argv);
    if (cmd == "code") return cmd_code(argc, argv);
    if (cmd == "train-realizer") return cmd_train_realizer(argc, argv);
    if (cmd == "evaluate-fluency") return cmd_evaluate_fluency(argc, argv);
    if (cmd == "write") return cmd_write(argc, argv);
    if (cmd == "math") return cmd_math(argc, argv);
    if (cmd == "chat" || cmd == "assistant")           return cmd_chat(argc, argv);

    print_usage();
    return 1;
}

int main(int argc, char** argv) {
    try { return run_cli(argc, argv); }
    catch (const std::exception& error) { std::cerr << "Erro: " << error.what() << "\n"; return 1; }
}
