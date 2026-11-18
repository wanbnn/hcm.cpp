#include "core/model_bundle.hpp"
#include "core/assistant.hpp"
#include "core/morphological_realizer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace cognicore;

namespace {

void print_usage() {
    std::cout << "Uso:\n"
              << "  cognicore ingest-books --input <dir_ou_txt> --model <dir_modelo> [--parser rules|conllu] [--use-gpu]\n"
              << "  cognicore train-dialogue --input <conversas.jsonl> --model <dir_modelo> [--epochs 20] [--use-gpu]\n"
              << "  cognicore consolidate --model <dir_modelo>\n"
              << "  cognicore stats --model <dir_modelo>\n"
              << "  cognicore ingest-code --repo <dir> --model <dir_modelo> [--use-gpu]\n"
              << "  cognicore code --model <dir_modelo> --workspace <dir> --lang cpp|py|js --task <descrição>\n"
              << "  cognicore export-realizer-dataset --model <dir_modelo> --out <pairs.jsonl>\n"
              << "  cognicore chat --model <dir_modelo> [--workspace <dir>] [--realizer symbolic|morphological] [--lang cpp|py|js] [--debug]\n";
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
    if (q1 == std::string::npos) return "";
    size_t q2 = q1 + 1;
    while (q2 < line.size()) {
        if (line[q2] == '"' && line[q2 - 1] != '\\') break;
        ++q2;
    }
    return line.substr(q1 + 1, q2 - q1 - 1);
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

    size_t total_triples = 0;
    for (const auto& fpath : files) {
        const std::string text = read_file_text(fpath);
        if (text.empty()) continue;

        auto doc = (parser_mode == "conllu" || fs::path(fpath).extension() == ".conllu")
            ? bundle.parser.parse_conllu(text) : bundle.parser.parse_text(text);
        for (const auto& s_toks : doc.sentences_tokens) {
            bundle.realizer.observe_sentence_tokens(s_toks);
            for (size_t i = 0; i < s_toks.size(); ++i) {
                std::vector<std::string> ctx;
                for (size_t j = (i >= 2 ? i - 2 : 0); j <= std::min(s_toks.size() - 1, i + 2); ++j) {
                    if (i != j) ctx.push_back(s_toks[j]);
                }
                bundle.codebook.observe_context(s_toks[i], ctx);
            }
        }
        for (const auto& tr : doc.triples) {
            bundle.graph.add_fact(tr.subject, tr.relation, tr.object, tr.type, tr.confidence, false,
                                  tr.negated, tr.tense, tr.modality, tr.subj_gender, tr.subj_number,
                                  tr.obj_gender, tr.obj_number);
            HyperVector addr = HyperVector::from_token(tr.subject);
            HyperVector data = HyperVector::from_token(tr.object);
            bundle.sdm.write(addr, data, use_gpu);
            ++total_triples;
        }
        for (auto& pair : realizer_document_pairs(doc)) bundle.realizer_pairs.push_back(std::move(pair));
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
              << " | Frames: " << bundle.realizer.num_frames() << "\n";
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

        auto toks = bundle.parser.tokenize(text);
        bundle.realizer.observe_sentence_tokens(toks);
        for (auto& pair : realizer_document_pairs(bundle.parser.parse_text(text)))
            bundle.realizer_pairs.push_back(std::move(pair));
        X.push_back(bundle.encode_dialogue_features(text));
        y.push_back(static_cast<size_t>(string_to_intent(intent_str)));
    }

    if (X.empty()) {
        std::cerr << "Nenhum exemplo valido encontrado em " << input << "\n";
        return 1;
    }

    bundle.tsetlin.fit(X, y, epochs, use_gpu);
    bundle.has_sparse_tsetlin = false;

    if (!bundle.save(model_dir)) {
        std::cerr << "Erro ao salvar modelo em " << model_dir << "\n";
        return 1;
    }

    std::cout << "[train-dialogue] Exemplos: " << X.size()
              << " | Epocas: " << epochs
              << " | GPU: " << (use_gpu && backend::is_gpu_available() ? "HIP/ROCm" : "CPU") << "\n";
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
              << bundle.graph.num_edges() << " arestas\n"
              << "  Realizador linguistico : " << bundle.realizer.num_frames() << " frames, "
              << bundle.realizer.num_ngrams() << " n-gramas\n"
              << "  Tamanho em disco       : " << (disk_bytes / 1024.0) << " KB\n"
              << "  RAM estimada           : " << (ram_bytes / 1024.0) << " KB\n";
    return 0;
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
    const auto realizer = get_arg(argc, argv, "--realizer", "symbolic");
    MorphologicalRealizer morphology;
    const ISurfaceRealizer* mouth = nullptr;
    if (realizer == "morphological") mouth = &morphology;
    else if (realizer != "symbolic") {
        std::cerr << "Realizador inválido: use symbolic ou morphological.\n"; return 1;
    }
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
        if (debug) {
            std::cout << "[DEBUG] Rota: " << route_name(reply.route) << " | Conceitos VSA: [";
            for (size_t i = 0; i < trace.activated_concepts.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << trace.activated_concepts[i];
            }
            std::cout << "] -> Intencao Tsetlin: [" << intent_to_string(trace.intent) << "]"
                      << " -> Caminhos Grafo: " << trace.paths.size();
            if (!trace.paths.empty()) {
                std::cout << " (";
                for (size_t i = 0; i < trace.paths[0].steps.size(); ++i) {
                    const auto& st = trace.paths[0].steps[i];
                    if (i > 0) std::cout << " -> ";
                    std::cout << st.src << " -[" << st.rel << "]-> " << st.dst;
                }
                std::cout << ")";
            }
            std::cout << "\n";
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
int cmd_export_pairs(int argc, char** argv) {
    ModelBundle bundle;
    const auto model = get_arg(argc, argv, "--model");
    const auto output = get_arg(argc, argv, "--out");
    if (model.empty() || output.empty() || !bundle.load(model)) return 1;
    std::ofstream out(output);
    if (!out) return 1;
    for (const auto& pair : bundle.realizer_pairs) out << pair << '\n';
    std::cout << "Pares exportados: " << bundle.realizer_pairs.size() << "\n";
    return out ? 0 : 1;
}

} // namespace

int run_cli(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    const std::string cmd = argv[1];
    if (cmd == "ingest-books")   return cmd_ingest_books(argc, argv);
    if (cmd == "train-dialogue") return cmd_train_dialogue(argc, argv);
    if (cmd == "consolidate")    return cmd_consolidate(argc, argv);
    if (cmd == "stats")          return cmd_stats(argc, argv);
    if (cmd == "ingest-code") return cmd_ingest_code(argc, argv);
    if (cmd == "code") return cmd_code(argc, argv);
    if (cmd == "export-realizer-dataset") return cmd_export_pairs(argc, argv);
    if (cmd == "chat" || cmd == "assistant")           return cmd_chat(argc, argv);

    print_usage();
    return 1;
}

int main(int argc, char** argv) {
    try { return run_cli(argc, argv); }
    catch (const std::exception& error) { std::cerr << "Erro: " << error.what() << "\n"; return 1; }
}
