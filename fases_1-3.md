# Missão: Construir um Motor Conversacional Generativo Pós-LLM (VSA + Kanerva SDM + Tsetlin Machine + Grafo Causal)

Você é um Arquiteto de Sistemas de IA e Engenheiro de Baixo Nível (C++20, HIP/ROCm, SIMD).
Sua missão é projetar e implementar do zero um **modelo conversacional generativo ultra-leve**, sem backpropagation tradicional em ponto flutuante e sem arquitetura Transformer/LLM densa.

## 1. Restrições Rígidas de Arquitetura (O que NÃO fazer e O que FAZER)
- **PROIBIDO:** Usar Transformers, llama.cpp, redes neurais densas com backpropagation em float16/32 como cérebro do sistema, ou gerar respostas apenas buscando frases prontas (FAQ/retrieval puro).
- **OBRIGATÓRIO:**
  1. **Memória e Associação:** Vetores Hiperdimensionais Binários (**VSA / Binary Spatter Codes**, `D = 4096` bits empacotados em `64 x uint64_t`) + **Memória Esparsa Distribuída de Kanerva (SDM)**.
  2. **Raciocínio e Política de Diálogo:** **Tsetlin Machine** (autômatos inteiros `int8` aprendendo cláusulas lógicas booleanas bit a bit, sem ponto flutuante) integrada a um **Grafo Causal** em disco/mmap.
  3. **Fluência Generativa ("A Boca"):** Um **Realizador Composicional** baseado em **Gramática de Construções (Construction Grammar) + Pontuador Estatístico Leve (CTW / N-grama condicionado)** que sintetiza frases *inéditas* combinando esquemas sintáticos abstratos com os conceitos inferidos pelo Grafo/VSA. *(Crie a interface `ISurfaceRealizer` desacoplada para permitir plugar outros realizadores no futuro).*
  4. **Hardware:**
     - **Treino/Indexação:** Suporte nativo a **AMD HIP/ROCm** (`hipcc`) para aceleração massiva em GPU de distância de Hamming (`XOR` + `__popcll`) e atualização de cláusulas Tsetlin, com **fallback completo para CPU** (C++20 + OpenMP + `__builtin_popcountll` / AVX2) via flag de CMake (`-DUSE_ROCM=ON/OFF`).
     - **Inferência:** Deve rodar **100% em CPU-only**, usando pouca RAM (< 1 GB para protótipos médios), usando operações bitwise e `mmap` para o grafo/construções em disco.

---

## 2. Estrutura do Projeto e Módulos

Crie um projeto CMake organizado da seguinte forma:

```text
/project-root
├── CMakeLists.txt               # Suporta -DUSE_ROCM=ON/OFF, C++20, OpenMP, -march=native
├── include/core/
│   ├── backend.hpp              # Abstração CPU vs HIP/ROCm
│   ├── vsa.hpp                  # Vetores 4096-bit: random, bind(XOR), permute(shift), bundle(majority), hamming
│   ├── sdm.hpp                  # Kanerva SDM: endereços binários, contadores int8, read/write/binarize
│   ├── tsetlin.hpp              # Tsetlin Machine: cláusulas, literais, Type I/II feedback, export esparso
│   ├── causal_graph.hpp         # Grafo Causal (nós, arestas tipadas, confiança, travessia multi-hop)
│   ├── text_parser.hpp          # Extrator leve UTF-8 (triplas SVO/causais + frames de construção sintática)
│   ├── realizer.hpp             # ISurfaceRealizer + ConstructionGrammarRealizer (geração composicional)
│   └── model_bundle.hpp         # Serialização/carregamento binário compacto e mmap
├── src/
│   ├── backends/
│   │   ├── cpu_backend.cpp      # Operações bitwise vetorizadas + OpenMP
│   │   └── hip_backend.hip      # Kernels HIP: batch Hamming, SDM read/write, Tsetlin clause eval & update
│   ├── vsa.cpp
│   ├── sdm.cpp
│   ├── tsetlin.cpp
│   ├── causal_graph.cpp
│   ├── text_parser.cpp
│   ├── realizer.cpp
│   ├── model_bundle.cpp
│   └── main.cpp                 # CLI unificada (ingest, train-dialogue, consolidate, chat)
├── scripts/
│   └── prepare_dataset.py       # (Opcional) Utilitário leve para limpar livros (.txt) e diálogos (.jsonl)
├── data/                        # Regras linguísticas configuráveis (ex: conectivos causais PT-BR / EN)
└── tests/                       # Testes unitários e de integração (CTest)
```

---

## 3. Especificação Técnica Detalhada de Cada Módulo

### A. VSA (`vsa.hpp`)
- Dimensão fixa configurável em compile-time/config (`D = 4096` bits = `64 x uint64_t`).
- Operações:
  - `bind(a, b)`: `XOR` bit a bit (associa papel e valor, ex: `SUJEITO ^ FOGO`).
  - `permute(a, k)`: rotação circular de bits (codifica ordem sequencial/sintática).
  - `bundle(list_of_vectors)`: soma por coluna e limiar majoritário com desempate aleatório determinístico.
  - `similarity(a, b)`: `1.0 - (2.0 * hamming_dist / D)`.
- **Random Indexing:** Ao ingerir livros, construa um `Codebook` onde cada token/lema recebe um vetor aleatório fixo e um vetor semântico acumulado pelos vizinhos na janela de contexto.

### B. Kanerva SDM (`sdm.hpp`)
- `M` endereços físicos (hard locations) com vetores de endereço de `D` bits e contadores de conteúdo `int8_t` (saturados entre `-127` e `+127`).
- **Write(address, data):** Ativa todos os endereços dentro de um raio de Hamming (ou top-K mais próximos) e soma `+1/-1` nos contadores conforme os bits de `data`.
- **Read(address):** Soma os contadores dos endereços ativos e aplica sinal (`> 0 -> 1`, `<= 0 -> 0`).
- **Compactação:** Função `binarize_for_inference()` opcional para reduzir memória na inferência.
- **HIP Kernel:** Paralelize a busca de Hamming sobre todos os `M` slots usando `__popcll` e redução por wavefront/bloco.

### C. Tsetlin Machine (`tsetlin.hpp`)
- Entrada: Vetor binário combinando `[VSA do input do usuário (D bits) + Bits de estado do grafo/contexto]`.
- Saída: Classes de **Ação Dialógica / Intenção Cognitiva** (ex: `EXPLAIN_CAUSE`, `ANSWER_FACT`, `COMPARE`, `ASK_CLARIFY`, `AGREE`, `HYPOTHESIZE`, `GREET`) + seleção de relações-alvo no grafo.
- Implemente:
  - Estados de autômatos `int8_t` (`1..2*N_STATES`, threshold `N_STATES`).
  - Avaliação de cláusulas em paralelo (CPU e HIP).
  - Feedback Tipo I (reforçar padrões verdadeiros) e Tipo II (suprimir falsos positivos).
  - **Exportação Esparsa:** Ao salvar o modelo treinado, exporte apenas os índices dos literais incluídos em cada cláusula (`uint16_t`), reduzindo o tamanho em ~8x–20x para inferência ultrarrápida na CPU.

### D. Grafo Causal + Extração de Texto (`causal_graph.hpp`, `text_parser.hpp`)
- Estrutura de nós (conceitos/lemas) e arestas direcionadas: `(src_id, relation_id, dst_id, frequency, confidence)`.
- Arquivo de configuração JSON para marcadores linguísticos (suporte inicial a **Português** e **Inglês**): conectivos causais (*"porque", "causa", "gera", "impede", "logo", "se... então"*), relações taxonômicas (*"é um", "tem", "faz parte"*), e papéis SVO básicos.
- **Travessia (Raciocínio Multi-hop):** Dado um conjunto de conceitos iniciais extraídos da pergunta + associações da SDM, faça busca em grafo (beam search / spreading activation ponderado por confiança e similaridade VSA) para derivar cadeias causais inéditas (ex: `A -> B -> C` => inferir que `A` leva a `C`).

### E. Realizador Generativo Composicional (`realizer.hpp`)
- Durante a ingestão de livros e diálogos, o `text_parser` extrai **Frames de Construção Abstratos** substituindo entidades/ações específicas por slots tipados (ex: `"[SUBJ:ENT] costuma provocar [OBJ:ENT] quando [COND:CLAUSE]"`), além de estatísticas locais de transição (CTW / bigramas e trigramas condicionados para coesão e concordância).
- **Na inferência (Geração Real):**
  1. A Tsetlin Machine escolhe a intenção dialógica.
  2. O Grafo + VSA geram o **Plano Semântico** (uma ou mais triplas/caminhos lógicos inéditos que respondem ao usuário).
  3. O Realizador seleciona e combina dinamicamente frames compatíveis com o plano semântico, preenche os slots, ajusta conectivos/concordância básica e pontua fluência via n-grama/CTW — **gerando sentenças novas que não existiam literalmente no dataset**.

### F. Consolidação ("Sono") e Aprendizado Online
- Comando `consolidate`: funde arestas duplicadas no grafo, promove propriedades compartilhadas em hierarquias (`é_um`), poda ruído de baixa frequência/confiança e reindexa construções redundantes.
- Durante o modo `chat`: cada turno do usuário atualiza a memória episódica (Kanerva SDM) e adiciona novos fatos declarados ao Grafo em **tempo real** (aprendizado *one-shot* sem retreinar a Tsetlin Machine).

---

## 4. Pipeline de CLI Esperado (`main.cpp`)

O binário final (ex: `./cognicore`) deve suportar os seguintes subcomandos:
1. `./cognicore ingest-books --input <dir_ou_txt> --model <dir_modelo> [--use-gpu]`
   - Faz Random Indexing (VSA), extrai triplas para o Grafo Causal, popula a SDM com contextos e extrai Frames de Construção + estatísticas CTW/n-gramas.
2. `./cognicore train-dialogue --input <conversas.jsonl> --model <dir_modelo> --epochs 20 [--use-gpu]`
   - Treina a Tsetlin Machine (política dialógica e mapeamento de intenções) e enriquece construções conversacionais.
3. `./cognicore consolidate --model <dir_modelo>`
   - Executa o passo de "sono" (poda, fusão e exportação esparsa da Tsetlin Machine + compactação binária).
4. `./cognicore stats --model <dir_modelo>`
   - Exibe métricas do modelo: bits VSA, ocupação/capacidade da SDM, nº de cláusulas Tsetlin ativas, nós/arestas no Grafo, frames linguísticos, e tamanho total em disco e RAM estimada.
5. `./cognicore chat --model <dir_modelo>`
   - Loop interativo CPU-only mostrando (com flag `--debug`) o raciocínio interno antes da resposta: `[Conceitos VSA ativados] -> [Intenção Tsetlin] -> [Caminho no Grafo Causal] -> [Resposta Gerada]`.

---

## 5. Plano de Execução Passo a Passo (Não pule etapas!)

Implemente e **valide com testes automatizados (`ctest`)** nesta ordem exata:
1. **Marco 1:** `CMakeLists.txt` + `vsa` + `cpu_backend` + testes de álgebra VSA (ortogonalidade, recuperação com `bind`/`unbind`, similaridade após `bundle`).
2. **Marco 2:** `sdm` + testes de capacidade de armazenamento e recuperação sob ruído (escrever N padrões aleatórios e medir taxa de acerto bit a bit).
3. **Marco 3:** `tsetlin` + testes de convergência lógica (ex: resolver XOR/padrões ruidosos com Tsetlin Machine e validar exportação esparsa idêntica ao modelo completo).
4. **Marco 4:** `hip_backend.hip` implementando os kernels de Hamming, SDM e Tsetlin, com teste de equivalência numérica exata contra o `cpu_backend`. *(Se o ambiente atual não tiver GPU AMD instalada, garanta que o código `.hip` seja sintaticamente válido e que todos os testes rodem pelo `cpu_backend`).*
5. **Marco 5:** `text_parser`, `causal_graph`, `realizer` e `model_bundle`.
6. **Marco 6:** CLI completa + **Teste End-to-End** usando um mini-dataset sintético embutido nos testes (ex: um pequeno texto explicando física básica + 20 exemplos de diálogo) provando que o sistema:
   - Ingere o texto,
   - Treina a Tsetlin,
   - Salva o modelo em disco,
   - Recarrega em um novo processo CPU-only,
   - Responde a uma pergunta combinando **dois fatos de frases diferentes** (raciocínio multi-hop) em uma frase **gerada dinamicamente** que não existia no texto original,
   - E aprende um fato novo dito pelo usuário durante o chat.

---

# Missão Fase 2: Evoluir o HCM para Assistente Completo (Fluência Nível A/B + Geração de Código por AST)

Inspecione o repositório atual (núcleo VSA + Kanerva SDM + Tsetlin Machine + Grafo Causal + CLI `cognicore`).
Agora vamos evoluir o sistema da prova de conceito inicial para um **Assistente Conversacional e de Código completo**, mantendo nossa filosofia central intacta.

## 1. Princípio Arquitetural Inegociável
- **O cérebro continua 100% pós-LLM:** Fatos, memória episódica de projeto, lógica, causalidade e tomada de decisão continuam **exclusivamente** no `VSA + Kanerva SDM + Tsetlin Machine + Grafo` (zero APIs externas, zero LLMs gigantes).
- **Desacoplamento estrito da "Boca":** Qualquer componente neural entra **apenas** atrás da interface `ISurfaceRealizer` como um tradutor cego *Grafo/Plano Semântico -> Texto Fluente* (~30M–50M params, incapaz de raciocinar ou lembrar fatos sozinho).
- **Inferência alvo:** 100% **CPU-only**, baixa RAM (< 1.5 GB total com tudo carregado), inicialização rápida via `mmap`.

---

## 2. Os 4 Pilares desta Evolução

### Pilar 1: Parser Morfossintático Real + Realizador Nível A (0% Neural)
Substitua o parser ingênuo por conectivos fixos e o realizador simples de templates:
1. **Parser de Dependências (`text_parser` v2):**
   - Integre suporte a análise morfossintática real (via **UDPipe C++** embutido via CMake `FetchContent` com modelos leves PT/EN, ou ingestão direta de fluxos **CoNLL-U** gerados por pré-processador Python leve + fallback de regras).
   - Extraia não apenas palavras soltas, mas: **Lema, POS-tag (Substantivo, Verbo, Adjetivo...), Tempo Verbal, Gênero, Número, Negação e Relações de Dependência (`nsubj`, `obj`, `obl`, `advcl`, `mark`)**.
   - Grave no Grafo Causal triplas ricas com metadados gramaticais e polaridade (ex: negação ou modalidade como *"pode causar"*, *"nunca causa"*).
2. **Gerador Morfológico e de Unificação (`MorphologicalRealizer`, Nível A):**
   - Implemente uma nova classe `ISurfaceRealizer` inspirada em *SimpleNLG / Meaning-Text Theory*:
     - **Planejador de Discurso:** Agrupa múltiplos *hops* do Grafo Causal em uma ordem retórica coerente (ex: *Tese -> Causa -> Consequência -> Exceção*), inserindo conectivos adequados sem repetição robótica.
     - **Anáfora / Pronomes:** Detecta quando o mesmo sujeito aparece em sentenças seguidas e substitui automaticamente por pronome (*"ele", "ela", "isso", "esse processo"*) ou elipse.
     - **Motor Morfológico (PT-BR e EN):** Tabela/regras de conjugação verbal (presente, passado, futuro, subjuntivo, infinitivo) e concordância nominal de gênero/número + contração de preposições (*"de + o -> do", "em + a -> na"*).

### Pilar 2: Realizador Nível B — Micro-Decodificador *Graph-to-Text* (~30M–50M params)
Para atingir fluência conversacional natural mantendo o cérebro simbólico:
1. **Gerador Automático de Dataset *Plan-to-Text* (`export-realizer-dataset`):**
   - Durante a leitura de livros e diálogos já ingeridos, o sistema extrai pares `(Plano_Semântico_Serializado -> Frase_Original_Fluente)`.
   - Formato do Plano Semântico de entrada (tokens estruturados compactos):
     `<PLAN> <INTENT: EXPLAIN> <TONE: NATURAL> <SUBJ: fogo> <REL: gera> <OBJ: calor> <COND: oxigenio> </PLAN>`
2. **Treino Offline Leve em Python + PyTorch (ROCm/AMD):**
   - Crie `scripts/train_micro_decoder.py` implementando um modelo minúsculo (~25M–50M params, ex: **Tiny RWKV-v5/v6** ou **Tiny GQA Transformer com RoPE** e vocabulário BPE/Unigram pequeno de 8k–16k tokens focado no idioma).
   - O modelo **não** faz pré-treino de conhecimento geral; ele treina apenas a tarefa *Sequence-to-Sequence / Prefix-LM* de traduzir `<PLAN>...</PLAN>` em texto natural fluente.
   - Exporte os pesos quantizados (`Q4_0` ou `int8` / `1.58-bit`) para um arquivo binário compacto (`mouth_q4.bin`, ~20–40 MB).
3. **Motor de Inferência C++ Puro (`NeuralMouthRealizer : public ISurfaceRealizer`):**
   - Escreva um executor em **C++20 puro (zero dependência de PyTorch/libtorch)** usando OpenMP/AVX2 em CPU para rodar `mouth_q4.bin` em milissegundos, mantendo o fluxo:
     `Input -> VSA/SDM + Tsetlin + Grafo -> Plano Semântico Inédito -> NeuralMouthRealizer -> Resposta Fluente`.
   - Permita alternar via flag `--realizer [symbolic|neural]` na CLI.

### Pilar 3: Motor de Código por AST (`Tree-sitter` + Síntese Estrutural)
Em vez de tratar código fonte como texto corrido, trate código como **Árvore Sintática Abstrata (AST)** + **Grafo de Chamadas e Tipos**:
1. **Ingestão de Repositórios (`ingest-code`):**
   - Integre **Tree-sitter** (biblioteca C oficial via CMake `FetchContent`) com gramáticas iniciais para **C++**, **Python** e **JavaScript/TypeScript**.
   - Extraia de repositórios reais:
     - Assinaturas de funções/métodos, parâmetros, tipos de retorno, imports/includes e docstrings.
     - Sub-árvores AST canônicas (padrões idiomáticos, ex: abrir arquivo com segurança, loop sobre coleção, tratamento `try/catch`, requisição HTTP).
     - **Grafo de Código:** Arestas `CALLS`, `RETURNS_TYPE`, `TAKES_PARAM`, `MEMBER_OF`, `IMPLEMENTS`, `DEPENDS_ON`.
   - Codifique cada função/padrão AST em **vetores VSA composicionais** (ex: `Bind(LANG, CPP) ^ Bind(INTENT, READ_FILE) ^ Bind(RET, STRING)`) e grave na **Kanerva SDM** para recuperação associativa instantânea de todo um projeto grande sem limite de janela de contexto!
2. **Planejador e Sintetizador de Código (`CodeSynthesizer`):**
   - A **Tsetlin Machine** (treinada em pares `Pedido/Intenção + Contexto do Projeto -> Padrões AST necessários`) decide quais blocos AST e APIs reais do grafo compor.
   - O **Compositor de AST** conecta os blocos selecionados verificando compatibilidade de tipos no Grafo, preenche identificadores/variáveis no escopo correto, resolve `#include` / `import` necessários e gera o código final via **AST Pretty-Printer**.
   - **Garantia Estrutural:** Antes de imprimir para o usuário, valide o código gerado passando-o novamente pelo `tree-sitter` para garantir **0 erros de sintaxe**.

### Pilar 4: Roteador de Assistente Unificado e Memória de Projeto
1. Crie uma **Tsetlin Machine Roteadora** no comando `chat` / `assistant` que classifica cada entrada do usuário em:
   - `CONVERSATIONAL_GENERAL` (diálogo social / esclarecimento / opinião baseada na SDM),
   - `CAUSAL_REASONING` (explicação multi-hop via Grafo Causal + VSA),
   - `CODE_QUERY_OR_EXPLAIN` (explicar como uma função/módulo do repositório ingerido funciona usando o Grafo de Código),
   - `CODE_GENERATE_OR_PATCH` (sintetizar função/trecho/refatoração via `CodeSynthesizer`),
   - `FACT_UPDATE` (aprendizado *one-shot* imediato na SDM e Grafo).
2. Adicione suporte a **sessões persistentes**: salvar/carregar a memória episódica da conversa e índices do repositório local do usuário (`--workspace <dir>`).

---

## 3. Novos Comandos de CLI Esperados (`cognicore`)

- `./cognicore ingest-books --input <txt_ou_conllu> --model <dir> [--parser udpipe|rules] [--use-gpu]`
- `./cognicore ingest-code --repo <dir_codigo> --model <dir> [--use-gpu]`
- `./cognicore export-realizer-dataset --model <dir> --out <plan_pairs.jsonl>`
- `./cognicore chat --model <dir> [--workspace <dir_repo>] [--realizer symbolic|neural] [--debug]`
- `./cognicore code --model <dir> --workspace <dir_repo> --lang <cpp|py|js> --task "<descricao>"`

---

## 4. Plano de Execução e Validação (Marcos 7 a 11)

Implemente incrementalmente, mantendo todos os testes antigos passando (`ctest`) e adicionando novos testes para cada marco:

1. **Marco 7 (Parser + Realizador Nível A):**
   - Parser morfossintático extraindo relações reais com negação/tempo verbal + `MorphologicalRealizer` com conjugação PT-BR/EN, concordância e substituição pronominal.
   - *Teste:* Gerar respostas multi-frase a partir de um grafo de 3 hops sem repetição mecânica do sujeito e com concordância correta.
2. **Marco 8 (`ingest-code` com Tree-sitter + VSA-AST):**
   - Compilar `tree-sitter` + gramáticas C++/Python/JS via CMake.
   - *Teste:* Ingerir o próprio código-fonte do projeto `cognicore` e responder via Grafo/VSA quais módulos chamam `sdm::read` e quais tipos recebem.
3. **Marco 9 (`CodeSynthesizer` por AST):**
   - Tsetlin + Grafo selecionando e compondo padrões AST + Pretty-Printer + validação sintática automática via `tree-sitter`.
   - *Teste:* Ingerir uma mini-biblioteca exemplo (ex: leitura de arquivos e manipulação de vetores em C++ e Python), pedir uma função nova combinando duas operações existentes e validar que o código gerado passa pelo `tree-sitter` com **zero nós `ERROR`**.
4. **Marco 10 (Nível B — Micro-Decodificador *Graph-to-Text*):**
   - `export-realizer-dataset` + `scripts/train_micro_decoder.py` (PyTorch/ROCm) + exportador `Q4` + `NeuralMouthRealizer` em C++20 puro (CPU).
   - *Teste:* Treinar um modelo minúsculo de fumaça (*smoke test*) num mini-dataset de planos semânticos e comprovar que o motor C++ carrega o binário quantizado (< 50 MB RAM) e gera frases fluentes inéditas na CPU.
5. **Marco 11 (Assistente Unificado & Teste End-to-End Completo):**
   - Roteador Tsetlin unificando conversa, raciocínio causal, aprendizado *one-shot* e assistência de código sobre um `--workspace` real.
   - *Teste End-to-End:* Simular uma sessão de chat contínua onde o usuário: (1) ensina uma regra nova ao vivo, (2) faz uma pergunta causal multi-hop, e (3) pede um trecho de código usando uma função real do workspace ingerido.

Antes de escrever código, inspecione os arquivos existentes no repositório e apresente um plano breve de compatibilidade com as interfaces atuais (`VSA`, `SDM`, `Tsetlin`, `CausalGraph`, `ISurfaceRealizer`), iniciando em seguida pelo **Marco 7**.

---

A maior diferença filosófica para um LLM (como o DeepSeek-R1 ou o1, que ficam "pensando" gerando centenas de palavras num monólogo interno) é que **o nosso HCM pensa em silêncio — usando bits, cláusulas booleanas e grafos** — e só traduz o pensamento para português no último milissegundo.

Se você rodar `./cognicore chat --debug`, a saída no terminal ficaria parecida com estes dois cenários reais:

---

### Cenário 1: Memória ao Vivo + Raciocínio Causal Multi-Hop

Imagine que há dois minutos você disse ao chat: *"Meu servidor antigo só tem 2 GB de RAM e roda em CPU."* Agora você faz uma pergunta nova:

```text
Você> Se eu tentar carregar um grafo gigante inteiro na RAM lá sem mmap, o que acontece?

[1. PARSER MORFOSSINTÁTICO] (1.8 ms)
  ├─ tokens_chave: [tentar, carregar, grafo, gigante, inteiro, RAM, lá, sem, mmap]
  ├─ anáfora_resolvida: "lá" -> [servidor_antigo] (SDM Episódica, sim=0.41)
  └─ estrutura: COND(carregar(grafo_gigante, RAM) & NOT(mmap)) -> ASK_EFFECT

[2. VSA + KANERVA SDM (Memória Associativa)] (2.4 ms | 4096-bit XOR/POPCNT)
  ├─ conceito("grafo_gigante")  -> ativa [estrutura_dados, alto_volume_memória] (d_H=1420)
  ├─ conceito("servidor_antigo") -> recupera episódio #14:
  │     ↳ Bind(SERVIDOR_ANTIGO, LIMITE_RAM) = "2_GB" (conf=0.94, active_slots=18)
  └─ conceito("sem mmap")       -> inibe [leitura_sob_demanda_disco] -> ativa [alocação_total_RAM]

[3. TSETLIN MACHINE (Roteador & Política)] (0.9 ms | 12.400 cláusulas esparsas)
  ├─ Rota vencedora: CAUSAL_REASONING (votos: +84 | margem: +51)
  ├─ Cláusula #312 disparou: IF (ctx_hardware AND cond_hipotetica AND NOT code_req) -> INTENT=WARN_CONSEQUENCE
  └─ Cláusula #709 disparou: IF (has_negation("sem mmap")) -> INCLUDE_ALTERNATIVE(mmap)

[4. GRAFO CAUSAL (Travessia Multi-Hop)] (3.1 ms | mmap em disco)
  ├─ Hop 1: [grafo_gigante] --(sem_mmap_exige)--> [alocação_total_RAM] (conf=0.91)
  ├─ Hop 2: [alocação_total_RAM] --(excede)--> [limite_2GB_servidor_antigo] (conf=0.89)
  ├─ Hop 3: [memória_excedida] --(causa)--> [swap_em_disco OU travamento_OOM] (conf=0.96)
  └─ Hop 4 (Alternativa): [mmap] --(mitiga)--> [alocação_total_RAM] (conf=0.93)

[5. PLANO SEMÂNTICO GERADO] (Zero palavras geradas até aqui!)
  <PLAN>
    <INTENT: WARN_CONSEQUENCE>
    <CTX: servidor_antigo (2GB RAM)>
    <CAUSE: carregar(grafo_gigante, RAM) + WITHOUT(mmap)>
    <EFFECT_1: exceder(2GB_RAM)>
    <EFFECT_2: lentidão_extrema(swap) OR crash(OOM_Killer)>
    <RECOMMENDATION: usar(mmap) BECAUSE mantém_dados(disco)>
  </PLAN>

[6. REALIZADOR ("A Boca")]
```
Aqui você vê na prática a diferença entre os dois realizadores que discutimos para transformar **exatamente o mesmo `<PLAN>` acima** em texto:

*   **Com Nível A (`MorphologicalRealizer`, 0% neural — ~2 ms):**
    > *"No seu servidor antigo de 2 GB de RAM, carregar um grafo gigante inteiro sem mmap vai exceder a memória disponível. Isso causa lentidão extrema por swap ou um travamento por falta de memória. Recomenda-se usar mmap porque ele mantém os dados em disco."*
*   **Com Nível B (`NeuralMouthRealizer`, 40M params Q4 na CPU — ~48 ms):**
    > *"Como esse seu servidor só tem 2 GB de RAM, tentar subir um grafo gigante inteiro direto para a memória vai estourar o limite rapidinho — você vai acabar caindo em swap pesado ou tomando um crash de falta de memória (OOM). O ideal aí é ativar o mmap para ir lendo tudo direto do disco sob demanda."*

*(Tempo total de pensamento + fala em CPU comum: ~10 ms no Nível A ou ~56 ms no Nível B!)*

---

### Cenário 2: Assistente de Código (Tree-sitter + VSA-AST + Grafo de Código)

Agora você pede código sobre um projeto que ingeriu com `ingest-code --repo ./meu_projeto`:

```text
Você> Cria uma função em C++ pra abrir o arquivo de cláusulas e retornar falso se ele estiver corrompido.

[1. PARSER + VSA-CODE] (2.1 ms)
  ├─ lang_detect: CPP | task: [abrir_arquivo, validar_integridade, retornar_bool]
  └─ VSA Query: Bind(LANG, CPP) ^ Bind(TARGET, "clausulas") ^ Bind(ACTION, LOAD_VALIDATE)

[2. TSETLIN MACHINE (Roteador & Padrões AST)] (1.2 ms)
  ├─ Rota: CODE_GENERATE_OR_PATCH (votos: +112)
  └─ Cláusulas AST disparadas:
      ├─ #1044: IF (lang_cpp AND file_io) -> SELECT_PATTERN(STD_IFSTREAM_BINARY)
      ├─ #1089: IF (check_corrupted)     -> SELECT_PATTERN(MAGIC_HEADER_AND_SIZE_CHECK)
      └─ #1102: IF (return_bool_on_fail) -> SELECT_PATTERN(EARLY_RETURN_FALSE)

[3. GRAFO DE CÓDIGO DO WORKSPACE] (2.8 ms)
  ├─ Match no projeto: struct `ClauseHeader` em `include/core/tsetlin.hpp`
  │     ↳ campos reais: { uint32_t magic = 0x54534C4E; uint32_t num_clauses; uint32_t checksum; }
  ├─ Dependência encontrada: `std::ifstream` (<fstream>), `uint32_t` (<cstdint>)
  └─ Assinatura compatível: `bool load_clauses(const std::string& path, TsetlinModel& out)`

[4. COMPOSITOR AST + VALIDAÇÃO TREE-SITTER] (4.5 ms)
  ├─ Montando AST: FunctionDef -> VarDecl(ifstream) -> If(!is_open) -> Read(ClauseHeader) -> If(magic != 0x54534C4E) -> Return(true)
  └─ Validação Tree-sitter: 0 nós ERROR | 100% tipos resolvidos no grafo local.

[5. SAÍDA GERADA]
```
```cpp
#include <fstream>
#include <string>
#include <cstdint>
#include "core/tsetlin.hpp"

bool load_clauses_safe(const std::string& path, TsetlinModel& out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    ClauseHeader header{};
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(ClauseHeader))) {
        return false;
    }

    if (header.magic != 0x54534C4E || header.num_clauses == 0) {
        return false;
    }

    return out.read_payload(file, header);
}
```
> *"Usei a `ClauseHeader` definida em `include/core/tsetlin.hpp` validando o magic number `0x54534C4E` e a leitura binária antes de carregar o payload."*

---

### O que esse `--debug` revela de poderoso?
1. **Zero "caixa preta":** Se um LLM de 27B alucinar um campo que não existe no seu código, você não faz ideia do porquê. Aqui, você olha o log `[3. GRAFO]` e vê exatamente qual nó, qual cláusula Tsetlin e qual episódio da SDM geraram a resposta.
2. **Conserto cirúrgico sem retreinar tudo:** Se ele errou um fato no passo 4, você não precisa gastar horas retreinando pesos — basta corrigir ou ensinar a relação no chat e a SDM/Grafo atualizam na hora.
3. **Velocidade absurda em CPU:** Porque o "pensamento" (passos 1 a 4) leva **menos de 10 milissegundos** em operações bit a bit e grafos, sem precisar cuspir 500 tokens de `<think>` em texto antes de começar a responder.

---

# Correção Arquitetural (Fase 2 — Marco 10): Zero PyTorch, Zero Transformers

Vamos corrigir a especificação do **Marco 10 (Realizador Nível B)** para manter o projeto **100% fiel à nossa arquitetura pós-LLM**:
- **REMOVA** qualquer menção a `PyTorch`, `transformers`, `scripts/train_micro_decoder.py`, atenção neural ou treinamento em Python.
- Todo o projeto (treino, indexação, raciocínio e fluência) deve permanecer em **C++20 + HIP/ROCm puro** dentro do binário único `cognicore`.

## Novo Design do Realizador Nível B: `AnalogicalFluencyRealizer` (100% Não-Neural)

Em vez de um decodificador Transformer, implemente um realizador generativo de alta fluência baseado em **Recombinação Analógica de Sub-árvores (Data-Oriented Parsing / DOP) + Rankeamento por Compressão (CTW) + Tsetlin/VSA**, construído sobre o motor morfológico do Marco 7:

### 1. Extração de "Árvores de Fraseado Expressivo" (Durante `ingest-books` e `train-dialogue`)
- Não extraia apenas esqueletos secos de Sujeito-Verbo-Objeto.
- Extraia **sub-árvores sintáticas expressivas** (fragmentos com ritmo, conectivos discursivos, expressões idiomáticas e marcadores de tom), deslexicalizadas em slots tipados, e indexe cada sub-árvore com um **Vetor Hiperdimensional de Estilo/Contexto (VSA)**:
  - `Bind(INTENT, ...) ^ Bind(TONE, CASUAL|DIDATIC|WARNING|EMPATHETIC) ^ Bind(SHAPE, MULTI_HOP_CAUSE)`
- Grave essas sub-árvores na **Kanerva SDM** e em um banco compacto em disco (`mmap`).

### 2. Síntese por Recombinação Analógica + Beam Search (`generate_candidates`)
Dado um `<PLAN>` inédito vindo do Grafo Causal / Código:
1. **Busca Analógica (VSA + SDM):** Consulte a SDM para recuperar múltiplas sub-árvores humanas que melhor combinam com a intenção, estrutura causal e tom da conversa atual (evitando repetir sempre a mesma fórmula).
2. **Recombinação DOP (Substituição de Sub-árvores):** Combine sub-árvores diferentes para expressar planos complexos multi-hop (ex: encaixando uma sub-árvore de causa dentro de uma sub-árvore de recomendação ou contraste), gerando de **8 a 16 variações sintáticas candidatas**.
3. **Polimento Morfológico (Marco 7):** Passe cada candidata pelo `MorphologicalRealizer` para garantir conjugação perfeita, concordância de gênero/número, contrações (*do, na, pelo*) e substituição de repetições por pronomes (anáfora).

### 3. Rankeador de Fluência por Compressão (`CTW` + `Tsetlin` + `Hyperdimensional Reservoir`)
Para escolher qual das 16 candidatas geradas soa mais natural e humana (sem rede Transformer):
1. **Context Tree Weighting (CTW) / PPM de Ordem Variável (C++/HIP):** Treine um modelo de compressão estatística em cima dos livros e conversas ingeridos. Ele calcula em microssegundos na CPU a *entropia/naturalidade* de cada frase candidata (penalizando construções esquisitas ou combinações pouco idiomáticas).
2. **Filtro de Adequação (Tsetlin Machine de Estilo):** Uma Tsetlin Machine leve avalia se a candidata preserva todos os conceitos obrigatórios do `<PLAN>` e respeita o tom do usuário (ex: não responder com gíria em contexto técnico ou formal demais num papo casual).
3. **Opcional (Sem Backprop):** Se desejar suavização sequencial contínua entre fragmentos, use apenas um **Hyperdimensional / Echo-State Reservoir** fixo em C++/HIP com leitura linear simples (Ridge/Hebbian, zero backpropagation) para pontuar fluidez de transição entre cláusulas.

### 4. Atualização dos Comandos e Marcos
- Substitua `export-realizer-dataset` e `train_micro_decoder.py` por:
  - `./cognicore train-realizer --model <dir> [--use-gpu]` (indexa árvores DOP, treina árvore CTW e Tsetlin de estilo diretamente em C++/HIP).
- A flag `--realizer` passa a alternar entre:
  - `strict` (Marco 7 — direto, conciso e determinístico)
  - `fluent` (Marco 10 — Recombinação DOP + Beam Search + Rankeamento CTW/Tsetlin).
- **Teste do Marco 10:** Dado o mesmo plano semântico multi-hop de 3 passos, o modo `fluent` deve gerar múltiplas candidatas inéditas, usar o CTW + Tsetlin para pontuá-las e selecionar uma resposta visivelmente mais natural e variada do que o modo `strict`, rodando 100% em C++ CPU-only sem nenhuma dependência de Python/PyTorch.

---

# Correção Arquitetural Definitiva (Fase 2 — Pilar 2 e Marco 10): Remoção do Micro-Decodificador Neural/Transformer

Ajuste o plano da Fase 2 com a seguinte regra estrita: **não queremos NENHUM Transformer, NENHUM modelo neural PyTorch e NENHUM Micro-Decodificador neural** no projeto — nem mesmo na "boca" (`ISurfaceRealizer`).

## 1. Itens da Fase 2 que devem ser REMOVIDOS / CANCELADOS do plano:
- **REMOVA** todo o `Micro-Decodificador Graph-to-Text` (~30M–50M params, Tiny GQA Transformer / Tiny RWKV).
- **REMOVA** `NeuralMouthRealizer`, `mouth_q4.bin`, `export-realizer-dataset`, `scripts/train_micro_decoder.py` e qualquer dependência de `PyTorch` / Python para treino neural.
- **REMOVA** a opção `--realizer neural` da CLI.

Todo o sistema (treino, indexação, raciocínio e geração fluente de texto) deve rodar **100% em C++20 + HIP/ROCm puro**, sem ponto flutuante pesado, sem backpropagation tradicional e sem Transformers.

---

## 2. O que entra no lugar do Micro-Decodificador (Novo Pilar 2 / Novo Marco 10):
Substitua o `NeuralMouthRealizer` pelo **`AnalogicalFluencyRealizer : public ISurfaceRealizer`** (modo `--realizer fluent`), construído inteiramente em C++/HIP sobre o motor morfológico do Marco 7:

1. **Extração de Árvores de Fraseado Expressivo (DOP — *Data-Oriented Parsing*):**
   - Durante `ingest-books` e `train-dialogue`, além das triplas do grafo, extraia sub-árvores sintáticas naturais completas (com ritmo, conectivos e marcadores de tom) deslexicalizadas em slots tipados.
   - Indexe cada sub-árvore com um **Vetor Hiperdimensional (VSA)** de Estilo/Contexto (`INTENT ^ TONE ^ SHAPE`) e armazene na **Kanerva SDM** + arquivo binário `mmap`.

2. **Síntese por Recombinação Analógica + Beam Search:**
   - Dado um `<PLAN>` inédito vindo do Grafo/Tsetlin, o `AnalogicalFluencyRealizer` recupera múltiplas sub-árvores compatíveis via VSA/SDM, **recombina fragmentos diferentes (DOP)** para expressar todos os hops do plano e gera de **8 a 16 frases candidatas inéditas**.
   - Todas as candidatas passam pelo **Motor Morfológico e de Unificação (Marco 7)** para garantir conjugação, concordância, contrações e pronomes (anáfora) corretos.

3. **Rankeador de Fluência Estatístico (CTW + Tsetlin de Estilo, em C++/HIP):**
   - **CTW (*Context Tree Weighting*) / PPM de ordem variável:** Treinado diretamente sobre os livros e diálogos ingeridos; mede em microssegundos na CPU a naturalidade/perplexidade de cada candidata gerada.
   - **Tsetlin Machine de Estilo:** Garante que a candidata escolhida preserva 100% dos conceitos do `<PLAN>` e bate com o tom da conversa (técnico, didático, alerta, casual).
   - *(Opcional em C++/HIP, sem Transformer e sem backprop)*: Um **Hyperdimensional / Echo-State Reservoir** fixo com leitura linear simples (Hebbian/Ridge) apenas para suavizar a pontuação de transição entre cláusulas.

---

## 3. Atualização da CLI e do Teste do Marco 10
- Substitua `export-realizer-dataset` e `train_micro_decoder.py` por:
  - `./cognicore train-realizer --model <dir> [--use-gpu]` (indexa árvores DOP, constrói o modelo CTW e treina a Tsetlin de estilo em C++/HIP).
- Atualize a flag da CLI para `--realizer [strict|fluent]`:
  - `strict` = `MorphologicalRealizer` (Marco 7, direto e determinístico).
  - `fluent` = `AnalogicalFluencyRealizer` (Marco 10, Recombinação DOP + Beam Search + CTW/Tsetlin).
- **Validação do Marco 10:** Comprove em teste automatizado (`ctest`) que, para um mesmo `<PLAN>` multi-hop, o modo `fluent` gera múltiplas variações inéditas, pontua via CTW + Tsetlin e escolhe uma resposta fluida em CPU-only — **com zero arquivos `.py` de PyTorch, zero Transformers e zero pesos neurais**.

Confirme essa substituição completa do Micro-Decodificador pelo `AnalogicalFluencyRealizer` e siga o plano atualizado.

---

O gargalo atual não é o VSA/SDM/Tsetlin. É a pilha linguística, o planejamento de documento, o conhecimento e os motores de domínio.
O que existe hoje
- Gerador de frases inéditas via SemanticPlan → DOP + morfologia + PPM/Tsetlin.
- Grafo causal multi-hop, memória episódica e contexto de turno.
- Síntese de código compondo APIs reais do workspace.
- Aritmética simples e problemas de palavra com duas quantidades.
- Raciocínio auditável, tudo em CPU, sem pesos neurais.
O que falta, por capacidade
Capacidade desejada	Hoje	Falta
Gerar textos longos	Frases e parágrafos curtos	Planejador de documento, seções, parágrafos, progressão tópica, revisão
Escolher palavras	Usa os rótulos do plano/grafo	Léxico com sinônimos, registro, valência, colocação, sentido
Planejar	Caminhos no grafo + ordem retórica curta	Planejamento hierárquico (objetivos → seções → frases), restrições globais
Coerência longa	Anáfora local e contexto de turno	Estado de discurso por documento, centering, cadeias lexicais, entidades
Escrever código	Compõe funções existentes	Síntese de controle de fluxo, tipos, algoritmos, testes, reparo
Matemática	2+2, 7*6, problemas simples	CAS simbólico, equações, frações, unidades, álgebra, verificação
Conhecimento	Grafo pequeno e curado	Ingestão massiva, ontologia, resolução de entidades, consistência
Competir com LLM	Domínios estreitos	Cobertura, escala, avaliação longa, generalização
Os gargalos técnicos concretos
1. Falta um planejador de documento
O DiscoursePlanner atual escolhe uma tese e uma alternativa. Para um texto grande, falta:
- DocumentPlanner: objetivo → outline → seções → parágrafos.
- Rhetorical Structure Theory explícita: tese, evidência, causa, contraste, concessão, conclusão.
- Seleção de conteúdo: o que entra, o que fica de fora, o que precisa de exemplo.
- Controle de progressão: não repetir o mesmo sujeito, não voltar de assunto, manter promessas do outline.
Sem isso, o realizador só consegue “verbalizar o próximo plano”; ele não sabe escrever um artigo.
2. Falta escolha lexical real
Hoje o plano já traz src/rel/dst como palavras. O realizador conjuga e concorda, mas não escolhe entre:
- “gerar”, “produzir”, “provocar”, “causar”, “levar a”;
- “servidor”, “máquina”, “host”;
- registro técnico, didático, casual, alerta;
- colocação (“gerar calor” vs. “produzir calor”);
- ambiguidade e sentido.
Falta um léxico construído do corpus com lema, POS, valência, sinônimos distribucionais (VSA), colocação e registro, e um LexicalChooser que escolhe lexemas antes da morfologia.
3. A gramática DOP ainda é rasa
Ela cobre cláusula + conectivos. Para texto longo e natural faltam:
- árvores lexicalizadas (Synchronous Tree Substitution Grammar);
- subordinação, relativas, coordenação complexa, elipse;
- pontuação e prosódia;
- agregação de sentenças;
- geração de expressões referenciais (“esse processo”, “o mesmo servidor”);
- probabilidades reais de árvore (PCFG/DOP probabilístico), não só PPM de bytes.
4. O rankeador estatístico é fraco para coerência longa
PPM/CTW de bytes mede naturalidade local. Para texto longo é preciso:
- n-gramas interpolados com Kneser-Ney;
- modelo de classes/skip-gram;
- cache de tópico;
- ranking por coerência de discurso;
- verificação de consistência com o grafo;
- talvez um log-linear reranker treinado com EM/perceptron, sem backprop.
5. Falta memória de trabalho de documento
ContextMemory guarda turnos e entidades. Para escrever 5 páginas falta:
- rascunho estruturado;
- entidades já apresentadas;
- pronomes já usados;
- fatos já citados;
- pendências do outline;
- checagem de contradição;
- revisão/reescrita.
6. Matemática precisa de um motor simbólico
O MathSolver atual resolve aritmética inteira e problemas de duas quantidades. Para GSM8K/MathVista de verdade faltam:
- parser de equações e variáveis;
- frações, decimais, porcentagem, unidades;
- álgebra e sistemas;
- geometria;
- planejamento de múltiplos passos;
- verificação do resultado;
- integração com fatos do grafo.
7. Código precisa de síntese, não só composição
Hoje ele liga funções existentes por dataflow e gera patches a partir de chamadas internas. Para HumanEval/SWE-bench de verdade faltam:
- biblioteca de padrões AST com controle de fluxo (loop, condicional, try/catch);
- inferência de tipos mais séria;
- síntese enumerativa/CEGIS guiada por testes;
- localização e reparo de bug;
- geração de testes;
- resolução de overloads/templates;
- verificação semântica além do Tree-sitter.
8. Conhecimento e escala
O v3 tem 19 nós e 12 arestas. Um LLM tem bilhões de parâmetros e um corpus enorme. Para competir em conhecimento faltam:
- ingestão massiva de livros/docs/código;
- extração de fatos com alta precisão;
- ontologia e regras;
- resolução de entidades;
- atualização incremental;
- avaliação de factualidade.
9. Avaliação
O benchmark atual é pequeno e escrito por nós. Para medir “competir com LLM” faltam:
- benchmark longo e cego;
- avaliação humana real;
- métricas de coerência, factualidade, repetição, cobertura;
- avaliação por domínio;
- comparação com baselines.
O que não é o gargalo
- VSA 4096 bits.
- SDM.
- Tsetlin.
- Grafo causal.
- Falta de GPU.
- Falta de float.
Esses componentes já são rápidos e auditáveis. O problema é que a “boca” e o planejamento” estão em estágio de protótipo.
Roadmap mínimo para chegar lá
Sugiro esta ordem:
1. Marco 12 — Gerador de documento
DocumentPlanner + outline + seções + parágrafos + estado de discurso + revisão.
2. Marco 13 — Léxico e escolha lexical
Léxico do corpus, sinônimos via VSA, colocação, registro e LexicalChooser.
3. Marco 14 — Gramática lexicalizada
STSG/DOP lexicalizado, relativas, subordinação, agregação e pontuação.
4. Marco 15 — Rankeador de coerência
Kneser-Ney + cache + centering + verificação no grafo.
5. Marco 16 — Motor matemático simbólico
Equações, unidades, frações, multi-step e verificação.
6. Marco 17 — Síntese de código com testes
Controle de fluxo, tipos, CEGIS e reparo.
7. Marco 18 — Ingestão massiva e avaliação longa
Corpus grande, ontologia, benchmark cego e avaliação humana.
Resposta honesta sobre “competir com LLM”
- Em domínios estreitos e verificáveis — explicação causal sobre um grafo, documentação técnica, geração de código sobre APIs conhecidas, matemática de problema de palavra — é possível competir ou até ganhar em auditabilidade e custo.
- Em conversa aberta, criatividade e conhecimento geral, sem uma camada neural massiva e sem escala de dados, não é realista no curto prazo. A arquitetura pode chegar lá em qualidade de texto se ganhar planejamento hierárquico, léxico e gramática; mas isso é um projeto de vários marcos, não um ajuste.
Se quiser, o próximo passo mais rentável é o Marco 12 (DocumentPlanner + estado de discurso + revisão). É ele que transforma “gerador de frases” em “gerador de textos”.

---

Diagnóstico
O problema atual não é “falta de um template melhor”. É que a geração ainda trata cada fato como uma frase isolada:
grafo → 1 fato → 1 sentence plan → 1 cláusula → concatenação
Isso produz:
- repetição do sujeito;
- ausência de estado discursivo;
- coordenação só no nível de string;
- pronomes por comparação de strings;
- ordinais fixos;
- nenhuma noção de tópico, foco, dado/novo, ambiguidade referencial;
- nenhuma busca entre formas alternativas de dizer a mesma coisa.
O salto de qualidade exige separar:
o que dizer  →  como organizar  →  como se referir  →  como realizar
E representar cada uma dessas etapas explicitamente.
Princípio arquitetural
Gerar texto como um plano discursivo e escolher a melhor realização por busca, não por concatenação gulosa.
O pipeline recomendado:
Grafo tipado + Entity IDs
        ↓
ContentPlan (objetivo, AnswerGraph, movimentos retóricos)
        ↓
TextPlan (parágrafos, sentenças, proposições, relações retóricas)
        ↓
DiscourseState + ReferencePlan (centering, pronomes, elipses)
        ↓
AggregationPlan (coordenação, relativas, listas, compartilhamento de sujeito)
        ↓
ConstructionPlan (DOP/construções, ordem, voz, conectores)
        ↓
SurfacePlan (morfologia, concordância, contrações, pontuação)
        ↓
Busca/ranqueamento (PPM + coerência + custo discursivo)
        ↓
Texto final
A chave é que cada etapa tem uma representação própria e o texto final é escolhido por uma função de custo, não por um if de template.
1. Camada semântica: entidades e proposições
Hoje o grafo usa strings. Para coesão real, precisamos de:
using EntityId = uint32_t;

struct Proposition {
    EntityId subject;
    RelationId relation;
    EntityId object;
    RelationType type;
    GramGender gender;
    GramNumber number;
    bool negated;
    GramTense tense;
    std::string modality;
    Provenance provenance;
};
Adicionar ao grafo:
- Entity com id, canonical_name, aliases, type, gender, number, salience.
- Mention com entity_id, surface, sentence_position, grammatical_role.
- Sequence como estrutura de ordem, não como fatos soltos.
- Coordination como estrutura: lista de entidades + determinante/preposição compartilhada.
Benefícios:
- pronome deixa de ser comparação de string;
- ambiguidade referencial pode ser detectada;
- concordância de coordenação é calculada;
- o mesmo fato pode aparecer com formas diferentes sem duplicar nós.
Arquivos sugeridos:
include/core/semantic_ir.hpp
include/core/entity_graph.hpp
src/entity_graph.cpp
2. ContentPlan → TextPlan
O ContentPlanner já escolhe objetivo e movimentos. Falta um TextPlan entre conteúdo e superfície:
struct SentencePlan {
    std::vector<Proposition> propositions;
    RhetoricalRelation relation;
    EntityId topic;
    EntityId focus;
    bool given_topic;
    bool contrast;
};

struct ParagraphPlan {
    std::string heading;
    std::vector<SentencePlan> sentences;
    std::vector<EntityId> discourse_stack;
};

struct TextPlan {
    CommunicativeGoal goal;
    ResponseShape shape;
    std::vector<ParagraphPlan> paragraphs;
    DiscourseState state;
};
O TextPlanner decide:
- quais proposições entram em cada sentença;
- qual é o tópico e o foco;
- o que é dado vs. novo;
- onde usar coordenação, relativa, subordinação ou elipse;
- a ordem das sentenças.
Isso evita a geração “um fato = uma frase”.
Arquivos:
include/core/text_plan.hpp
include/core/text_planner.hpp
src/text_planner.cpp
3. DiscourseState e centering
Adicionar um estado discursivo por parágrafo:
struct DiscourseState {
    std::vector<EntityId> forward_centers; // Cf: sujeito > objeto > oblíquo
    EntityId backward_center = INVALID;    // Cb
    std::unordered_map<EntityId, int> salience;
    std::unordered_map<EntityId, Mention> last_mention;
    std::set<std::pair<GramGender, GramNumber>> ambiguous_features;
};
Algoritmo de centering:
1. ordenar os centros por papel gramatical: sujeito > objeto > outros;
2. Cb = entidade mais saliente do Cf anterior que aparece na sentença atual;
3. usar pronome se:
- a entidade é o Cb atual;
- não há outra entidade compatível em gênero/número;
- a distância discursiva não é grande;
4. usar NP definido completo se:
- a entidade foi reintroduzida;
- há ambiguidade;
- mudou o foco;
5. usar elipse/zero anafórico em coordenação de mesmo sujeito quando a construção permitir.
Isso resolve:
O motor de combustão interna contém o bloco.
Ele contém o cabeçote.
e evita:
O motor contém o bloco. O motor contém o cabeçote. O motor contém ...
Arquivos:
include/core/discourse_state.hpp
include/core/reference_generator.hpp
src/reference_generator.cpp
4. Agregação e coordenação como estruturas
Não coordenar strings; coordenar estruturas.
struct CoordinationPlan {
    std::vector<EntityId> conjuncts;
    std::string determiner;    // o/a/os/as
    std::string preposition;   // de/por/em...
    GramGender gender;         // resolvido
    GramNumber number;         // plural
};
Regras:
- mesmo sujeito + mesma relação + múltiplos objetos:
- agrupar em uma única sentença;
- contém X, Y e Z;
- concordância plural;
- preposição repetida ou compartilhada conforme a construção.
- mesmo sujeito + relações diferentes:
- usar pronome ou elipse;
- escolher entre coordenação de VP, relativa ou subordinação.
- mesma relação + sujeitos diferentes:
- usar coordenação de sujeitos quando semanticamente coerente;
- caso contrário, sentenças separadas com conectivo.
Agreement de coordenação:
- todos femininos → feminino plural;
- mistura → masculino plural;
- singular + plural → plural;
- preposições contraídas por conjunto (pelas bielas e pelos êmbolos), não só no primeiro.
Arquivos:
include/core/aggregation_plan.hpp
src/aggregator.cpp
5. Planejamento de sentença e construções (DOP)
Substituir a escolha fixa de padrão por um inventário de construções com restrições e custos:
struct Construction {
    std::string id;
    std::string pattern;       // [SUBJ] [VERB] [OBJ], [SUBJ] [VERB] [OBJ1] e [OBJ2]
    std::vector<SlotConstraint> slots;
    RelationType relation;
    RhetoricalRelation rhetorical;
    int formality;
    double cost;
};
O ConstructionPlanner escolhe entre:
- SVO simples;
- coordenação de objetos;
- coordenação de VPs;
- relativa (que ...);
- subordinação causal (porque ...);
- passiva;
- elipse de sujeito;
- construção de sequência (primeiro ...; em seguida ...);
- construção de definição (X é um Y que ...).
As construções vêm do corpus/modelo:
constructions_pt.jsonl
Aprender DOP a partir de sentenças alinhadas a fatos/proposições. O AnalogicalFluencyRealizer já tem fragmentos DOP; ele pode ser integrado como fonte de construções, não como concatenação.
Arquivos:
include/core/construction_planner.hpp
src/construction_planner.cpp
data/constructions_pt.jsonl
6. Escolha lexical e colocação
O Lexicon já tem colocações e sinônimos. Usar isso no planejamento:
- escolher verbo por relação + objeto (gere calor vs. produza calor);
- escolher conectivo por relação retórica;
- evitar repetição de verbo/conectivo no parágrafo;
- respeitar registro/tom (técnico, didático, casual).
Isso não deve ser hardcoded; deve vir de:
language_rules.json
lexicon.bin
collocations
7. Geração como busca
O ponto mais importante: não escolher a primeira realização; enumerar e ranquear.
Estados de busca:
proposições ainda não realizadas
+ DiscourseState
+ TextPlan parcial
Ações:
- agregar proposições;
- escolher construção;
- escolher referência (pronome/NP/elipse);
- escolher conector;
- escolher ordem.
Custo:
custo_total =
    PPM/coerência
  + penalidade de repetição
  + penalidade de ambiguidade referencial
  + penalidade de violação de centering
  + penalidade de comprimento
  + penalidade de conectivo repetido
  + custo da construção
  - cobertura de fatos
Busca em beam search com largura pequena (ex.: 4–16), podando por custo e por cobertura.
Isso permite escolher entre:
O motor contém o bloco, o cabeçote e o conjunto móvel.
O motor é formado pelo bloco, pelo cabeçote e pelo conjunto móvel.
O motor inclui bloco, cabeçote e conjunto móvel.
e escolher a melhor segundo o modelo, o tom e o contexto.
Arquivos:
include/core/generation_search.hpp
src/generation_search.cpp
8. Dados do modelo (nada fixo no C++)
Toda a informação linguística deve ficar no diretório do modelo:
parser_pt.json            # morfologia, verbos, preposições, gênero, stopwords
language_rules.json       # templates, conectivos, ordinais, seções
discourse_pt.json         # preferências de centering, RST, pronomes
constructions_pt.jsonl    # inventário DOP/construções e custos
morphology_pt.json        # concordância, contrações, ordinais, irregulars
entity_types.json         # tipos de entidade e restrições
lexicon.bin               # sinônimos, colocações, frequência
O código só consome. O usuário pode editar a prosódia, os conectivos, as construções e a morfologia sem recompilar.
9. Aprendizado simbólico
Sem Transformer, mas com aprendizado real:
1. Mineração de construções do corpus:
- extrair padrões sintáticos recorrentes do TextParser;
- armazenar slots, relação e custo;
- usar frequency, PMI e colocação do Lexicon.
2. Ranqueador de coesão:
- features: centering, distância, ambiguidade, repetição, conector, comprimento;
- treinar um Tsetlin pequeno ou uma regressão simbólica com rótulos heurísticos + humanos;
- persistir em discourse_ranker.bin.
3. Preferências de referência:
- aprender quando o corpus usa pronome vs. NP definido vs. elipse, condicionado a gênero/número/distância/papel.
4. Avaliação humana:
- planilhas reais de fluência/coerência; nenhuma nota fabricada.
10. Avaliação
Além de PPM/perplexidade:
- coesão referencial: precisão de pronomes, taxa de ambiguidade, entity grid;
- coesão lexical: cadeias lexicais, repetição de conteúdo;
- agregação: sentenças por fato, tamanho médio, variação de comprimento;
- ordenação: acurácia de sequência, uso de ordinais;
- cobertura: proposições realizadas / proposições disponíveis;
- naturalidade: avaliação humana cega (Likert + preferência);
- aderência ao formato: BRIEF/EXPLANATORY/DOCUMENT, número de parágrafos;
- ablação: sem pronomes, sem agregação, sem busca, sem ordinais.
11. Fases de implementação
Fase 0 — IR e discurso
- EntityId, Proposition, Mention, TextPlan, DiscourseState.
- Testes de identidade, aliases, gênero/número, sequência.
Fase 1 — Reference Generation
- Centering, pronomes, NP definido/indefinido, elipse, detecção de ambiguidade.
- Testes com “motor”, “biela”, “árvore de manivelas”, “válvulas”.
Fase 2 — Agregação e coordenação
- CoordinationPlan, agreement de coordenação, contrações por conjunto.
- Testes: “é formado pelo bloco, pelo cabeçote e pelo conjunto móvel”; “pelas bielas e pelos êmbolos”.
Fase 3 — Construções e sequência
- ConstructionPlanner, DOP, relativas, subordinação, passiva, ordinais.
- Testes de mecanismo: definição → partes → funcionamento → sequência.
Fase 4 — Coesão de parágrafo
- RST, conectivos por relação, tópico/foco, cadeias lexicais, transições.
- Testes de documento multi-parágrafo.
Fase 5 — Busca e ranqueamento
- Beam search com PPM + custo discursivo.
- Testes de escolha entre construções alternativas.
Fase 6 — Aprendizado e avaliação
- Mineração de construções, ranker de coesão, avaliação humana, benchmark.
- Documentação e model cards.
12. Riscos e decisões
- Explosão combinatória: beam pequeno + poda por custo/cobertura.
- Pronome ambíguo: fallback conservador para NP completo.
- Concordância de coordenação mista: regra explícita, testada.
- Construções ruidosas do corpus: confiança mínima e validação sintática.
- Não prometer fluência de LLM: o objetivo é prosa simbólica coesa, auditável e melhor que templates.
- Não colocar texto no C++: tudo em arquivos do modelo.
Recomendação final
O caminho mais inteligente não é “melhorar o template de parágrafo”. É implementar:
1. Entity IDs + proposições tipadas;
2. TextPlan + DiscourseState;
3. Reference Generation com centering;
4. Aggregation/Coordination como estruturas;
5. ConstructionPlanner com DOP;
6. Busca com custo discursivo + PPM;
7. Tudo parametrizado por arquivos do modelo.
Isso transforma a geração de “juntar fatos” em planejamento de discurso com escolha de realizações, que é o que produz prosa coesa de verdade — e continua 100% simbólico, auditável e treinável sem Transformer.
