# CogniCore — motor conversacional pós-LLM

A arquitetura de referência é a do commit `559df3f`: VSA binária de 4096 bits,
Kanerva SDM, Tsetlin Machine, grafo causal e gramática de construções com pontuação
n-grama. A implementação atual preserva essa base e as extensões dos marcos 7–11.
Não há Transformer, PyTorch, libtorch, pesos neurais ou backpropagation no projeto.

## Build

CPU:

```bash
cmake -B build -DUSE_ROCM=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

AMD HIP/ROCm:

```bash
cmake -B build_rocm -DUSE_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200
cmake --build build_rocm -j$(nproc)
ctest --test-dir build_rocm --output-on-failure
```

HIP acelera indexação e treino simbólico; a inferência conversacional usa CPU.
Tree-sitter e suas gramáticas são obtidos pelo CMake FetchContent em versões
fixadas. Asserções ficam ativas nos testes mesmo em Release.

## CLI

```bash
./build_rocm/cognicore ingest-books --input livros/ --model modelo --use-gpu
./build_rocm/cognicore ingest-books --input analise.conllu --model modelo --parser conllu
./build_rocm/cognicore train-dialogue --input conversas.jsonl --model modelo --epochs 20 --use-gpu
./build_rocm/cognicore consolidate --model modelo
./build_rocm/cognicore stats --model modelo
./build_rocm/cognicore evaluate-fluency --model modelo --input amostra.txt
./build_rocm/cognicore ingest-corpus --input corpus/ --model modelo
./build_rocm/cognicore write --model modelo --topic calor --realizer fluent
./build_rocm/cognicore math --expr "2x + 3 = 11"
./build_rocm/cognicore chat --model modelo --realizer lexical --debug
```

O parser padrão de texto livre agora inclui análise de dependências interna por
regras (sem UDPipe): sujeito, objeto, oblíquos, relativas, `advcl`, negação,
modalidade, quantificadores e condicionais temporais/causais/concessivas. O
CoNLL-U externo continua aceito como entrada de maior precisão.

As pistas lexicais de roteamento, correção, geração de código, aritmética e
acompanhamento ficam em `language_rules.json`, gravado dentro do diretório do
modelo. `data/portuguese_model_rules.json` é a semente copiada no treino; o
código de inferência apenas consome as listas. Trocar idioma/estilo significa
trocar/editar esse arquivo, não editar `assistant.cpp` ou `model_bundle.cpp`.

`evaluate-fluency` reporta entropia PPM-C e perplexidade de uma amostra. A
perplexidade é uma calibração interna para relatório, não uma métrica de LLM.
Para avaliação humana, `scripts/evaluate_fluency_human.py` gera uma planilha
cega e pontua as notas devolvidas; nenhuma nota é fabricada.

A entrada de treino de diálogo é JSONL com campos `text` e `intent`. As intenções
são `EXPLAIN_CAUSE`, `ANSWER_FACT`, `COMPARE`, `ASK_CLARIFY`, `AGREE`, `HYPOTHESIZE`
e `GREET`. O treino atualiza autômatos inteiros e contagens de construções/n-gramas;
não há treino de uma rede neural.

```bash
./build_rocm/cognicore ingest-code --repo /caminho/projeto --model modelo --use-gpu
./build_rocm/cognicore code --model modelo --workspace /caminho/projeto \
  --lang py --task 'combine load_items e compute_total'
./build_rocm/cognicore chat --model modelo --workspace /caminho/projeto \
  --lang py --realizer fluent --debug
./build_rocm/cognicore train-realizer --model modelo --use-gpu
```

`assistant` é um alias de `chat`. `--lang` seleciona a linguagem do código.
`--realizer fluent` é o padrão: recombinação de fragmentos DOP, morfologia,
PPM de ordem variável e Tsetlin de estilo. `--realizer strict` seleciona apenas
`MorphologicalRealizer`. Execute `train-realizer` após ingerir o corpus; modelos
antigos sem treinamento de fluência usam strict com aviso. Outras opções são rejeitadas.

Sem workspace, a sessão é salva no modelo; com workspace, em
`<workspace>/.cognicore`. Essa pasta é excluída da ingestão. Os últimos 128 episódios,
os fatos aprendidos e o índice de código são recarregados. `conversation.jsonl`
registra o histórico em disco. `train-realizer` constrói o índice SDM/VSA e a política
de estilo em C++/HIP; `fluency.bin` mantém produções e contagens PPM consultáveis por mmap.

## Marcos preservados e readequados

| Marco | Implementação |
|---|---|
| 1–6 | VSA, SDM, Tsetlin, equivalência HIP/CPU, persistência e teste integrado original. |
| 7 | CoNLL-U com lema, UPOS, dependências, gênero/número, tempo, modalidade e negação; fallback de regras; morfologia PT/EN, anáfora e conectivos. |
| 8 | Tree-sitter C++/Python/JavaScript/TypeScript; assinaturas, documentação, formas AST canônicas, VSA na SDM e grafo de chamadas/tipos/parâmetros/membros/herança/imports. |
| 9 | Seleção Tsetlin e composição estrutural de chamadas compatíveis; impressão e validação sintática com Tree-sitter. Testes compilam/executam C++ e executam Python gerados. |
| 10, readequado | `AnalogicalFluencyRealizer`: produções DOP restritas, beam search, verificação semântica/morfológica, PPM e Tsetlin de estilo; `train-realizer`, `strict` e `fluent`; arquivo de fluência consultado por mmap. |
| 11 | Roteador Tsetlin de cinco classes, conversa, aprendizado imediato, raciocínio causal, assistência de código e sessões persistentes. |
| 12 | `DocumentPlanner`: outline, seções, parágrafos, revisão e comando `write`. |
| 13 | `Lexicon`: escolha lexical, sinônimos, colocação, registro e `lexicon.bin`. |
| 14 | `LexicalizedGrammarRealizer`: relativas, subordinação, agregação e pontuação (`--realizer lexical`). |
| 15 | `CoherenceModel`: Kneser-Ney/cache, centering e ancoragem no grafo (`coherence.bin`). |
| 16 | `SymbolicMath`: frações, potências, porcentagens, equações lineares e verificação (`math`). |
| 17 | `synthesize_tested`: candidatos com controle de fluxo, testes Python e reparo CEGIS. |
| 18 | `ingest-corpus` incremental com manifesto/checkpoint, benchmark de texto longo e planilhas humanas. |

O realizador recebe somente um `SemanticPlan` através de `ISurfaceRealizer`.
As construções preenchem sujeitos, predicados e objetos do plano. N-gramas escolhem
entre realizações equivalentes e conectivos compatíveis; não escolhem fatos nem
substituem a relação por outro verbo. Contagens aprendidas são serializadas e
recarregadas, sem busca de uma resposta pronta no corpus. A gramática DOP agora
distingue cláusulas genéricas, taxonômicas e de propriedade, e inclui produções de
contraste aprendidas de marcadores observados (`mas`, `porém`, `however`).

O teste do Marco 10 comprova composição com conceito ausente no texto observado,
influência das contagens na escolha dos conectivos, preservação de negação,
rejeição de frame com predicado incompatível e persistência após recarga.

O parser de texto livre agora extrai SVO geral por léxico verbal, coordenação de
cláusulas e complementos, além de condicionais `se ... então` / `if ... then`.
A memória contextual (`ContextMemory`) rastreia entidades do discurso e resolve
pronomes/dêixis por gênero, número, saliência e recência; continuidade de tópico
só é herdada quando o turno não introduz conceito próprio. O planejador
(`DiscoursePlanner`) escolhe a tese que termina em uma semente da pergunta, poda
prefixos cobertos e anexa no máximo uma rota alternativa fundamentada no grafo.

## Compatibilidade e limites

Os arquivos anteriores do núcleo continuam utilizáveis. O grafo migra o layout
original sem metadados; índice de código, episódios e pares são arquivos opcionais
independentes. O histórico Git não foi reescrito na readequação.

A análise de dependências é interna e baseada em regras; não é UDPipe nem um
parser neural. Quantificadores, orações relativas e `advcl` simples viram
metadados/regras executáveis, mas quantificação aninhada, escopo arbitrário e
discurso complexo continuam limitados. O CoNLL-U externo segue disponível.

A síntese AST liga parâmetros por dataflow, aceita múltiplos parâmetros, retorno
`void`, funções sem argumentos, overloads, métodos estáticos/de instância e
patches compostos pelas chamadas internas de um alvo. Ela não cobre qualquer
patch textual arbitrário, resolução de templates ou prova de tipos do compilador.
O grafo é consultado por `graph_mmap.bin` em modo somente leitura; após escritas,
o índice mapeado é regenerado.

O roteador e a fluência ainda precisam de avaliação em corpus amplo e de
avaliadores humanos reais. A meta de RAM depende do tamanho do corpus e do
índice; não foi comprovada para repositórios arbitrariamente grandes.

## Modelo de conversa em português

Um modelo simbólico local de teste e os scripts de preparação/treino estão
descritos em [docs/MODEL_CONVERSA_PT.md](docs/MODEL_CONVERSA_PT.md).
Inicie com `./scripts/chat_portuguese.sh --debug`. As novas intenções de diálogo
são `THANK`, `FAREWELL`, `INTRODUCE`, `HELP` e `SMALLTALK`; os identificadores
originais permanecem com os mesmos valores. `--style-only` aprende construções
e estatísticas sem importar ficção ou respostas de datasets para o grafo factual.

A terceira geração treinada (`models/conversa-pt-fluent-v3`) usa o mesmo corpus
público e adiciona o parser contextual, grafo mmap, AST ampliada e ferramentas de
calibração/avaliação. Ficha: [modelo v3](docs/MODEL_CONVERSA_PT_FLUENT_V3.md).

## Benchmarks próprios

`benchmarks/run_benchmarks.py` executa categorias inspiradas em GPQA, HLE, GSM8K,
HumanEval e SWE-bench, além de checks de fluência strict/fluent. Os conjuntos são
pequenos, escritos para este projeto e não são os benchmarks oficiais:

```bash
python3 benchmarks/run_benchmarks.py \
  --cli build/cognicore --model models/conversa-pt-fluent-v3 \
  --out benchmarks/results
```

O relatório inclui respostas, latências, calibração PPM, categorias de geração
longa e pico de RSS. As planilhas de avaliação humana (fluência e texto longo)
são geradas vazias; notas só entram quando avaliadores reais devolvem o arquivo
preenchido.

Detalhes da execução simbólica, memória episódica, regras condicionais e limites atuais: [Raciocínio auditável](docs/SYMBOLIC_REASONING.md).

Implementação e limites do novo Marco 10: [fluência analógica](docs/ANALOGICAL_FLUENCY.md).
Marcos 12–18 (documento, léxico, gramática lexicalizada, coerência, matemática,
código testado e ingestão massiva): [detalhes](docs/MILESTONES_12_18.md).
