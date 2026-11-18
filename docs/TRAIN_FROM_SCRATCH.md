# Treinando um modelo CogniCore do zero

Este guia cobre o ciclo completo: build, coleta e montagem de datasets, indexação
("pré-treinamento" simbólico), treino de diálogo, treino do realizador, alinhamento,
avaliação, fine-tuning incremental, inferência e publicação do artefato.

CogniCore **não usa Transformer, backpropagation, PyTorch ou pesos neurais**. O
"modelo" é um conjunto de estruturas simbólicas:

- codebook VSA de 4096 bits;
- Kanerva SDM;
- Tsetlin Machines inteiras;
- grafo causal;
- gramática DOP/construction grammar;
- léxico, n-gramas/PPM, rankeador de coerência;
- regras de linguagem e construções de código específicas do modelo.

## Mapa dos termos

| Termo comum em LLM | Equivalente no CogniCore |
|---|---|
| Pré-treinamento | Ingestão/indexação de corpus em VSA + SDM + grafo + DOP + léxico |
| SFT | `train-dialogue` (intenções Tsetlin) + `train-realizer` (estilo/fluência) |
| Alignment | Regras do modelo, correções one-shot, tons, recusas explícitas e avaliação humana |
| Fine-tuning | Nova ingestão + `train-dialogue`/`train-realizer` sobre o modelo existente |
| Decoding | Plano semântico → DOP/gramática lexicalizada → morfologia → rankeamento |
| Pesos | Cláusulas Tsetlin, produções DOP, léxico, grafo, SDM e contagens |

## 1. Requisitos

- Linux com C++20, CMake ≥ 3.20, compilador GCC/Clang e OpenMP.
- Python 3 apenas para utilitários de preparação/avaliação.
- Opcional: AMD ROCm/HIP para acelerar indexação e treino simbólico.
- Inferência: 100% CPU.
- Disco: depende do corpus e do codebook. Para protótipos médios, alguns GB são
  confortáveis; o modelo v3 de exemplo ocupa ~67 MB.
- RAM: depende do corpus, do léxico, dos n-gramas e do PPM. O v3 treinado tem
  ~146 MB estimados de estruturas, com pico de processo maior durante o benchmark.

Não instale `torch`, `transformers` ou `tensorflow`. O projeto não os utiliza.

## 2. Build

CPU:

```bash
cd /home/wanderson/Projetos/tessel
cmake -B build -DUSE_ROCM=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

ROCm/HIP opcional:

```bash
cmake -B build_rocm -DUSE_ROCM=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_rocm -j"$(nproc)"
ctest --test-dir build_rocm --output-on-failure
```

A inferência usa sempre CPU. `--use-gpu` só acelera ingestão/treino onde o
backend HIP estiver disponível. No `train-dialogue` e no `train-realizer`, o
estado TA e o dataset empacotado ficam residentes na VRAM durante o `fit`.
Para escolher a GPU discreta (por exemplo, `gfx1200`), use
`HIP_VISIBLE_DEVICES=0` conforme a ordem de `rocminfo`/`hipGetDeviceProperties`.

## 3. Organização do corpus

Sugestão de árvore:

```text
meu_corpus/
├── style/                  # livros, artigos, diálogos em texto limpo
│   ├── livro_a.txt
│   └── conversas_estilo.txt
├── fatos/                  # fatos para o grafo causal
│   ├── fisica.txt
│   └── dominio.conllu
├── dialogue_train.jsonl    # SFT de intenções
├── dialogue_eval.jsonl     # held-out
├── dialogue_probes.jsonl   # probes manuais
└── manifest.json           # fontes, licenças, hashes, splits
```

### 3.1 Livros e textos

- UTF-8, texto limpo, um arquivo `.txt` por livro/documento.
- Evite HTML/boilerplate; preserve pontuação e parágrafos.
- O parser interno extrai SVO, relativas, `advcl`, negação, modalidade,
  quantificadores simples e condicionais.
- Para maior precisão de dependências, forneça `.conllu` e use `--parser conllu`.

Limpeza básica:

```bash
python3 scripts/prepare_dataset.py \
  --input bruto/livro.txt \
  --output meu_corpus/style/livro_a.txt
```

### 3.2 Diálogos para SFT

Formato JSONL, uma linha por exemplo:

```json
{"text":"Por que o atrito gera calor?","intent":"EXPLAIN_CAUSE"}
{"text":"O que é o gelo?","intent":"ANSWER_FACT"}
{"text":"Compare gelo e vapor.","intent":"COMPARE"}
```

Intenções aceitas:

```text
EXPLAIN_CAUSE, ANSWER_FACT, COMPARE, ASK_CLARIFY, AGREE, HYPOTHESIZE,
GREET, THANK, FAREWELL, INTRODUCE, HELP, SMALLTALK
```

Validação/limpeza:

```bash
python3 scripts/prepare_dataset.py \
  --input bruto/dialogos.jsonl \
  --output meu_corpus/dialogue_train.jsonl
```

Recomendações:

- Remova duplicatas exatas e quase duplicatas.
- Separe `train/eval/probes` por grupo/fonte, não por linha aleatória.
- Mantenha o held-out fora da ingestão de estilo e de fatos.
- Não use o mesmo diálogo em treino e avaliação.

### 3.3 Fatos para o grafo causal

Texto simples:

```text
O atrito gera calor.
O calor derrete gelo.
A chuva causa umidade.
```

Ou CoNLL-U:

```bash
./build/cognicore ingest-books \
  --input meu_corpus/fatos/dominio.conllu \
  --model modelos/meu-modelo \
  --parser conllu
```

### 3.4 Preparação automática de corpus público em português

O script `prepare_portuguese_conversation.py` baixa fontes, calcula hashes,
registra licenças e monta `style/`, `facts.conllu`, `dialogue_*.jsonl` e
`manifest.json`. Requer `duckdb` para ler Parquet.

```bash
python3 -m venv .venv-data
. .venv-data/bin/activate
pip install duckdb

python3 scripts/prepare_portuguese_conversation.py \
  --out /dados/public_pt_v1 \
  --max-per-intent 700
```

O manifesto registra origem, licença, split e hashes. **Leia e respeite as
licenças** antes de redistribuir o modelo ou o corpus. Alguns datasets públicos
têm restrições não comerciais.

### 3.5 Código para o workspace

```bash
./build/cognicore ingest-code \
  --repo /caminho/do/repositorio \
  --model modelos/meu-modelo \
  --use-gpu   # opcional
```

Isso indexa funções, métodos, assinaturas, imports, chamadas, tipos e padrões
AST. O workspace pode ser reindexado a qualquer momento.

## 4. Dados específicos do modelo

Estes arquivos são copiados/gravados no diretório do modelo:

- `language_rules.json`: pistas de linguagem, sociais, recursos, condicionais,
  conectivos, sinônimos, quantificadores e regras de documento.
- `code_patterns.jsonl`: construções de código com slots.
- `linguistic_rules.json`: marcadores sintáticos do parser (semente em `data/`).

Para customizar um idioma ou estilo, edite primeiro as sementes:

```text
data/portuguese_model_rules.json
data/portuguese_code_patterns.jsonl
data/linguistic_rules.json
```

Depois de criar o modelo, o arquivo `models/<nome>/language_rules.json` passa a
ser a fonte usada na inferência. Não coloque listas de palavras novas dentro de
`assistant.cpp` ou `model_bundle.cpp`.

## 5. Pré-treinamento simbólico (indexação)

"Pré-treinar" aqui significa construir os índices e as estruturas do modelo.

### 5.1 Estilo primeiro

```bash
./build/cognicore ingest-books \
  --input meu_corpus/style \
  --model modelos/meu-modelo \
  --style-only \
  --max-vocab 16000 \
  --context-stride 4
```

Isso popula codebook VSA, SDM, n-gramas do realizador, léxico, coerência,
produções DOP e PPM. `--style-only` evita importar ficção/diálogos como fatos.

### 5.2 Fatos depois

```bash
./build/cognicore ingest-books \
  --input meu_corpus/fatos \
  --model modelos/meu-modelo
```

O grafo passa a conter relações SVO/causais/taxonômicas. O mesmo comando pode
ser repetido para novos fatos; ele carrega o modelo existente.

### 5.3 Corpus grande e incremental

```bash
./build/cognicore ingest-corpus \
  --input /dados/corpus_grande \
  --model modelos/meu-modelo \
  --format auto \
  --checkpoint-every 10
```

- `ingestion_manifest.jsonl` registra arquivos já processados.
- Rodar novamente retoma do ponto anterior.
- `--format jsonl` lê o campo `text` (ou `content`) de cada linha.
- `--limit-files`, `--limit-records`, `--max-vocab` e `--context-stride` ajudam
  a controlar custo.

### 5.4 O que verificar

```bash
./build/cognicore stats --model modelos/meu-modelo
```

Confira vocabulário, ocupação da SDM, nós/arestas, frames, produções DOP,
contextos PPM, léxico e coerência. Se o grafo estiver vazio, seus fatos não
foram extraídos: teste o parser em uma amostra e ajuste o formato/regras.

## 6. SFT simbólico

### 6.1 Política de diálogo (Tsetlin)

```bash
./build/cognicore train-dialogue \
  --input meu_corpus/dialogue_train.jsonl \
  --model modelos/meu-modelo \
  --epochs 6 \
  --clauses 20 \
  --balance \
  --use-gpu   # opcional
```

- `--balance` replica exemplos de classes menores para reduzir viés.
- `--clauses` define cláusulas por classe; valores maiores aumentam capacidade e
  custo.
- As intenções são validadas contra a lista fixa. Um rótulo desconhecido falha.
- A Tsetlin aprende cláusulas booleanas inteiras; não há gradiente.

### 6.2 Realizador de superfície (DOP/PPM/estilo)

```bash
./build/cognicore train-realizer --model modelos/meu-modelo
```

Treina a política de estilo, reconstrói o índice VSA/SDM de produções e grava
`fluency.bin`. **Execute depois de qualquer nova ingestão de estilo.** Sem isso,
o modo `fluent` avisa e usa `strict`.

### 6.3 Verificação rápida

```bash
./build/cognicore write \
  --model modelos/meu-modelo \
  --topic calor \
  --sections 4 \
  --paragraphs 2 \
  --realizer fluent
```

```bash
printf 'O atrito gera calor.\nPor que o atrito causa gelo?\nsair\n' | \
  ./build/cognicore chat --model modelos/meu-modelo --realizer fluent --debug
```

## 7. Alignment / alinhamento

No CogniCore, alignment não é RLHF nem ajuste de pesos. É alinhamento de
comportamento por dados, regras e avaliação:

### 7.1 Regras de linguagem e comportamento

Edite `models/meu-modelo/language_rules.json`:

- tons e conectivos;
- sociais (`social_*`);
- recusas e tópicos desconhecidos;
- condicionais e recursos;
- sinônimos e colocações;
- construções de documento;
- palavras de código, matemática e correção.

Valide a mudança com o teste que prova que não há frase hardcoded:

```bash
./build/test_no_fallback
```

### 7.2 Correções one-shot no chat

Durante o chat, fatos declarativos atualizam o grafo/SDM imediatamente:

```text
Você> Meu servidor antigo tem 2 GB de RAM.
Você> Meu servidor antigo tem 4 GB de RAM.
```

Correções/instruções não devem ser aprendidas como fatos. O classificador usa
`instruction_prefixes`/`correction_markers` do modelo.

### 7.3 Preferências e avaliação humana

Gere uma planilha cega:

```bash
python3 scripts/evaluate_fluency_human.py generate \
  --candidates benchmarks/results/fluency_candidates.jsonl \
  --out benchmarks/results/fluency_rating_sheet.jsonl
```

Depois que avaliadores reais preencherem:

```bash
python3 scripts/evaluate_fluency_human.py score \
  --ratings benchmarks/results/fluency_rating_sheet.jsonl \
  --out benchmarks/results/human_fluency_report.json
```

O repositório não inventa notas humanas. Use os resultados para ajustar regras,
corpus, tons e construções — não para “treinar pesos”.

## 8. Consolidação ("sono")

```bash
./build/cognicore consolidate --model modelos/meu-modelo
```

Funde arestas duplicadas, poda ruído de baixa frequência/confiança, compacta a
SDM para inferência, exporta a Tsetlin esparsa e regrava o índice mmap.

Rode após uma rodada grande de ingestão/treino. Se for continuar treinando,
pode consolidar de novo ao final.

## 9. Avaliação

### 9.1 Intenções em held-out

```bash
./build/cognicore evaluate-dialogue \
  --input meu_corpus/dialogue_eval.jsonl \
  --model modelos/meu-modelo \
  --out modelos/meu-modelo/heldout_metrics.json
```

O JSON traz amostras, acertos, acurácia e matriz de confusão.

### 9.2 Fluência/calibração PPM

```bash
./build/cognicore evaluate-fluency \
  --model modelos/meu-modelo \
  --input meu_corpus/style/heldout.txt \
  --out modelos/meu-modelo/fluency_metrics.json
```

Reporta bits/byte e perplexidade PPM-C. É uma calibração interna, não uma
perplexidade de LLM e não substitui avaliação humana.

### 9.3 Sessão conversacional real

```bash
python3 scripts/evaluate_portuguese_conversation.py \
  --cli build/cognicore \
  --model modelos/meu-modelo \
  --out modelos/meu-modelo/conversation_evaluation.json
```

O script copia o modelo para um diretório temporário, roda uma sessão CPU com
`--debug`, verifica identidade, agradecimento, multi-hop, aprendizado one-shot,
memória episódica, despedida e tópico desconhecido, e mede RSS quando possível.

### 9.4 Benchmark próprio

```bash
python3 benchmarks/run_benchmarks.py \
  --cli build/cognicore \
  --model modelos/meu-modelo \
  --out benchmarks/results
```

Categorias: ciência estilo GPQA, raciocínio difícil estilo HLE, matemática,
geração de código, patch, fluência e texto longo. Os conjuntos são pequenos e
próprios; **não são os benchmarks oficiais** e não devem ser comparados a
leaderboards públicos.

### 9.5 Avaliação humana

Use as planilhas geradas pelo benchmark:

```text
benchmarks/results/fluency_rating_sheet.jsonl
benchmarks/results/long_form_rating_sheet.jsonl
```

## 10. Fine-tuning incremental

Não há re-treino de pesos. Fine-tuning é continuar a indexação/treino sobre o
modelo existente.

Adicionar novos fatos:

```bash
./build/cognicore ingest-books \
  --input novos_fatos/ \
  --model modelos/meu-modelo
```

Adicionar novos diálogos:

```bash
./build/cognicore train-dialogue \
  --input novos_dialogos.jsonl \
  --model modelos/meu-modelo \
  --epochs 6 \
  --clauses 20
```

Adicionar estilo:

```bash
./build/cognicore ingest-books \
  --input novos_textos/ \
  --model modelos/meu-modelo \
  --style-only
./build/cognicore train-realizer --model modelos/meu-modelo
```

Adicionar código:

```bash
./build/cognicore ingest-code --repo novo_repo/ --model modelos/meu-modelo
```

Depois:

```bash
./build/cognicore consolidate --model modelos/meu-modelo
./build/cognicore stats --model modelos/meu-modelo
```

Cuidados:

- O modelo não “esquece” por gradiente, mas regras conflitantes podem coexistir.
  Prefira corrigir/remover dados antigos e reindexar quando necessário.
- Para mudanças grandes de distribuição, é mais limpo criar uma nova versão do
  modelo do que acumular indefinidamente.
- Reavalie held-out, fluência e sessão depois de cada rodada.

## 11. Inferência

Chat interativo:

```bash
./build/cognicore chat \
  --model modelos/meu-modelo \
  --realizer fluent \
  --debug
```

Realizadores:

- `strict`: morfologia determinística;
- `lexical`: relativas, subordinação e agregação;
- `fluent`: DOP + beam search + PPM/Tsetlin + coerência.

Com workspace de código:

```bash
./build/cognicore chat \
  --model modelos/meu-modelo \
  --workspace /caminho/do/repo \
  --lang cpp \
  --realizer fluent \
  --debug
```

O chat unificado entende pedidos de conversa, texto longo, matemática, código,
consulta de código e aprendizado de fato. Exemplos:

```text
Você> Escreva um texto sobre calor
Você> Resolva 2x + 3 = 11
Você> Quanto é 1/2 + 1/3?
Você> Escreva um script python que soma uma lista de valores
Você> O atrito gera calor.
```

Comandos diretos também existem para automação:

```bash
./build/cognicore write  --model modelos/meu-modelo --topic calor
./build/cognicore math   --expr "2x + 3 = 11"
./build/cognicore code   --model modelos/meu-modelo --workspace repo --lang py --task "identity e count"
```

## 12. Publicação do modelo

O treino salva automaticamente no diretório do modelo. Arquivos esperados:

```text
modelos/meu-modelo/
├── codebook.bin
├── sdm.bin
├── tsetlin.bin
├── dialogue_context.bin
├── graph.bin
├── graph_mmap.bin
├── realizer.bin
├── realizer_pairs.jsonl
├── fluency.bin
├── episodic_facts.bin
├── context.bin
├── episodes.bin
├── lexicon.bin
├── coherence.bin
├── code_patterns.jsonl
├── language_rules.json
├── code_index.bin
├── heldout_metrics.json
├── probe_metrics.json
├── training_report.json
├── corpus_manifest.json
└── MODEL_CARD.md
```

Checklist de publicação:

1. Rode `consolidate`.
2. Rode `stats` e guarde a saída.
3. Rode `evaluate-dialogue`, `evaluate-fluency` e a sessão de avaliação.
4. Rode o benchmark próprio e guarde `benchmark_report.json`.
5. Gere/atualize o model card com:
   - dados e licenças;
   - hiperparâmetros (`epochs`, `clauses`, `max-vocab`, `context-stride`);
   - métricas;
   - limitações;
   - comandos de inferência;
   - data e versão.
6. Verifique integridade:

```bash
find modelos/meu-modelo -type f -print0 | sort -z | xargs -0 sha256sum \
  > modelos/meu-modelo/SHA256SUMS
```

7. Copie o artefato **seguindo symlinks**, se o diretório de modelos for um link:

```bash
cp -rL modelos/meu-modelo /destino/meu-modelo-v1
```

`cp -a` em um symlink copia o link, não o conteúdo. Use `cp -rL` ou `rsync -aL`.

8. Não versione binários grandes no Git. Publique em armazenamento de modelos e
   deixe um symlink/ponteiro no repositório.
9. Preserve versões anteriores. Não sobrescreva um modelo publicado sem mudar a
   versão.
10. Inclua `corpus_manifest.json` e licenças; o modelo herda restrições do corpus.

## 13. Pipeline completo de exemplo

```bash
set -euo pipefail
cd /home/wanderson/Projetos/tessel

cmake -B build -DUSE_ROCM=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

MODEL=modelos/meu-modelo-v1
CORPUS=meu_corpus

./build/cognicore ingest-books \
  --input "$CORPUS/style" --model "$MODEL" --style-only \
  --max-vocab 16000 --context-stride 4

./build/cognicore ingest-books \
  --input "$CORPUS/fatos" --model "$MODEL"

./build/cognicore ingest-corpus \
  --input "$CORPUS/extra" --model "$MODEL" --checkpoint-every 10

./build/cognicore train-dialogue \
  --input "$CORPUS/dialogue_train.jsonl" --model "$MODEL" \
  --epochs 6 --clauses 20 --balance

./build/cognicore train-realizer --model "$MODEL"
./build/cognicore consolidate --model "$MODEL"
./build/cognicore stats --model "$MODEL"

./build/cognicore evaluate-dialogue \
  --input "$CORPUS/dialogue_eval.jsonl" --model "$MODEL" \
  --out "$MODEL/heldout_metrics.json"

./build/cognicore evaluate-fluency \
  --model "$MODEL" --input "$CORPUS/style/heldout.txt" \
  --out "$MODEL/fluency_metrics.json"

python3 scripts/evaluate_portuguese_conversation.py \
  --cli build/cognicore --model "$MODEL" \
  --out "$MODEL/conversation_evaluation.json"

python3 benchmarks/run_benchmarks.py \
  --cli build/cognicore --model "$MODEL" --out benchmarks/results

./build/cognicore chat --model "$MODEL" --realizer fluent --debug
```

## 14. Problemas comuns

**`fluent` avisa que não foi treinado**
Rode `train-realizer` depois da última ingestão de estilo.

**Grafo vazio**
Os fatos não foram extraídos. Teste o parser em amostras e use texto/CoNLL-U
adequado. `--style-only` não importa fatos por definição.

**Intenção desconhecida**
O JSONL de diálogo tem um `intent` fora da lista aceita. Use exatamente os nomes
documentados.

**Modelo não salva**
Verifique permissões e espaço em disco. `save` usa arquivos temporários +
rename; erros de escrita retornam código diferente de zero.

**Muitos n-gramas/léxico e RAM alta**
Reduza `--max-vocab`, aumente `--context-stride`, limite o corpus e use
checkpoints. O PPM tem limites internos de contextos/entradas.

**Benchmark artificial demais**
O benchmark é próprio e pequeno. Para uma decisão de publicação, combine:
held-out de intenções, avaliação de sessão, fluência PPM, benchmark do domínio e
avaliação humana cega.

## 15. Limites

- O modelo é simbólico e auditável, mas não é um LLM de conhecimento geral.
- A fluência é construída por DOP/PPM/morfologia; não há decodificador neural.
- A matemática cobre aritmética racional/equações lineares, não um CAS completo.
- A síntese de código cobre APIs indexadas e construções do modelo; não prova
  qualquer programa arbitrário.
- Alinhamento por regras e avaliação humana não é RLHF; mudanças de
  comportamento exigem editar dados/regras e reindexar.
- A cobertura, factualidade e naturalidade dependem do corpus, do grafo e do
  repertório de construções.
