# Resumo da implementação — CogniCore / conversa-pt-fluent-v3.1

Documento de consolidação do que foi feito no projeto até aqui: corpus,
treino, modelo v3.1, parser orientado a dados, planejamento de discurso,
prosa natural, mineração de construções e busca.

Data: 2026-11-18.

> Atualização: o artefato mais recente é o **conversa-pt-fluent-v3.2**
> (128.000 tokens de codebook, ~110 MB de estilo, 1.306 artigos da Wikipédia,
> 245 livros do Gutenberg e 366 nós/290 arestas no grafo). Ficha:
> [MODEL_CONVERSA_PT_FLUENT_V32.md](MODEL_CONVERSA_PT_FLUENT_V32.md).

---

## 1. Visão geral

O CogniCore continua sendo um motor **100% simbólico**: VSA binária, Kanerva
SDM, Tsetlin Machine, grafo causal, DOP/PPM e morfologia. Não há Transformer,
peso neural, backpropagation ou PyTorch no treino ou na inferência.

Nesta etapa o foco foi:

1. ampliar o corpus e o vocabulário do modelo conversacional;
2. fazer o Tsetlin treinar com estado e dataset residentes na VRAM;
3. desligar o aprendizado online por padrão;
4. permitir fatos curados sem deslocar o codebook;
5. tirar o texto linguístico do C++ e colocá-lo em arquivos do modelo;
6. melhorar o parser para termos multi-palavra;
7. introduzir planejamento content-first e discurso;
8. gerar prosa com coordenação, pronomes, ordinais e busca;
9. minerar construções do corpus e treinar um rankeador Tsetlin de coesão.

---

## 2. Corpus v3.1

Dataset de treino:

```text
/run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31
```

Conteúdo:

| Fonte | Quantidade |
|---|---:|
| Wikipédia em português | 695 artigos |
| Project Gutenberg | 99 livros novos + Dom Casmurro da v3 |
| PDF educacional | 1 (SENAI, motores de combustão interna) |
| Corpus conversacional da v3 | preservado |

Tamanho:

- ~66,2 MB de texto;
- ~10,4 milhões de tokens observados;
- 801 arquivos de estilo;
- 48.000 tipos de vocabulário no codebook (48.008 com tokens especiais).

O corpus foi montado com:

```text
scripts/fetch_public_pt_corpus.py
scripts/prepare_portuguese_v31.py
```

O `prepare` cria uma amostra balanceada
`00_vocabulary_seed.txt` para que o codebook limitado veja todos os registros
antes de atingir o teto.

### Licenças e atribuição

- Wikipédia: CC BY-SA 4.0; título, página, revisão, URL e hash no manifesto.
- Gutenberg: domínio público nos EUA; metadados e SHA-256 no manifesto.
- PDF SENAI: material público; direitos de redistribuição não afirmados.
- Corpus conversacional: licenças originais no `corpus_manifest.json`.

---

## 3. Treino Tsetlin residente na VRAM

Antes, o backend HIP fazia `hipMalloc`/`hipMemcpy` por amostra. Agora
`TsetlinMachine::fit(..., use_gpu=true)` usa
`backend::tsetlin_train_device`:

- estado TA, sinais das cláusulas e dataset `X` são copiados para a VRAM uma vez;
- cada avaliação/atualização de cláusula roda em kernels HIP sobre o estado na VRAM;
- o host só recebe uma cópia no início e no fim;
- não há `class_sum` voltando para o host por amostra.

Equivalência CPU × VRAM validada em `test_hip_equiv`, inclusive com 12 classes.

GPU usada: **Radeon RX 9060 XT / gfx1200**, via `HIP_VISIBLE_DEVICES=0`.

Logs do treino:

```text
[HIP] Tsetlin residente na VRAM: TA=2.00 MB, X=4.37 MB, amostras=8400, epocas=6
[HIP] Tsetlin residente na VRAM: TA=0.00 MB, X=0.10 MB, amostras=12286, epocas=40
```

Tempos da v3.1:

| Estágio | Tempo |
|---|---:|
| `style` (CPU, 801 arquivos) | 390,5 s |
| `facts` (HIP/SDM) | 17,6 s |
| `policy` (Tsetlin na VRAM) | 18,0 s |
| `fluency` (Tsetlin na VRAM) | 27,3 s |
| `consolidate` | 13,1 s |
| `heldout` + `probes` + `stats` | ~19 s |

---

## 4. Modelo v3.1

Artefato:

```text
models/conversa-pt-fluent-v3.1
→ /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.1
```

Métricas atuais:

| Item | Valor |
|---|---:|
| Codebook VSA | 48.008 tokens |
| Grafo causal | 60 nós / 43 arestas |
| SDM | 453/2048 slots |
| Tsetlin | 240 cláusulas (esparso) |
| Heldout de intenção | 740/1.060 (69,81%) |
| Probes sintéticos | 18/18 (100%) |
| Disco | ~544 MB |
| RAM estimada pelo `stats` | ~1,33 GB |
| Calibração PPM (`heldout_pt.txt`) | 2,2656 bits/byte; perplexidade 4,8084 |

A v3 original foi preservada e restaurada limpa:

```text
conversa-pt-fluent-v3
19 nós / 12 arestas
heldout 769/1.060 (72,55%)
probes 18/18
```

A diferença de acurácia entre v3 e v3.1 não é bug do treino GPU: o codebook
mudou de 16k para 48k tipos, portanto o vetor de entrada do Tsetlin também
mudou. O conjunto de diálogo e o seed são os mesmos.

### Benchmark próprio

`benchmarks/results_v31/benchmark_report.json`:

| Categoria | Resultado |
|---|---:|
| science (GPQA-style) | 5/5 |
| hard reasoning (HLE-style) | 3/3 |
| math (GSM8K-style) | 5/5 |
| code (HumanEval-style) | 2/2 |
| patch (SWE-bench-style) | 1/1 |
| fluency | 3/3 |
| long-form | 3/3 |

O conjunto é pequeno e project-owned; não é comparável aos benchmarks oficiais.

---

## 5. Aprendizado online e fatos

O chat agora inicia com:

```text
Aprendizado online: desligado
```

`--online-learning` é opt-in. Sem ele, uma pergunta/instrução do usuário não é
gravada no grafo.

Modos de ingestão:

- `--style-only`: vocabulário/estilo, sem fatos.
- `--fact-n-style-only`: fatos + estilo.
- `--facts-only`: apenas grafo + SDM, sem alterar codebook/léxico/coerência.

O `--facts-only` é o caminho recomendado para fatos curados, porque preserva o
classificador de diálogo.

Fatos de motor:

```text
data/conversation_pt/engine_facts.txt
```

Ingestão:

```bash
./build/cognicore ingest-books \
  --input data/conversation_pt/engine_facts.txt \
  --model models/conversa-pt-fluent-v3.1 \
  --facts-only
./build/cognicore consolidate --model models/conversa-pt-fluent-v3.1
```

Resultado:

- 31 triplas adicionadas;
- grafo de 19/12 para 60/43;
- codebook e Tsetlin preservados;
- heldout 740/1.060 e probes 18/18.

Tentar ingerir a apostila inteira como fatos gerou **2.320 triplas / 2.804
nós** em 10 páginas — ruído demais. Por isso o corpus grande continua em
`--style-only`.

---

## 6. Parser orientado a dados

O `text_parser.cpp` deixou de carregar listas de texto. Tudo que é linguístico
foi para:

```text
parser_pt.json
```

no diretório do modelo. Ele contém artigos, preposições, pronomes, relativos,
subordinadores, conjunções, advérbios, verbos/lemas, irregulares, negações,
modais, cópulas, adjetivos, quantificadores, condições, stopwords, listas de
gênero, exceções de plural, marcadores e auxiliares.

O código só consome o arquivo. O usuário pode editar o JSON do modelo sem
recompilar.

Melhorias no parser:

- NPs multi-palavra: `motor de combustão interna`, `vela de ignição`,
  `árvore de manivelas`;
- gênero/número pelo núcleo do NP;
- preposições de fronteira para não engolir o complemento verbal;
- merge regra × dependência prefere o NP mais limpo;
- ontologia ampliada: `HAS_PART`, `PART_OF`, `FUNCTION`, `ENABLES`,
  `SEQUENCE`, `PRECEDES`.

Aliases:

```text
terms.tsv
```

`motor` ↔ `motor de combustão interna`; o `TermResolver` resolve o termo
canônico para o grafo/planejador.

---

## 7. Planejamento content-first

`ContentPlanner`:

1. resolve o tópico;
2. monta o `AnswerGraph`;
3. identifica objetivo: `DEFINITION`, `MECHANISM`, `CAUSE`, `COMPARISON`,
   `PROCEDURE`, `ESSAY`;
4. mede fatos, partes, funções, sequência e relações distintas;
5. só então escolhe `BRIEF`, `EXPLANATORY` ou `DOCUMENT`.

O formato não vem de uma palavra-chave isolada. “Resuma” encurta; “como
funciona” com partes/funções vira documento.

O chat também aceita override:

```bash
--shape brief|explain|document
```

---

## 8. Discurso e prosa natural

### Pipeline

```text
ContentPlan
  → TextPlan (EntityIds, proposições, tipos de sentença)
  → DiscourseState + ReferenceGenerator (Cf/Cb, pronomes, ambiguidade)
  → Aggregator (CoordinationPlan, agreement)
  → ConstructionBank (constructions_pt.jsonl)
  → DiscourseGenerator
  → GenerationSearch (beam + custo + rankeador Tsetlin)
```

### Entidades e proposições

- `EntityId`, `Entity`, `Mention`, `Proposition`;
- `EntityRegistry` com identidade canônica;
- `Entity` carrega `type` (`MECHANISM`, `PROCESS`, `STAGE`, ...).

### Centering e referência

`DiscourseState` mantém:

- `forward_centers`;
- `backward_center`;
- saliência;
- última menção;
- features ambíguas.

`ReferenceGenerator` usa pronome apenas quando:

- a entidade é o centro atual/recente;
- não há outra entidade com mesmo gênero/número;
- a distância discursiva é pequena.

Caso contrário, usa NP definido completo.

### Agregação e coordenação

`Aggregator` gera `CoordinationPlan` para partes/todo:

```text
O motor de combustão interna contém o conjunto móvel, o cabeçote e o bloco.
```

Agreement:

- todos femininos → feminino plural;
- mistura → masculino plural;
- singular + plural → plural;
- preposição distribuída por conjuncto:

```text
pelas bielas e pelos êmbolos
```

### Coordenação de VPs

Proposições com o mesmo sujeito são coordenadas em um único período:

```text
O motor de combustão interna transforma a queima da mistura,
produz o trabalho e usa o ciclo otto.
```

Em vez de:

```text
O motor ... transforma ... . Ele produz ... . Ele usa ... .
```

### Ordinais

Sequências usam ordinais do modelo:

```text
Primeiro, a admissão precede a compressão; em seguida, ...; por fim, ...
```

---

## 9. Banco de construções, mineração e busca

### Construções

```text
data/constructions_pt.jsonl
```

Inclui construções curadas e mineradas:

- `é formado`;
- `é composto`;
- `faz parte`;
- `contém`;
- `é um`;
- `segue`;
- `permite`;
- `funciona como`;
- `serve para`.

### Mineração DOP

Comando:

```bash
./build/cognicore mine-constructions \
  --input /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31/style \
  --model models/conversa-pt-fluent-v3.1 \
  --out constructions_pt.jsonl \
  --limit-files 30
```

O minerador:

- usa o `TextParser` para extrair triplas;
- filtra stopwords e relações relevantes;
- usa as colocações do `Lexicon` (`collocation_counts()`) para PMI;
- grava `frequency`, `cost` e `pmi`.

### Busca

`GenerationSearch`:

- gera variantes por sentença;
- mantém um beam de `beam_width` candidatos;
- pontua repetição, pronome, comprimento, conector, construção e coesão;
- poda pelo menor custo.

`discourse_pt.json` controla:

```json
{
  "max_pronoun_distance": 2,
  "allow_zero_anaphora": true,
  "repetition_penalty": 0.35,
  "pronoun_bonus": 0.15,
  "length_penalty": 0.01,
  "connector_penalty": 0.05,
  "beam_width": 4
}
```

### Rankeador Tsetlin

`CohesionRanker`:

- TsetlinMachine de 8 features;
- features: pronome, repetição, ambiguidade, comprimento, conector repetido,
  coordenação, sequência, parte/todo;
- treinado com 256 vetores sintéticos;
- integrado ao custo da busca.

O treino é sintético/estrutural. O próximo refinamento é usar rótulos reais de
corpus/avaliação humana.

---

## 10. Arquivos do modelo

No diretório do modelo v3.1:

```text
parser_pt.json            # léxico/sintaxe/morfologia do parser
language_rules.json       # templates, conectivos, ordinais, seções
discourse_pt.json         # centering, pesos, beam_width
constructions_pt.jsonl    # banco de construções/DOP
morphology_pt.json        # contrações, ordinais, agreement
entity_types.json         # tipos de entidade
terms.tsv                 # aliases canônicos
lexicon.bin               # sinônimos, colocações, frequência
codebook.bin              # VSA
graph.bin / graph_mmap.bin
sdm.bin
tsetlin.bin
coherence.bin
realizer.bin
fluency.bin
```

Artefatos documentais:

```text
MODEL_CARD_FLUENT_V31.md
BENCHMARK_REPORT.md
benchmark_report.json
corpus_manifest.json
training_report.json
```

---

## 11. Comandos principais

Chat:

```bash
./scripts/chat_portuguese.sh \
  --model models/conversa-pt-fluent-v3.1 \
  --realizer fluent
```

Com aprendizado online:

```bash
./scripts/chat_portuguese.sh \
  --model models/conversa-pt-fluent-v3.1 \
  --realizer fluent --online-learning
```

Override de formato:

```bash
./scripts/chat_portuguese.sh \
  --model models/conversa-pt-fluent-v3.1 \
  --realizer fluent --shape document
```

Fatos curados:

```bash
./build/cognicore ingest-books \
  --input data/conversation_pt/engine_facts.txt \
  --model models/conversa-pt-fluent-v3.1 --facts-only
./build/cognicore consolidate --model models/conversa-pt-fluent-v3.1
```

Mineração de construções:

```bash
./build/cognicore mine-constructions \
  --input style/ --model models/conversa-pt-fluent-v3.1 \
  --out constructions_pt.jsonl --limit-files 30
```

Treino completo:

```bash
HIP_VISIBLE_DEVICES=0 python3 scripts/train_portuguese_model.py \
  --data /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31 \
  --model /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.1 \
  --cli build_rocm/cognicore --epochs 6 \
  --max-vocab 48000 --context-stride 2 --use-gpu
```

---

## 12. Testes e verificação

- `ctest`: **22/22** testes passando.
- `test_hip_equiv`: Hamming, SDM e Tsetlin GPU × CPU idênticos.
- `test_marco19_content`: ContentPlanner e prosa natural.
- `test_marco20_discourse`: Entity IDs, ambiguidade/pronome, agregação,
  coordenação composta, prosa fim-a-fim e custo de repetição.
- `test_cli_phase2`: fluxo de CLI/ingestão/chat.

---

## 13. Commits desta etapa

```text
3e1675f feat: coordinate same-subject verb phrases in discourse realization
0d82667 feat: morphology/type resources, DOP mining, beam search and cohesion ranker
5ce9a9c feat: discourse IR, centering, coordination, construction bank and search
b0055bd feat: natural prose realization with coordination and anaphora
8a3de2f feat: content-first discourse planning and model-owned parser data
fffe179 feat: make online learning opt-in and add fact-only ingestion
9a6804e feat: add v3.1 corpus pipeline and VRAM-resident Tsetlin training
```

Remote: `origin/main` sincronizado até `3e1675f`.

---

## 14. Limites atuais

- A geração é simbólica; não é geração neural palavra por palavra.
- O rankeador Tsetlin usa treino sintético, não anotação humana.
- RST/cadeias lexicais completas ainda são parciais.
- A mineração de construções é estatística e ainda ruidosa em alguns padrões.
- O corpus inteiro não é convertido em fatos; fatos de domínio são curados.
- Concordâncias de compostos menos frequentes ainda podem falhar.
- A calibração PPM é interna, não perplexidade de LLM.

---

## 15. Próximos passos naturais

1. Treinar o rankeador Tsetlin com avaliação humana/corpus.
2. Expandir o banco DOP com construções por tipo de discurso.
3. Implementar RST completo (elaboração, contraste, causa, sequência).
4. Adicionar cadeias lexicais/sinônimos ao custo de coesão.
5. Melhorar agreement de coordenações complexas e elipse.
6. Revisar o benchmark de prosa longa com sentenças únicas, cobertura e
   aderência ao formato.
7. Avaliar o chat com raters reais; nenhuma nota deve ser fabricada.

---

## 16. Onde ler mais

- [Modelo v3.1](MODEL_CONVERSA_PT_FLUENT_V31.md)
- [Modelo v3](MODEL_CONVERSA_PT_FLUENT_V3.md)
- [Marcos 12–18](MILESTONES_12_18.md)
- [Treinar do zero](TRAIN_FROM_SCRATCH.md)
- [Raciocínio auditável](SYMBOLIC_REASONING.md)
- [Fluência analógica](ANALOGICAL_FLUENCY.md)
