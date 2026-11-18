# Modelo português v3.2 — corpus ampliado, 128k de vocabulário e fatos de domínio

Artefato:

```text
models/conversa-pt-fluent-v3.2
→ /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.2
```

A v3.1 e a v3 foram preservadas. A v3.2 usa a mesma arquitetura simbólica
(VSA, Kanerva SDM, Tsetlin, grafo causal, DOP/PPM, morfologia e discurso), sem
Transformer, peso neural ou backpropagation.

## O que mudou

| Item | v3.1 | v3.2 |
|---|---:|---:|
| Texto de estilo | 66,2 MB | ~110,1 MB |
| Arquivos de estilo | 801 | 1.560 |
| Artigos da Wikipédia | 695 | 1.306 |
| Livros do Gutenberg | 99 novos | 245 (234 + 11 de domínio) |
| PDFs | 1 | 1 |
| Vocabulário VSA | 48.008 | **128.006** |
| Contexto | stride 2 | stride 2 |
| Grafo factual | 60/43 | **366 nós / 290 arestas** |
| Léxico | 353.317 | 447.341 entradas |
| Coerência KN | 5,97M trigramas | 10,19M trigramas |
| Disco | ~544 MB | ~553 MB |
| Treino | CPU/GPU | GPU gfx1200, 128k |

## Corpus

Dataset:

```text
/run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v32
```

Fontes:

- 1.306 artigos da Wikipédia em português;
- 245 livros do Project Gutenberg (incluindo geometria, ciências naturais,
  medicina, agricultura, botânica, engenharia e história natural);
- a apostila SENAI de motores;
- corpus conversacional e estilo herdados da v3.1;
- `00_vocabulary_seed.txt` balanceado.

O corpus completo continua em `--style-only`. Fatos são ingeridos
separadamente com `--facts-only`.

## Treino

```bash
HIP_VISIBLE_DEVICES=0 python3 scripts/train_portuguese_model.py \
  --data /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v32 \
  --model /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.2 \
  --cli build_rocm/cognicore --epochs 6 \
  --max-vocab 128000 --context-stride 2 --use-gpu
```

O estágio `style` levou cerca de 19 minutos nesta máquina e produziu um
codebook de 128.000 tipos. O estágio Tsetlin usou estado TA, sinais e dataset
residentes na VRAM (`HIP_VISIBLE_DEVICES=0`, gfx1200).

Tempos dos estágios finais:

| Estágio | Tempo |
|---|---:|
| `policy` | 21,0 s |
| `fluency` | 29,7 s |
| `consolidate` | 13,8 s |
| `heldout` + `probes` + `stats` | ~21,7 s |

## Fatos

Depois do treino, foram ingeridos com `--facts-only`:

```text
data/conversation_pt/engine_facts.txt   # motor
data/conversation_pt/world_facts.txt    # Brasil, França, Itália e outros países
data/conversation_pt/domain_facts.txt   # matemática, estatística, T.I., computação,
                                        # ciência, saúde e animais
```

Resultado:

- 31 triplas de motor;
- 122 triplas de países;
- 119 triplas de domínio;
- grafo final: **366 nós / 290 arestas**;
- codebook e Tsetlin preservados;
- SDM com 1.551/2.048 slots.

## Métricas

| Item | v3.2 |
|---|---:|
| Codebook VSA | 128.006 tokens |
| Heldout de intenção | 752/1.060 (70,94%) |
| Probes sintéticos | 16/18 (88,9%) |
| Léxico | 447.341 entradas / 1.292.954 colocações |
| Coerência KN | 455.818 unigramas / 4.560.125 bigramas / 10.194.014 trigramas |
| Fluência DOP/PPM | 2.058 produções / 80.000 contextos |
| Disco | ~553 MB |
| RAM estimada pelo `stats` | ~1,58 GB |

O heldout mudou em relação à v3.1 porque o codebook cresceu de 48k para 128k;
o vetor de entrada do Tsetlin mudou. O corpus de diálogo e o seed continuam os
mesmos. A v3.2 não foi otimizada para maximizar esse heldout; ela foi treinada
para ampliar vocabulário, contexto, léxico e conhecimento.

## Código

`code_patterns.jsonl` agora tem 27 construções. Exemplos que funcionam sem
workspace:

```text
crie um script python que calcula o fatorial de um numero
crie um script python que verifica se um numero é primo
crie um script python que inverte uma string
crie um script python que calcula a média de uma lista
crie um script python que lê um arquivo
```

A síntese continua simbólica: escolhe um padrão do modelo, preenche slots e
valida a sintaxe. Não é geração neural de código.

## Exemplos de conversa

```text
fale sobre o Brasil
→ O Brasil é o maior país da América do Sul, é uma república federativa
  e é o maior produtor de café do mundo.
  O Brasil produz o petróleo e o café e a soja.

qual é a capital da França?
→ Paris é uma capital da França.

fale sobre a Itália
→ A Itália é um país da Europa e é uma república.
  A Itália produz a macarrão e o azeite e a pizza e fica a Europa.
```

As respostas usam o grafo factual. A gramática de superfície ainda tem
limitações: preposições de valência (`precisa de água`, `fica na Europa`),
gênero de alguns compostos (`macarrão`) e coordenações de objetos repetem
conectivos. Esses são alvos da próxima rodada de parser/realizador.

## Dados do modelo

```text
parser_pt.json
language_rules.json
discourse_pt.json
constructions_pt.jsonl
morphology_pt.json
entity_types.json
terms.tsv
lexicon.bin
codebook.bin
graph.bin / graph_mmap.bin
sdm.bin
tsetlin.bin
coherence.bin
realizer.bin
fluency.bin
code_patterns.jsonl
```

### Limitação conhecida do `realizer.bin`

O estágio de estilo da v3.2 gerou um `realizer.bin` de ~369 MB. O carregador
recusa archives acima de 300 MB para não estourar a RAM; a v3.2 carrega com um
realizador de construções reduzido (11 frames / 34.550 n-gramas, reconstruído
no diálogo). A prosa natural usa `DiscourseGenerator`/`NaturalProseRealizer` e
não depende desses mapas. O léxico, a coerência, o PPM e o codebook completos
foram preservados. Uma próxima versão deve limitar os mapas do realizer durante
a indexação para manter também o realizador lexicalizado completo.

## Comandos

```bash
./scripts/chat_portuguese.sh \
  --model models/conversa-pt-fluent-v3.2 \
  --realizer fluent
```

```bash
./build/cognicore ingest-books \
  --input data/conversation_pt/world_facts.txt \
  --model models/conversa-pt-fluent-v3.2 --facts-only
./build/cognicore consolidate --model models/conversa-pt-fluent-v3.2
```

```bash
./build/cognicore mine-constructions \
  --input /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v32/style \
  --model models/conversa-pt-fluent-v3.2 \
  --out constructions_pt.jsonl --limit-files 30
```

## Limites

- 128k é o teto do codebook; o corpus tem centenas de milhares de tipos únicos.
- O corpus grande é estilo, não fato; a extração automática de fatos é ruidosa.
- Os fatos de país/domínio são curados, não uma base enciclopédica completa.
- O parser ainda erra valência de preposições e alguns gêneros.
- O heldout/probes são pequenos e não substituem avaliação humana.
- O rankeador Tsetlin de coesão usa treino sintético.
- Não é um LLM: a prosa é simbólica, auditável e limitada ao grafo/corpus.
