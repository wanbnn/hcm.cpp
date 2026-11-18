# Modelo português v3.1 — corpus público ampliado

Artefato: `models/conversa-pt-fluent-v3.1` (link para
`/run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.1`).
As versões `conversa-pt-v1`, `conversa-pt-fluent-v2` e `conversa-pt-fluent-v3`
foram preservadas.

A v3.1 preserva a arquitetura simbólica da v3 (VSA binária, Kanerva SDM,
Tsetlin, grafo causal, DOP/PPM e morfologia), amplia o corpus/vocabulário e
adiciona um caminho de treino Tsetlin residente na VRAM. Não há Transformer,
peso neural, PyTorch, libtorch ou backpropagation.

```
./scripts/chat_portuguese.sh --model models/conversa-pt-fluent-v3.1 \
  --realizer fluent --debug
```

## O que mudou em relação à v3

| Item | v3 | v3.1 |
|---|---:|---:|
| Texto de estilo | 6,0 MB | 66,2 MB |
| Tokens observados | ~1,3 milhão | ~10,4 milhões |
| Vocabulário VSA (`--max-vocab`) | 16.000 | 48.000 |
| Contexto (`--context-stride`) | 4 | 2 |
| Artigos da Wikipédia | 0 | 695 |
| Livros do Project Gutenberg | 1 (Dom Casmurro) | 100 |
| PDFs educacionais | 0 | 1 |
| Treino simbólico | CPU | HIP/ROCm com estado na VRAM |
| Inferência | CPU | CPU |

O corpus de diálogo (intenções, `dialogue_train.jsonl`, fatos curados e probes)
é o mesmo da v3. O que cresceu foi o corpus linguístico usado para léxico,
coerência, realizador e contextos PPM: Wikipédia em português, literatura
brasileira/portuguesa de domínio público e uma apostila técnica do SENAI.

`style/00_vocabulary_seed.txt` é uma amostra determinística e balanceada de
todos os grupos (conversa, Wikipédia, Gutenberg e PDF). Ela é ingerida primeiro
para que o codebook limitado a 48 mil tipos veja todos os registros antes de
atingir o teto.

## Fontes e licenças

- **Wikipédia em português** — 695 artigos, licença CC BY-SA 4.0. Título,
  `pageid`, revisão, data e URL ficam em
  `public_pt_v31/sources_manifest.json`; o texto bruto e os metadados por
  artigo ficam em `public_pt_v31/raw/wikipedia_pt/`.
- **Project Gutenberg** — 99 livros de domínio público nos Estados Unidos,
  obtidos do espelho `gutenberg.pglaf.org`. Metadados e `sha256` no manifesto.
  A marca Project Gutenberg não é reivindicada.
- **PDF educacional** — `Apostila_motores_de_combustao_interna.pdf` (SENAI,
  Mecânica de Automóveis). O material é publicamente acessível; os direitos de
  redistribuição não foram afirmados pelo projeto. Texto extraído com
  `pdftotext`.
- **Corpus conversacional da v3** — OpenAssistant, PortugueseChat, atendimento
  sintético, atos de fala e Dom Casmurro; licenças originais no
  `corpus_manifest.json`.

Antes de redistribuir o modelo ou o corpus, consulte os manifestos e as fichas
originais. A licença CC BY-SA exige atribuição e compartilhamento sob a mesma
licença para o texto derivado da Wikipédia.

## Como o corpus foi construído

1. `scripts/fetch_public_pt_corpus.py` baixa a Wikipédia em lotes (API MediaWiki,
   texto completo via revisões), os livros do Gutenberg e os PDFs. Tudo é
   gravado com hash e metadados; falhas são registradas, não inventadas.
2. `scripts/prepare_portuguese_v31.py` parte de `public_pt_v1`, adiciona o texto
   novo em `style/`, remove duplicatas por `sha256` e gera a semente de
   vocabulário e o manifesto consolidado.
3. `scripts/train_portuguese_model.py` treina com `--max-vocab 48000
   --context-stride 2 --use-gpu` usando o binário HIP/ROCm.

O modo padrão do corpus é `--style-only` (vocabulário/estilo). Para extrair
também triplas factuais do texto, use `--fact-n-style-only` no `ingest-books`
ou `--style-facts` no script de treino. O parser de regras é conservador, mas
texto enciclopédico/PDF ainda pode gerar triplas ruidosas; valide o grafo
antes de tratar tudo como conhecimento asserido.

Para acrescentar fatos a um modelo já treinado sem mexer no vocabulário, no
léxico, na coerência ou no classificador de diálogo, use `--facts-only`:

```bash
./build/cognicore ingest-books --input data/conversation_pt/engine_facts.txt \
  --model models/conversa-pt-fluent-v3.1 --facts-only
./build/cognicore consolidate --model models/conversa-pt-fluent-v3.1
```

Esse modo grava apenas grafo causal + SDM. É o caminho recomendado para fatos
curados de um domínio (por exemplo, motores), porque preserva as métricas de
intenção e evita que o texto factual desloque o codebook.

```bash
python3 scripts/fetch_public_pt_corpus.py \
  --out /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31 \
  --gutenberg-catalog pg_catalog.csv.gz --gutenberg-limit 120

python3 scripts/prepare_portuguese_v31.py \
  --base /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v1 \
  --out  /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31

python3 scripts/train_portuguese_model.py \
  --data /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v31 \
  --model /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3.1 \
  --cli build_rocm/cognicore --epochs 6 --max-vocab 48000 --context-stride 2 \
  --use-gpu
```

O palco de estilo/parsing é inerentemente sequencial e roda em CPU dentro do
CLI; os estágios Tsetlin/SDM e o treino do realizador usam HIP/ROCm quando
`--use-gpu` está presente. A inferência (`chat`, `write`, `math`, benchmarks)
usa o binário CPU `build/cognicore`.

O chat não aprende fatos por padrão. `--online-learning` é opt-in; sem ele, uma
pergunta/instrução nunca é gravada no grafo. Os livros, a Wikipédia e o PDF
foram ingeridos com `--style-only`, então alimentam vocabulário, léxico,
coerência, DOP/PPM e realizações — não o grafo factual. O grafo da v3.1
continua sendo o `facts.conllu` curado (19 nós/12 arestas), por isso ela
consegue explicar calor/atrito/gelo e não “motor de carro”. Para fatos sobre
motores seria necessário ingerir/curar triplas desse domínio, não apenas o
texto da apostila.

### Treino Tsetlin residente na VRAM

`TsetlinMachine::fit(..., use_gpu=true)` usa
`backend::tsetlin_train_device`. O estado TA, os sinais das cláusulas e o
dataset empacotado (`X`) são copiados para a VRAM uma única vez; o shuffling e
os seeds são gerados no host (baratos), mas cada avaliação/atualização de
cláusula roda em kernels HIP sobre o estado que permanece na VRAM. Não há
`hipMalloc`/`hipMemcpy` por amostra nem leitura de `class_sum` pelo host; a
cópia de volta acontece só ao final do `fit`.

Isso reduz o tráfego host↔device e mantém o custo de treino proporcional ao
trabalho dos kernels. O corpus, o codebook, o léxico, a coerência e o PPM
continuam em RAM: são estruturas sequenciais de indexação, não o estado do
classificador Tsetlin. Os dois caminhos (CPU e VRAM) são verificados como
numericamente idênticos em `test_hip_equiv`.

### Fatos de domínio: motor

A apostila do SENAI é texto técnico. Ingeri-la inteira como fatos gera ruído:
o parser extraiu **2.320 triplas / 2.804 nós** só das 10 páginas. Por isso o
corpus completo continua em `--style-only`.

Para o exemplo de motores, `data/conversation_pt/engine_facts.txt` contém 29
frases verificadas (combustão, quatro tempos, partes, válvulas, biela, árvore
de manivelas, ciclo Otto). A ingestão com `--facts-only` adicionou 31 triplas
e elevou o grafo de 19/12 para **60 nós / 43 arestas**, sem alterar codebook,
léxico, coerência ou o classificador de diálogo. O heldout ficou em 740/1.060
e os probes em 18/18.

O chat agora reconhece que “como funciona” pede um documento de mecanismo e
emite múltiplas seções:

```text
Você> Como funciona o motor de combustão interna?
CogniCore> # motor de combustão interna: uma explicação causal

## Partes
O motor de combustão interna contém o conjunto móvel. O motor de combustão interna contém o cabeçote. O motor de combustão interna contém o bloco.

## Funcionamento
O motor de combustão interna transforma a queima da mistura. O motor de combustão interna produz o trabalho. O motor de combustão interna usa o ciclo otto.
```

O formato não vem de uma palavra-chave isolada: o `ContentPlanner` monta o
`AnswerGraph`, mede fatos/partes/funções/sequência e só então escolhe
`BRIEF`, `EXPLANATORY` ou `DOCUMENT`. “Resuma” continua curto; “como funciona”
com partes e funções vira documento. `--shape brief|explain|document` serve
apenas como override de teste.

### Parser e discurso content-first

- `TextParser` agora extrai NPs multi-palavra (`motor de combustão interna`,
  `vela de ignição`, `árvore de manivelas`) e infere gênero/número pelo núcleo.
- As listas lexicais/sintáticas não ficam mais em `text_parser.cpp`: elas vêm
  de `parser_pt.json`, copiado para o diretório do modelo. O usuário pode
  editar artigos, preposições, pronomes, verbos, marcadores, listas de gênero
  e stopwords sem recompilar. O código só consome o arquivo.
- `TermResolver` guarda aliases (`motor` ↔ `motor de combustão interna`) em
  `terms.tsv`; o grafo, o planejador e o chat resolvem o termo canônico.
- A ontologia ganhou `HAS_PART`, `PART_OF`, `FUNCTION`, `ENABLES`,
  `SEQUENCE` e `PRECEDES`, além de `TAXONOMIC`, `PROPERTY`, `CAUSAL` e
  `ACTION`.
- `ContentPlanner` escolhe um objetivo comunicativo (DEFINIÇÃO, MECANISMO,
  CAUSA, COMPARAÇÃO, PROCEDIMENTO, ENSAIO) e monta movimentos retóricos
  (definição, partes, funcionamento, sequência, conclusão). O
  `DocumentPlanner` virou renderizador de `ContentPlan`; chat e `write` usam o
  mesmo motor.

### Prosa natural: planejamento de discurso

O pipeline agora é:

```text
ContentPlan
  → TextPlan (EntityIds, proposições, tipos de sentença)
  → DiscourseState + ReferenceGenerator (Cf/Cb, pronomes, ambiguidade)
  → Aggregator (CoordinationPlan, agreement)
  → ConstructionBank (constructions_pt.jsonl)
  → DiscourseGenerator
  → GenerationSearch (repetição + pronome + comprimento + construções)
```

- `TextPlan` mantém entidades/proposições com IDs, tópico, foco e
  dado/novo.
- `DiscourseState` mantém centros forward/backward, saliência, última menção
  e features ambíguas.
- `ReferenceGenerator` usa pronome só quando o centro é o mesmo e não há
  outro candidato com o mesmo gênero/número.
- `Aggregator` transforma fatos de parte/todo em uma `CoordinationPlan`
  plural, com preposição e concordância resolvidas.
- `constructions_pt.jsonl` contém as construções de superfície e custos
  (`é formado`, `é composto`, `faz parte`).
- `discourse_pt.json` contém distância máxima de pronome e pesos da busca.
- `GenerationSearch` não escolhe a primeira realização: gera variantes e
  seleciona pela função de custo.

Exemplo:

```text
O motor de combustão interna é formado pelo conjunto móvel, pelo cabeçote e pelo bloco.
O motor de combustão interna transforma a queima da mistura. Ele produz o trabalho. Ele usa o ciclo otto.
```

Ainda é composição simbólica, não geração neural palavra por palavra. O
ganho é estrutural: coordenação, referência e discurso são decididos antes da
superfície. Limitações restantes: agreement de coordenações compostas
(`pelas bielas e pelos êmbolos`) e construções menos frequentes.


## Métricas

Artefato final: 48.008 tokens no codebook VSA (teto de 48.000 + os 8 tokens
especiais), 60 nós e 43 arestas no grafo factual (19/12 do núcleo curado +
fatos de motor), 55 frames, 2.048 produções DOP, 80.000 contextos PPM e
8.900.814 n-gramas.

| Item | v3.1 |
|---|---:|
| Vocabulário VSA | 48.008 tokens |
| Léxico do corpus | 353.317 entradas, 1.005.066 colocações |
| Coerência KN | 359.940 unigramas, 2.930.398 bigramas, 5.968.686 trigramas |
| Disco | 556.176 KB (~543 MB) |
| RAM estimada pelo `stats` | 1.397.920 KB (~1,33 GB) |
| Intenção reservada (heldout) | 740/1.060 (69,81%) |
| Probes sintéticos | 18/18 (100%) |
| Calibração PPM (`heldout_pt.txt`) | 2,2656 bits/byte; perplexidade 4,8084 |

A acurácia de intenção não é diretamente comparável à v3: o codebook mudou de
16 mil para 48 mil tipos, portanto o vetor de entrada do classificador Tsetlin
também mudou. O conjunto de diálogo, o seed e a configuração são os mesmos; os
probes sintéticos continuam 18/18. A sonda PPM tem apenas 367 bytes e serve
como calibração interna, não como perplexidade de LLM.

Tempos desta máquina (treino com `--use-gpu` na gfx1200):

| Estágio | Tempo |
|---|---:|
| `style` (CPU, 801 arquivos) | 392,57 s |
| `facts` (HIP/SDM) | 17,30 s |
| `policy` (Tsetlin na VRAM) | 17,41 s |
| `fluency` (Tsetlin na VRAM) | 26,67 s |
| `consolidate` | 12,32 s |
| `heldout` + `probes` + `stats` | 17,63 s |

No `policy`, o log registra `TA=2,00 MB, X=4,37 MB, amostras=8400, épocas=6`
residentes na VRAM. No `fluency`, `TA≈0,00 MB, X=0,10 MB, amostras=12286,
épocas=40`. A GPU gfx1200 ficou em 100% de uso durante esses estágios; o pico
de RSS do processo ficou em ~2,2 GB, contra ~5,5 GB no caminho antigo que
copiava o estado a cada amostra.

## Benchmark próprio

`benchmarks/run_benchmarks.py` cria um modelo de diagnóstico a partir da v3.1,
ingere fatos científicos e executa categorias inspiradas em GPQA, HLE, GSM8K,
HumanEval, SWE-bench e geração longa:

| Categoria | Resultado |
|---|---:|
| science (GPQA-style) | 5/5 |
| hard reasoning (HLE-style) | 3/3 |
| math/word problems (GSM8K-style) | 5/5 |
| code generation (HumanEval-style) | 2/2 |
| patch (SWE-bench-style) | 1/1 |
| fluency checks (strict vs fluent) | 3/3 |
| long-form generation | 3/3 |

Calibração PPM do modelo de diagnóstico: 2,2645 bits/byte, perplexidade
4,8048. Pico de RSS medido pelo runner: ~3,6 GB (carrega o artefato de 543 MB
e o modelo de diagnóstico). O conjunto é pequeno e escrito para este projeto;
não é comparável aos números oficiais dos benchmarks citados. Respostas,
latências e detalhes estão em
`benchmarks/results_v31/benchmark_report.json`.

As planilhas de avaliação humana
(`fluency_rating_sheet.jsonl`, `long_form_rating_sheet.jsonl`) são geradas
vazias. Nenhuma nota humana foi inventada; use
`scripts/evaluate_fluency_human.py score` depois que avaliadores reais
preencherem as notas.

## Limites

- O corpus de diálogo e os rótulos de intenção continuam os da v3; a v3.1 não
  inventa diálogos novos. O ganho está em vocabulário, colocação, coerência,
  morfologia e contextos de superfície.
- Os 80.000 contextos PPM e 400.000 entradas são limites internos do motor e
  não foram alterados. O corpus maior preenche esses limites com textos mais
  variados; não aumenta a janela teórica do modelo.
- O grafo factual continua sendo o conjunto curado da v3. Texto da Wikipédia e
  dos livros entra como estilo (`--style-only`), não como fato asserido pelo
  modelo.
- A perplexidade PPM reportada é uma calibração interna, não uma perplexidade
  de LLM nem uma avaliação humana.
- A Wikipédia em português pode conter trechos em outras línguas, listas e
  marcação residual após a limpeza. O corpus é tratado como texto de estilo,
  não como verdade factual.
- Textos antigos do Gutenberg preservam ortografia histórica. Eles ampliam o
  vocabulário, mas não normalizam automaticamente grafias pré-reforma.
