Versão mais recente para teste: [conversa-pt-fluent-v3.1](MODEL_CONVERSA_PT_FLUENT_V31.md). Esta ficha descreve a versão 1 preservada.

# Conversa PT v1 — modelo simbólico para teste local

Modelo pronto em `models/conversa-pt-v1`, um link local para:

```text
/run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-v1
```

## Conversar agora

Na raiz do projeto:

```bash
./scripts/chat_portuguese.sh --debug
```

O launcher usa `build/cognicore`, compilado somente para CPU. Para manter o modelo
base separado da memória das experiências:

```bash
mkdir -p /tmp/minha-conversa-pt
./scripts/chat_portuguese.sh --workspace /tmp/minha-conversa-pt --debug
```

A sessão persistirá em `/tmp/minha-conversa-pt/.cognicore`. Sem `--workspace`, os
novos episódios e fatos são gravados no próprio diretório do modelo. É possível
selecionar outro modelo por `COGNICORE_MODEL=/caminho/do/modelo`.

Experimente:

```text
Olá!
Quem é você?
Tudo bem?
O que você pode fazer?
Obrigado pela ajuda.
Por que o atrito pode derreter gelo?
O que é a SDM?
O vento esfria a água.
O que eu disse?
O que o vento faz?
O que você sabe sobre teletransporte?
Tchau!
sair
```

A informação sobre vento não pertence ao grafo inicial: ela é aprendida no turno
de ensino. A pergunta sobre teletransporte deve receber uma indicação de
conhecimento insuficiente, não um fato arbitrário recuperado pela SDM.

## Arquitetura e treino

- VSA binária de 4096 bits, Kanerva SDM e grafo causal.
- Tsetlin semântica: 4160 características, 12 classes e 20 cláusulas por classe;
  6 épocas com os kernels HIP na Radeon RX 9060 XT, `gfx1200`.
- Tsetlin de atos de fala: 16 características simbólicas, 12 classes e 64
  cláusulas por classe; 12 épocas em CPU. Generaliza sinais de apresentação,
  agradecimento, despedida, ajuda e outros atos curtos. A Tsetlin semântica
  continua responsável pelos casos sem esses sinais específicos.
- Gramática de construções, unificação morfológica e n-gramas interpolados.
  O pontuador escolhe realizações do plano; não escolhe fatos.
- Nenhum treino neural, Transformer, backpropagation, API de LLM ou pesos
  pré-treinados. DuckDB é usado somente para ler os arquivos Parquet do corpus.

A política foi treinada em **4.254 exemplos distintos**; o balanceamento repete
exemplos até 8.400 amostras por época. Isso não aumenta o número de exemplos
originais. Há **1.060 exemplos reservados**, separados por conversa/documento,
com zero sobreposição exata dos textos de classificação após normalização.
Rótulos são mapeados de atos de fala ou atribuídos por supervisão lexical fraca:
a avaliação não constitui um benchmark humano gold.

Os exemplos reservados não entram na aprendizagem de estilo. Os testes de
fumaça manuais também são separados do treino textual, mas foram usados no
desenvolvimento e diagnóstico; não são uma avaliação independente.

## Corpus em português

Foram observados **35.797 segmentos linguísticos** e aproximadamente 5,5 MB de
texto UTF-8. Os segmentos incluem turnos, instruções e spans anotados; não são
35.797 conversas independentes. Literatura não entra nessa contagem.

| Fonte | Segmentos usados em estilo | Origem e licença declaradas |
|---|---:|---|
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 1.031 | Mensagens humanas `pt-BR`, revisadas; linhas sintéticas/deletadas excluídas. Apache-2.0. |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | 1.196 | Mensagens humanas `pt-BR` adicionais; deduplicação por ID entre as versões. Apache-2.0. |
| [PortugueseChat](https://huggingface.co/datasets/rishiraj/portuguesechat) | 19.962 | Tradução portuguesa de no_robots, incluindo instruções e conversa; não se presume tradução humana. CC-BY-NC-4.0. |
| [Brazilian Customer Service Conversations](https://huggingface.co/datasets/RichardSakaguchiMS/brazilian-customer-service-conversations) | 5.110 | Diálogos de atendimento sintéticos, já publicados pelo autor; não são conversas reais. Apache-2.0. |
| [Atos de Fala PT-BR](https://huggingface.co/datasets/lucianfialho/atos-de-fala-ptbr-dataset) | 8.226 | Exemplos e spans sintéticos anotados; não são gold humano. CC-BY-4.0. |
| Exemplos controlados do projeto | 272 | Variações de construções de diálogo e perguntas sobre o motor. |

Também foram aprendidas estatísticas de *Dom Casmurro*, de Machado de Assis,
[edição Gutenberg 55752](https://www.gutenberg.org/ebooks/55752), obtida pelo
[espelho GITenberg](https://github.com/GITenberg/Dom-Casmurro_55752). O Gutenberg
identifica a obra como domínio público nos EUA; os avisos do arquivo original
foram conservados no cache de dados.

Parte do corpus foi criada anteriormente com LLMs pelos publicadores dos dados.
Neste experimento, esses textos alimentam contagens de estilo e exemplos de
atos de fala. Nenhum LLM é executado e nenhuma resposta desses datasets é
importada como verdade no grafo.

O corpus inclui uma fonte marcada CC-BY-NC-4.0; esta entrega é destinada a
experimentação local não comercial. As licenças individuais e os créditos
constam dos cartões originais e de `corpus_manifest.json`, com URLs e SHA-256.
A licença do código do motor é independente das licenças dos dados.

## Conhecimento factual inicial

O grafo inicial contém **19 conceitos e 12 relações**, curados separadamente:
alguns exemplos de física adaptados de [OpenStax Physics](https://openstax.org/books/physics/pages/11-section-summary)
e informações sobre a arquitetura retiradas do código/README do projeto.
A atribuição de OpenStax é CC-BY-4.0, conforme o
[prefácio do livro](https://openstax.org/books/physics/pages/preface).

`data/conversation_pt/facts.jsonl` registra cada fonte. Dois enunciados que
exigiam qualificações sobre mudança de fase foram excluídos da indexação.
Livros de ficção e diálogos públicos são ingeridos com `--style-only`: não criam
fatos ou gravações factuais na SDM. Mais texto de conversa não equivale a mais
conhecimento verificável no grafo.

## Resultados medidos

- Modelo consolidado: aproximadamente **37 MiB em disco**.
- Vocabulário VSA: **16.010 entradas**; aprendizado contextual de estilo limitado
  a 16 mil entradas, com amostragem de uma posição a cada quatro tokens.
- **42 construções** e **804.232 n-gramas** distintos.
- Treino da política semântica + contextual: aproximadamente **54 segundos**
  nesta máquina; ingestão inicial de estilo: aproximadamente 11 segundos.
- Acurácia de intenção no conjunto reservado: **72,55% (769/1.060)**, com rótulos
  fracos e distribuição desigual entre classes. A matriz completa está em
  `heldout_metrics.json`; não é medida de qualidade geral de conversa.
- Frases diagnósticas de fumaça: **18/18** intenções reconhecidas. Usadas durante
  desenvolvimento; não representam generalização ampla.
- Sessão CPU de 12 turnos: **7/7 verificações funcionais** — identidade,
  agradecimento, dois hops, aprendizado imediato, memória episódica,
  despedida e recusa de associação sem fundamento para tema desconhecido.
- Pico de RSS medido na sessão: aproximadamente **121 MiB**, incluindo o processo
  completo. Os cerca de 58 MiB exibidos por `stats` são apenas uma estimativa das
  estruturas, não o RSS real.

Relatórios e transcrição integral estão no diretório do modelo:
`training_report.json`, `heldout_metrics.json`, `probe_metrics.json` e
`conversation_evaluation.json`. Uma tentativa inicial e o diagnóstico anterior
à correção de associações sem fundamento foram preservados para comparação.

## Reproduzir

Preparação com Python e DuckDB, sem bibliotecas neurais:

```bash
uv venv /caminho/venv-dados
uv pip install --python /caminho/venv-dados/bin/python duckdb
/caminho/venv-dados/bin/python scripts/prepare_portuguese_conversation.py \
  --out /caminho/corpus-pt
```

O livro é opcional: o preparador usa `raw/dom_casmurro_55752_latin1.txt` quando o
arquivo estiver presente. Os cinco datasets são obtidos automaticamente.
Os arquivos reservados não são ingeridos pelo comando de treino.

```bash
python3 scripts/train_portuguese_model.py \
  --data /caminho/corpus-pt --model /caminho/modelo-novo \
  --cli ./build_rocm/cognicore --epochs 6 --clauses 20 --use-gpu
python3 scripts/evaluate_portuguese_conversation.py \
  --model /caminho/modelo-novo --out /caminho/relatorio-sessao.json
```

O diretório de saída deve estar vazio. `--resume-policy` reutiliza o corpus já
ingerido, reinicializa os classificadores e treina outra política; preserva o
grafo, mas volta a observar os exemplos nas contagens de estilo. Para reprodução
com contagens idênticas, comece em um diretório novo com o mesmo cache de dados.

## Limites para o teste

Este é um protótipo de conversa social e aprendizado explícito, com construções
sociais limitadas. Respostas abertas longas, opiniões, narrativas, resolução
ampla de pronomes e de contexto, aprendizagem de preferências e conhecimento
geral ainda não estão resolvidos. O aumento do corpus amplia estatísticas e
exemplos de política; não cria sozinho novas capacidades semânticas ou regras
corretas. A SDM só expande conceitos ancorados na entrada para evitar que uma
consulta sem evidência produza uma resposta factual sobre outro assunto.
