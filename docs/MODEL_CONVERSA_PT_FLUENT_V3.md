# Modelo português v3 com parser contextual, grafo mmap e AST ampliada

Artefato: `models/conversa-pt-fluent-v3` (link para
`/run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3`).
As versões `conversa-pt-v1` e `conversa-pt-fluent-v2` foram preservadas.

Treinamento CPU, 100% simbólico: nenhum Transformer, peso neural, PyTorch,
libtorch ou backpropagation. O modelo é o mesmo tipo de artefato dos anteriores;
as mudanças desta versão são de parser, memória contextual, planejamento,
grafo mmap, síntese AST, calibração PPM e ferramentas de avaliação.

As pistas linguísticas específicas do modelo (prefixos de instrução/correção,
perguntas, geração de código, aritmética, acompanhamento, quantificadores e
palavras de operadores matemáticos) ficam em `language_rules.json`, salvo no
diretório do artefato. A semente `data/portuguese_model_rules.json` é copiada
para o modelo; o código de inferência não contém essas listas.

```
./scripts/chat_portuguese.sh --model models/conversa-pt-fluent-v3 --realizer fluent --debug
```

O script usa por padrão `models/conversa-pt-fluent-v2`; defina
`COGNICORE_MODEL` ou passe `--model` para a v3.

## Treinamento

Dados: `public_pt_v1` (OpenAssistant, PortugueseChat, atendimento sintético,
atos de fala e Dom Casmurro). O script foi executado com o CLI CPU:

```
python3 scripts/train_portuguese_model.py \
  --data /run/media/wanderson/DADOS_1TB/cognicore/data/public_pt_v1 \
  --model /run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v3 \
  --cli build/cognicore --epochs 6
```

Tempos desta máquina: indexação de estilo 25,5 s; fatos CoNLL-U 1,7 s;
política 11,4 s; treino do realizador 1,7 s; consolidação 0,7 s. Não são uma
previsão para outros corpora.

## Métricas

- 16.010 tokens no codebook VSA; 19 nós e 12 arestas no grafo factual.
- 1.136 produções estruturais; 80.000 contextos PPM; 47 frames; 804.232 n-gramas.
- Disco: ~67 MB. RAM estimada: ~146 MB (sem contar mmap do grafo/fluência e RSS
  de arquivos mapeados; o pico medido no benchmark foi ~733 MB no processo CLI).
- Intenção reservada: 769/1.060 (72,55%); probes sintéticos: 18/18 (100%).
  Os rótulos são parcialmente heurísticos e não constituem avaliação humana.
- Calibração PPM na sonda de `benchmarks/data/heldout_pt.txt`:
  2,216 bits/byte, perplexidade 4,647. É uma sonda pequena, não um corpus
  representativo e não uma perplexidade de LLM.

## Benchmark próprio (inspirado, não os benchmarks originais)

`benchmarks/run_benchmarks.py` cria um modelo de diagnóstico a partir da v3,
ingere fatos científicos e executa categorias inspiradas em GPQA, HLE, GSM8K,
HumanEval e SWE-bench:

| Categoria | Resultado |
|---|---:|
| science (GPQA-style) | 5/5 |
| hard reasoning (HLE-style: condicional, temporal, relativa, quantificador) | 3/3 |
| math/word problems (GSM8K-style) | 5/5 |
| code generation (HumanEval-style) | 2/2 |
| patch (SWE-bench-style) | 1/1 |
| fluency checks (strict vs fluent) | 3/3 |

O conjunto é pequeno e foi escrito para este projeto; não é comparável aos
números oficiais dos benchmarks citados. O relatório completo, respostas e
latências estão em `benchmarks/results/benchmark_report.json`.

A planilha de avaliação humana de fluência é gerada em
`benchmarks/results/fluency_rating_sheet.jsonl`. Ela fica vazia por projeto:
nenhuma nota humana foi inventada. Use
`scripts/evaluate_fluency_human.py score` depois que avaliadores reais
preencherem as notas.

## Marcos 12–18

O artefato agora carrega também `lexicon.bin`, `coherence.bin`,
`ingestion_manifest.jsonl` e as regras de documento/gramática em
`language_rules.json`. Comandos novos: `write`, `ingest-corpus`, `math` e
`--realizer lexical`. Detalhes em [Marcos 12–18](MILESTONES_12_18.md).

## Limites

- O parser de dependências é interno e baseado em regras; não é UDPipe nem um
  parser neural de dependências. Quantificadores, relativas e `advcl` simples
  são convertidos em metadados/regras executáveis, mas escopo, quantificação
  aninhada e discurso arbitrário continuam limitados.
- A síntese AST cobre overloads, métodos estáticos/instância e patches
  compostos pelas chamadas internas do alvo; não cobre qualquer patch textual
  arbitrário nem prova de tipos do compilador.
- O grafo é consultado por `graph_mmap.bin` em modo somente leitura; escritas
  continuam no grafo em memória e exigem regeneração do índice.
- PPM é um rankeador estatístico. A perplexidade reportada é uma calibração
  interna, não uma garantia de naturalidade.
- As licenças herdadas do corpus incluem restrições não comerciais; consulte
  `corpus_manifest.json` e a ficha original antes de redistribuir.
