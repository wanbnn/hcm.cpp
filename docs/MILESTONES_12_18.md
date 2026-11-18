# Marcos 12–18 — geração longa, léxico, coerência, matemática, código e escala

Esta série mantém a arquitetura pós-LLM: VSA binária, Kanerva SDM, Tsetlin,
grafo causal, DOP/PPM e morfologia. Não há Transformer, backpropagation ou peso
neural. Todo dado linguístico específico do modelo fica em
`language_rules.json` dentro do diretório do artefato.

## Marco 12 — Gerador de documento

- `DocumentPlanner` transforma tópico → outline → seções → parágrafos → planos.
- Seções: introdução, causas, consequências, alternativas, conclusão.
- Revisão determinística: remove sentenças duplicadas, evita conectivo repetido,
  normaliza pontuação.
- CLI: `./cognicore write --model <dir> --topic calor --sections 5 --paragraphs 2`.
- Teste: `test_marco12_document`.

## Marco 13 — Léxico e escolha lexical

- `Lexicon` aprende lemas, POS, gênero/número, frequência, tons e colocações do
  corpus.
- `LexicalChooser` aplica escolhas de superfície (`gerar` → `produzir`,
  `servidor` → `host`) sem alterar a aresta/fato do grafo.
- Sinônimos e evidência de colocação vêm de `language_rules.json` e do corpus.
- Persistência: `lexicon.bin` no modelo.
- Teste: `test_marco13_lexicon`.

## Marco 14 — Gramática lexicalizada

- `LexicalizedGrammarRealizer` monta relativas, subordinações e agregações:
  - `O atrito gera o calor, que derrete o gelo.`
  - `O atrito gera o calor porque o calor derrete o gelo.`
  - `O atrito gera o calor e aquece o gelo.`
- Padrões e pontuação são dados do modelo (`relative_patterns`,
  `subordinate_patterns`, `aggregation_patterns`).
- CLI: `--realizer lexical`.
- Teste: `test_marco14_lexicalized`.

## Marco 15 — Rankeador de coerência

- `CoherenceModel`: n-gramas com desconto absoluto, cache tópico, centering e
  verificação de ancoragem no grafo.
- Reordena candidatas do realizador fluent antes da escolha final.
- Persistência: `coherence.bin`.
- Teste: `test_marco15_coherence`.

## Marco 16 — Motor matemático simbólico

- `SymbolicMath`: frações, parênteses, potências, porcentagens e equações
  lineares de uma variável, com verificação por substituição.
- CLI: `./cognicore math --expr "2x + 3 = 11"`.
- O assistente usa o motor simbólico antes do solver aritmético simples.
- Teste: `test_marco16_symbolic_math`.

## Marco 17 — Síntese de código com testes

- `CodeSynthesizer::synthesize_tested`: gera candidatos, inclusive controle de
  fluxo (loop de acumulação), executa testes Python em workspace e escolhe o
  primeiro candidato aprovado.
- Registra iterações, testes gerados e notas de reparo.
- Teste: `test_marco17_code_tests`.

## Marco 18 — Ingestão massiva, benchmark longo e avaliação humana

- `ingest-corpus`: ingestão incremental com `ingestion_manifest.jsonl`,
  checkpoint, retomada e suporte a texto/JSONL.
- Benchmark próprio inclui geração longa (estrutura, cobertura de conceitos,
  taxa de sentenças únicas).
- `long_form_rating_sheet.jsonl` é gerado vazio; notas só entram com avaliadores
  reais.
- Teste de CLI cobre manifesto/retomada em `test_cli_phase2.py`.

## Chat unificado

Os marcos 12–18 não são comandos separados para o usuário final. O `chat`
roteia automaticamente:

- `Escreva um texto sobre calor` → `DocumentPlanner`.
- `Resolva 2x + 3 = 11` → `SymbolicMath`.
- `Quanto é 1/2 + 1/3?` → `SymbolicMath`.
- `Escreva um script python...` → síntese de código.
- `O atrito gera calor.` → aprendizado one-shot no grafo/SDM.
- Perguntas causais continuam no grafo.

## Comandos

```bash
./build/cognicore ingest-corpus --input corpus/ --model modelo --format auto
./build/cognicore write --model modelo --topic calor --realizer fluent
./build/cognicore chat --model modelo --realizer lexical --debug
./build/cognicore math --expr "2x + 3 = 11"
python3 benchmarks/run_benchmarks.py --cli build/cognicore --model modelo --out benchmarks/results
```

## Sem fallback de resposta

Todo texto de resposta vem de plano/construções do modelo:
- sociais, concordância, clarificação, tópico desconhecido, recursos e
  condicionais são templates em `language_rules.json`, com slots;
- conectivos de discurso também são dados do modelo;
- código vem de `code_patterns.jsonl` ou do grafo/AST do workspace;
- matemática vem do motor simbólico e dos templates de resultado do modelo.

Mensagens de erro/recusa (API ausente, workspace sem símbolo, expressão inválida)
são explícitas e não se apresentam como texto gerado. O teste
`test_no_fallback` altera templates do modelo e verifica que a saída muda;
nenhuma resposta é buscada em uma frase pronta no C++.

## Limites

- O plano de documento é retórico e simbólico; não é criatividade aberta.
- O léxico é limitado ao corpus e às colocações observadas.
- A gramática lexicalizada cobre os padrões em `language_rules.json`; não é uma
  STSG irrestrita.
- Kneser-Ney/cache são rankeadores simbólicos, não uma perplexidade de LLM.
- A matemática cobre aritmética racional e equações lineares; não é um CAS
  completo (integrais, sistemas gerais, geometria).
- A síntese de código é generativa por construções do modelo
  (`code_patterns.jsonl`), com slots preenchidos e validação AST; não há
  template hardcoded no assistente. A prova de correção ainda não é semântica
  para qualquer programa.
- A avaliação humana exige raters reais; o repositório não inventa notas.
