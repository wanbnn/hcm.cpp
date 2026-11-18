# Modelo português com realizador analógico (v2)

Esta ficha descreve o artefato v2. A versão atual, com parser contextual, grafo
mmap, AST ampliada e benchmarks, é a
[v3](MODEL_CONVERSA_PT_FLUENT_V3.md).

Artefato: `models/conversa-pt-fluent-v2` (link para
`/run/media/wanderson/DADOS_1TB/cognicore/models/conversa-pt-fluent-v2`).
A versão anterior `conversa-pt-v1` foi preservada.

```
mkdir -p /tmp/hcm-fluent-workspace
./scripts/chat_portuguese.sh --workspace /tmp/hcm-fluent-workspace --realizer fluent --debug
```

Para comparar a verbalização determinística, use `--realizer strict`.

O modelo reutiliza as cláusulas de diálogo da versão 1, reingere em C++ os sete
arquivos de estilo em português e treina a nova Tsetlin de estilo com HIP/gfx1200.
Não foi treinada nenhuma rede neural. A indexação do corpus levou aproximadamente
19 segundos; `train-realizer --use-gpu`, 19 segundos nesta máquina, incluindo
carregamento e persistência. Esses tempos não são uma previsão para outros corpora.

São 47 produções estruturais e 80 mil contextos PPM; `fluency.bin` tem cerca de
4,6 MiB. O limite de contextos foi atingido: isto é um treinamento limitado em
memória, não retenção irrestrita de todos os contextos do corpus. As 12 arestas
de conhecimento da versão anterior permanecem separadas do corpus de estilo.
Livros literários e diálogos não foram promovidos automaticamente a fatos.

A avaliação de 1.060 exemplos reservados de intenção registrou 778 acertos
(73,40%). São rótulos parcialmente heurísticos, não um benchmark humano de
qualidade de conversa. A sessão de 12 turnos passou nos sete checks funcionais,
com aproximadamente 128 MiB de RSS no teste CPU. Uma avaliação separada do
realizador exige 8–16 candidatas para um plano causal inédito de três hops e
confere que nenhuma saída completa aparece literalmente no corpus ingerido.

Relatórios no diretório do artefato:

- `fluent_conversation_evaluation.json`: respostas, checks e pico de RAM.
- `fluent_heldout_metrics.json`: classificação e matriz de confusão.
- `novel_plan_evaluation.json`: plano, variações, custos PPM e votos de estilo.
- `corpus_manifest.json`: fontes e licenças herdadas da versão anterior.

O corpus inclui OpenAssistant, PortugueseChat, dados sintéticos de atendimento
/ atos de fala e Dom Casmurro. A combinação de licenças inclui restrições não
comerciais; consulte o manifesto e a [ficha original](MODEL_CONVERSA_PT.md) antes
de redistribuir o artefato.

Esta versão não é um assistente geral concluído. O parser de texto livre agora
extrai SVO por léxico verbal, coordenação e condicionais `se ... então`; a memória
contextual resolve pronomes/dêixis e herda tópico apenas em turnos de
acompanhamento sem conceitos próprios; o planejador escolhe a tese ligada às
sementes da pergunta e pode anexar uma rota alternativa do grafo; a síntese AST
liga parâmetros por dataflow, com múltiplos argumentos, retorno `void`, funções
sem argumentos e métodos qualificados. A gramática DOP distingue cláusulas
genéricas, taxonômicas e de propriedade e aprende conectivos de contraste.

Continuam limitados: escopo/quantificação aninhada, discurso arbitrário,
conhecimento factual amplo, patches textuais arbitrários e resolução de templates.
O parser de dependências interno cobre SVO, relativas, `advcl` e quantificadores
simples, mas não é UDPipe. Overloads, métodos e patches estruturais já são
cobertos pela síntese AST. A recombinação DOP não é um parser constituinte
irrestrito; o PPM agora reporta bits/byte e perplexidade de calibração, mas isso
não é garantia humana de fluência. Detalhes: [Marco 10](ANALOGICAL_FLUENCY.md)
e [modelo v3](MODEL_CONVERSA_PT_FLUENT_V3.md).
