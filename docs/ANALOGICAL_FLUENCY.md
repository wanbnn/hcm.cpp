# Marco 10: fluência analógica sem rede neural

O componente é `AnalogicalFluencyRealizer : ISurfaceRealizer`. Treino, indexação,
ranking e geração são C++20; `--use-gpu` acelera a Tsetlin de estilo e a escrita
SDM através do backend HIP existente. A inferência usa exclusivamente CPU.
Não existem executor neural, pesos neurais ou comando de exportação para decoder.

```
./build_rocm/cognicore ingest-books --input corpus --model modelo --style-only
./build_rocm/cognicore train-dialogue --input dialogos.jsonl --model modelo --epochs 6
./build_rocm/cognicore train-realizer --model modelo --use-gpu
./build/cognicore chat --model modelo --realizer fluent --debug
./build/cognicore chat --model modelo --realizer strict --debug
```

## Mecanismo implementado

1. A ingestão aprende produções de cláusula com folhas tipadas de sujeito,
   predicado e objeto. Há templates genéricos (qualquer relação), taxonômicos
   (`é [OBJ]` / `is [OBJ]`) e de propriedade, todos licenciados pelo tipo da
   aresta do grafo. Produções binárias cobrem coordenação, consequência, adição
   e contraste (`mas`, `porém`, `contudo`, `entretanto`, `however`, `but`) com
   conectivos observados. Os identificadores lexicais das entidades são removidos.
   Predicados lexicalizados restringem a compatibilidade, nunca trocam o verbo.
2. O índice VSA/SDM agrupa IDs de produções por forma, idioma e tom. A recuperação
   associativa ordena candidatos; a compatibilidade gramatical é uma restrição
   obrigatória. Uma leitura SDM não demonstra verdade ou compatibilidade sozinha.
   O contraste só é licenciado quando o elo causal é descontínuo, inibidor ou
   negado; ele não é inserido numa cadeia causal contígua.
3. O beam search combina produções de origens diferentes, mantendo até 16
   derivações distintas. Cada hop do caminho selecionado é realizado pelo motor
   morfológico. Conectivos de consequência exigem elos causais contíguos sem
   negação/modalidade; os demais casos usam coordenação neutra.
4. Um verificador determinístico confere os nós semânticos e a sequência textual
   produzida pela morfologia. Remover um hop, inserir outra afirmação ou mudar
   polaridade invalida a candidata. Uma Tsetlin probabilística não constitui
   garantia de preservação semântica: a garantia é estrutural.
5. O ranking usa PPM com escape C e backoff de ordens 5 a 0 sobre bytes, sem
   exclusão, mais o voto da Tsetlin para o tom desejado e uma penalidade para repetição do mesmo conectivo. O custo é expresso em
   bits aproximados por byte em Q16; não é uma perplexidade linguística calibrada.
   A aproximação inteira de log evita álgebra neural ou matrizes densas.
6. `fluency.bin` guarda produções, contagens e cláusulas esparsas. Produções e
   contagens são consultadas diretamente no mapeamento, sem reconstruí-las em
   mapas durante inferência. O pequeno índice SDM é reconstruído ao carregar.
   Escritas usam arquivo temporário e rename para preservar leitores mapeados.

PPM usa a família de escape/backoff descrita por
[Moffat, Implementing the PPM data compression scheme](https://cs.nyu.edu/home/people/in_memoriam/roweis/csc310-2006/extras/implementing_ppm.pdf).
A pontuação aqui é uma variante limitada, com log aproximado e limites explícitos
para manter memória previsível: 80 mil contextos, 400 mil transições, 2.048
produções. Contextos existentes continuam recebendo contagens após o limite.

## Validação e limites

`tests/test_improvements.cpp` cobre templates taxonômicos, recombinação de
contraste, resolução contextual e planejamento de rota alternativa, mantendo os
testes dos marcos anteriores. O teste do Marco 10 exige 8–16 variações distintas para um plano de três hops
com construções suficientes, inclui conceitos ausentes do corpus, verifica os
custos/ranking, negação e modalidade, rejeita perda/inserção de conteúdo e testa
recarga mmap e atualização atômica. O número de alternativas depende do plano e
do repertório: um único hop, hipóteses estruturadas ou corpus insuficiente podem
produzir apenas a realização strict. Não são fabricadas variações para cumprir
uma contagem.

A extração DOP atual cobre uma gramática de cláusulas (genérica, taxonômica,
propriedade) e combinações binárias (coordenação, consequência, adição,
contraste); não é um parser constituinte geral de português nem DOP irrestrito.
A verificação reconstrói a cláusula com o template efetivamente usado e o
conectivo declarado pela produção, tanto em memória quanto após recarga mmap.
Não há Transformer, pesos neurais ou decodificador estatístico neural.
Os tons recebem supervisão fraca de marcadores observados e exemplos de
calibração; isso não prova avaliação humana de naturalidade. Avisos de recursos,
citações episódicas e conclusões condicionais estruturadas usam o realizador
strict para preservar os campos especiais do plano. O mesmo vale para modelos
anteriores sem `train-realizer`, com aviso explícito na CLI.

Memória contextual, planejamento de discurso e síntese AST foram tratados em
módulos separados: `ContextMemory` (entidades, pronomes e tópico),
`DiscoursePlanner` (tese e rota alternativa) e o compositor por dataflow de
assinaturas. O grafo causal agora também é consultado por `graph_mmap.bin` em
modo somente leitura. A compreensão geral de texto é baseada em regras internas
e aceita CoNLL-U externo; não é um parser neural/UDPipe.
