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
./build_rocm/cognicore chat --model modelo --debug
```

A entrada de treino de diálogo é JSONL com campos `text` e `intent`. As intenções
são `EXPLAIN_CAUSE`, `ANSWER_FACT`, `COMPARE`, `ASK_CLARIFY`, `AGREE`, `HYPOTHESIZE`
e `GREET`. O treino atualiza autômatos inteiros e contagens de construções/n-gramas;
não há treino de uma rede neural.

```bash
./build_rocm/cognicore ingest-code --repo /caminho/projeto --model modelo --use-gpu
./build_rocm/cognicore code --model modelo --workspace /caminho/projeto \
  --lang py --task 'combine load_items e compute_total'
./build_rocm/cognicore chat --model modelo --workspace /caminho/projeto \
  --lang py --realizer symbolic --debug
./build_rocm/cognicore export-realizer-dataset --model modelo --out planos.jsonl
```

`assistant` é um alias de `chat`. `--lang` seleciona a linguagem do código.
`--realizer symbolic` é o padrão: gramática de construções + unificação morfológica
+ pontuador n-grama. `--realizer morphological` seleciona a realização morfológica
determinística para comparação. Outras opções são rejeitadas.

Sem workspace, a sessão é salva no modelo; com workspace, em
`<workspace>/.cognicore`. Essa pasta é excluída da ingestão. Os últimos 128 episódios,
os fatos aprendidos e o índice de código são recarregados. `conversation.jsonl`
registra o histórico em disco. `export-realizer-dataset` exporta planos e frases
originais para auditoria e análise linguística, sem dependência de treino neural.

## Marcos preservados e readequados

| Marco | Implementação |
|---|---|
| 1–6 | VSA, SDM, Tsetlin, equivalência HIP/CPU, persistência e teste integrado original. |
| 7 | CoNLL-U com lema, UPOS, dependências, gênero/número, tempo, modalidade e negação; fallback de regras; morfologia PT/EN, anáfora e conectivos. |
| 8 | Tree-sitter C++/Python/JavaScript/TypeScript; assinaturas, documentação, formas AST canônicas, VSA na SDM e grafo de chamadas/tipos/parâmetros/membros/herança/imports. |
| 9 | Seleção Tsetlin e composição estrutural de chamadas compatíveis; impressão e validação sintática com Tree-sitter. Testes compilam/executam C++ e executam Python gerados. |
| 10, readequado | Fluência não neural: frames compatíveis, slots morfológicos e seleção de composições inéditas por unigramas/bigramas/trigramas interpolados. Exportação simbólica dos planos preservada. |
| 11 | Roteador Tsetlin de cinco classes, conversa, aprendizado imediato, raciocínio causal, assistência de código e sessões persistentes. |

O realizador recebe somente um `SemanticPlan` através de `ISurfaceRealizer`.
As construções preenchem sujeitos, predicados e objetos do plano. N-gramas escolhem
entre realizações equivalentes e conectivos compatíveis; não escolhem fatos nem
substituem a relação por outro verbo. Contagens aprendidas são serializadas e
recarregadas, sem busca de uma resposta pronta no corpus.

O teste do Marco 10 comprova composição com conceito ausente no texto observado,
influência das contagens na escolha dos conectivos, preservação de negação,
rejeição de frame com predicado incompatível e persistência após recarga.

## Compatibilidade e limites

Os arquivos anteriores do núcleo continuam utilizáveis. O grafo migra o layout
original sem metadados; índice de código, episódios e pares são arquivos opcionais
independentes. O histórico Git não foi reescrito na readequação.

A análise morfossintática real usa CoNLL-U produzido externamente; não há UDPipe
embutido. O fallback por regras reconhece um conjunto limitado de construções.
A conversão geral de `advcl`/`mark` em condições executáveis ainda precisa ser
ampliada. A síntese atual compõe até duas funções livres unárias com tipos
compatíveis; patches arbitrários, overloads e métodos com instâncias não estão
concluídos. O roteador e a fluência ainda precisam de avaliação em corpus amplo.
A meta de RAM depende do tamanho do corpus e do índice; não foi comprovada para
repositórios arbitrariamente grandes.
