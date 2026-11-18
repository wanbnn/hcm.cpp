# Raciocínio simbólico auditável

O fluxo CPU é tokenização → memória/associação → Tsetlin → grafo → plano semântico → `ISurfaceRealizer`. Não há Transformer nem treinamento neural. O plano fica disponível em `InferenceTrace::plan`; um realizador substituto recebe a mesma decisão estruturada antes da produção do texto.

Execute `scripts/chat_portuguese.sh --workspace /tmp/meu-teste-hcm --debug` para usar o modelo português existente em uma sessão isolada. O diretório de workspace deve existir.

## O que foi implementado

- Memória episódica de propriedades e regras, com IDs estáveis, fonte original e índice Kanerva separado. Endereços ligam entidade e episódio; conteúdos ligam propriedade e valor. Leituras com similaridade abaixo de 0,15 são rejeitadas. Esse valor mede recuperação de bits, não probabilidade de verdade.
- `ContextMemory` mantém entidades do discurso com gênero, número, menções, recência e saliência. Pronomes (`ele`, `ela`, `isso`, `they`, `it`) e dêixis (`lá`, `nesse servidor`) são resolvidos por compatibilidade gramatical e saliência; empates marcam a referência como ambígua em vez de adivinhar. Tópicos só são herdados em turnos de acompanhamento (`e isso?`, `por quê?`, `continue`) sem conceito próprio. O estado é persistido em `context.bin` quando presente.
- O planejador de discurso escolhe a tese que termina em uma semente da pergunta, remove prefixos cobertos por derivações mais longas e anexa no máximo uma rota alternativa que compartilhe nó com a tese. A rota alternativa é verbalizada como contraste/alternativa, não como novo elo causal. A seleção é simbólica; não há LLM no planejamento.
- Capacidade de RAM declarada e demanda explícita são comparadas por aritmética inteira. GB é decimal; GiB é binário. Tamanho ausente ou múltiplas quantidades deixam a demanda desconhecida. Corrigir uma capacidade muda a próxima decisão sem retreino.
- Regras explícitas `Se A então B` ficam na memória como implicações, fora das arestas incondicionais. A pergunta hipotética precisa conter os literais da premissa, inclusive marcadores negativos. Uma consequência pode iniciar uma continuação no grafo.
- O plano contém hipóteses, avaliações de recursos e IDs de evidência. Os realizadores só verbalizam esses resultados. As respostas sociais continuam usando um repertório limitado de construções; não equivalem a conversa aberta de um LLM.
- O debug exibe tempos medidos com relógio monotônico, votos das duas políticas Tsetlin, cláusulas realmente ativas da política de diálogo, seus literais, slots/Hamming, fontes episódicas e IDs de arestas. Cláusulas vazias são identificadas. Os tempos incluem a instrumentação; não há promessa de menos de 10 ms.
- O compositor liga parâmetros por dataflow: cada argumento vem da entrada, de um retorno anterior de tipo compatível ou de um literal primitivo explícito. Suporta múltiplos parâmetros, funções sem argumentos, retorno `void` e métodos qualificados, com validação Tree-sitter. Registra IDs de símbolos, votos e IDs de cláusulas do seletor. Não troca APIs por vizinhos VSA arbitrários quando falta evidência, e parâmetros não primitivos sem produtor válido rejeitam a composição.

## Exemplo para testar

```
Meu servidor antigo só tem 2 GB de RAM e roda em CPU.
Se eu tentar carregar um grafo gigante inteiro na RAM lá sem mmap, o que acontece?
Se eu carregar 3 GB na RAM do servidor antigo, o que acontece?
Meu servidor antigo tem 4 GB de RAM.
Se eu carregar 3 GB na RAM lá, o que acontece?
```

A primeira pergunta pede a demanda ausente; a segunda excede a capacidade declarada; após a correção, 3 GB não excedem a capacidade total de 4 GB. Isso não prova disponibilidade de RAM livre. Não são inventados swap, OOM ou efeitos de mmap.

## Limites explícitos

A análise de regras inclui SVO por léxico verbal, coordenação, relativas,
`advcl`, quantificadores simples e condicionais temporais/causais/concessivas em
PT/EN. Escopo aninhado, quantificação complexa e discurso arbitrário continuam
limitados; CoNLL-U externo permanece disponível para dependências de maior
precisão. O grafo é consultado por `graph_mmap.bin` em modo somente leitura. O
registro episódico fica em `episodic_facts.bin`, o estado contextual em
`context.bin`; o índice SDM é reconstruído em CPU ao carregar. Formatos
anteriores permanecem legíveis.

O cenário completo de geração de um carregador binário com campos, checksums e contratos de APIs arbitrárias ainda não está implementado. Pedidos desse tipo recebem uma recusa específica, em vez de código sem relação com o pedido. Tree-sitter verifica sintaxe, não prova correção de tipos ou integridade binária.

O modelo português existente é reutilizável: estas mudanças adicionam mecanismos simbólicos e não alteram a representação dos pesos/cláusulas treinados. Não foi necessário novo treinamento nem download de corpus.
