# API OpenAI-compatible para CogniCore

`scripts/openai_api.py` expõe os modelos CogniCore em uma API HTTP compatível
com o subconjunto OpenAI usado por ferramentas como Hermes, OpenCode, Continue,
LangChain e clientes OpenAI locais.

Não é um servidor neural. Cada completion chama o CLI `cognicore chat` em modo
one-shot (`--prompt ... --no-save`), portanto a resposta vem do grafo, do
discurso e do realizador simbólico do modelo escolhido.

## Iniciar

```bash
cd /home/wanderson/Projetos/tessel
python3 scripts/openai_api.py \
  --host 127.0.0.1 --port 8000 \
  --models-dir models \
  --default-model conversa-pt-fluent-v3.2
```

O padrão é `127.0.0.1:8000` e CLI `build/cognicore`.

Em ferramentas externas:

```text
Base URL: http://127.0.0.1:8000/v1
API key:  local   (qualquer valor, a menos que COGNICORE_API_KEY esteja definido)
Model:    conversa-pt-fluent-v3.2
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/v1/models` | lista os modelos em `models/` com `codebook.bin` |
| GET | `/v1/models/<id>` | metadados de um modelo |
| POST | `/v1/chat/completions` | chat no formato OpenAI, com `stream` opcional |
| POST | `/v1/completions` | completion de texto |
| POST | `/v1/embeddings` | embeddings determinísticos de compatibilidade |
| GET | `/health` | status, modelos e modelo padrão |
| OPTIONS | `*` | CORS |

## Exemplos

Listar modelos:

```bash
curl http://127.0.0.1:8000/v1/models
```

Chat:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "conversa-pt-fluent-v3.2",
    "messages": [{"role": "user", "content": "fale sobre o brasil"}]
  }'
```

Streaming (SSE):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "conversa-pt-fluent-v3.2",
    "stream": true,
    "messages": [{"role": "user", "content": "olá"}]
  }'
```

Completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"conversa-pt-fluent-v3.2","prompt":"o que é o calor?"}'
```

Embedding:

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"conversa-pt-fluent-v3.2","input":"brasil"}'
```

Resposta de embedding:

```json
{"object":"list","data":[{"object":"embedding","index":0,"embedding":[...]}],
 "model":"conversa-pt-fluent-v3.2","usage":{"prompt_tokens":1,"total_tokens":1}}
```

`encoding_format: "base64"` também é aceito.

## Model routing

`/v1/models` descobre automaticamente diretórios e symlinks em `models/` que
contenham `codebook.bin`:

```text
models/conversa-pt-fluent-v3.2 -> /run/.../conversa-pt-fluent-v3.2
```

O campo `model` aceita:

- `conversa-pt-fluent-v3.2`;
- `cognicore/conversa-pt-fluent-v3.2`;
- um caminho absoluto para um diretório de modelo.

Modelo desconhecido retorna erro 404 no formato OpenAI.

## Variáveis de ambiente

| Variável | Padrão | Função |
|---|---|---|
| `COGNICORE_MODELS_DIR` | `models/` | diretório de modelos |
| `COGNICORE_CLI` | `build/cognicore` | binário do CLI |
| `COGNICORE_MODEL` | `v3.2` se existir | modelo padrão |
| `COGNICORE_REALIZER` | `fluent` | `strict`, `lexical` ou `fluent` |
| `COGNICORE_TIMEOUT` | `180` | timeout por completion, em segundos |
| `COGNICORE_MAX_CONCURRENCY` | `2` | completions simultâneos |
| `COGNICORE_EMBED_DIM` | `384` | dimensão do embedding de compatibilidade |
| `COGNICORE_API_KEY` | vazio | exige `Authorization: Bearer ...` |
| `COGNICORE_INCLUDE_SYSTEM` | `0` | inclui mensagens `system` no prompt |
| `COGNICORE_INCLUDE_HISTORY` | `0` | inclui histórico multi-turno |
| `COGNICORE_DEBUG` | `0` | liga `--debug` no CLI |

## Limitações importantes

- **Uma completion = uma carga de modelo.** O CLI one-shot carrega o modelo a
  cada request; na v3.2 (~553 MB) isso leva alguns segundos. Para uso intenso,
  o próximo passo é um modo servidor persistente no CLI.
- **Sem function calling/tools.** O servidor aceita o campo `tools` do cliente,
  mas o CogniCore não executa chamadas de ferramentas. Ferramentas agênticas
  precisam de uma camada externa de tool runner.
- **Streaming é SSE de um chunk.** O texto completo é enviado como um delta e
  depois `finish_reason: stop` + `[DONE]`; não é streaming token a token.
- **Embeddings são shim de compatibilidade.** São hashes determinísticos de
  tokens normalizados, não os vetores VSA do modelo. Servem para clientes que
  exigem o endpoint, não para busca semântica de alta qualidade.
- **System prompt/histórico são opt-in.** Por padrão usa-se a última mensagem
  do usuário, porque o chat CogniCore é single-turn e responde melhor assim.
- **API key é opcional.** Sem `COGNICORE_API_KEY`, qualquer chave é aceita.
- **Sem autenticação TLS.** Para expor fora da máquina, use um proxy reverso
  com HTTPS e uma chave real.
- **Modelos simbólicos.** A API não transforma o CogniCore em um LLM; ela
  apenas permite que ferramentas externas conversem com o modelo simbólico.

## Exemplo de configuração em clientes

OpenAI Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="conversa-pt-fluent-v3.2",
    messages=[{"role": "user", "content": "qual é a capital da França?"}],
)
print(response.choices[0].message.content)
```

OpenCode/Hermes/Continue: use o provedor OpenAI-compatible com:

```text
baseURL: http://127.0.0.1:8000/v1
apiKey: local
model: conversa-pt-fluent-v3.2
```

## Teste rápido

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

Se o `/health` mostrar `"neural": false`, o servidor está no motor simbólico
correto.
