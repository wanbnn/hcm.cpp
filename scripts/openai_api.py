#!/usr/bin/env python3
"""OpenAI-compatible HTTP API for CogniCore models.

Endpoints:
  GET  /v1/models
  GET  /v1/models/<id>
  POST /v1/chat/completions
  POST /v1/completions
  POST /v1/embeddings
  GET  /health
  OPTIONS *

The server is a compatibility bridge, not a neural inference engine. Each
completion invokes the local CogniCore CLI in one-shot mode, so the symbolic
model answers with its graph/discourse pipeline. The ``model`` field selects a
directory under ``models/`` (or an absolute model directory).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path(os.environ.get("COGNICORE_MODELS_DIR", ROOT / "models"))
DEFAULT_CLI = Path(os.environ.get("COGNICORE_CLI", ROOT / "build" / "cognicore"))
DEFAULT_REALIZER = os.environ.get("COGNICORE_REALIZER", "fluent")
DEFAULT_TIMEOUT = float(os.environ.get("COGNICORE_TIMEOUT", "180"))
MAX_CONCURRENCY = max(1, int(os.environ.get("COGNICORE_MAX_CONCURRENCY", "2")))
EMBED_DIM = max(8, int(os.environ.get("COGNICORE_EMBED_DIM", "384")))
API_KEY = os.environ.get("COGNICORE_API_KEY", "")
DEBUG = os.environ.get("COGNICORE_DEBUG", "") in {"1", "true", "yes"}
INCLUDE_SYSTEM = os.environ.get("COGNICORE_INCLUDE_SYSTEM", "") in {"1", "true", "yes"}
INCLUDE_HISTORY = os.environ.get("COGNICORE_INCLUDE_HISTORY", "") in {"1", "true", "yes"}

SEMAPHORE = threading.Semaphore(MAX_CONCURRENCY)
START = int(time.time())


class APIError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.param = param
        self.code = code


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def discover_models(models_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not models_dir.is_dir():
        return found
    for entry in sorted(models_dir.iterdir(), key=lambda p: p.name):
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        if not (entry / "codebook.bin").is_file():
            continue
        found[entry.name] = entry.resolve()
    return found


def resolve_model(requested: str, models_dir: Path, default_model: str | None) -> tuple[str, Path]:
    models = discover_models(models_dir)
    name = (requested or default_model or "").strip()
    if name.startswith("cognicore/") or name.startswith("openai/"):
        name = name.split("/", 1)[1]
    if not name:
        if not models:
            raise APIError(503, "Nenhum modelo CogniCore encontrado em " + str(models_dir),
                           "server_error", code="model_unavailable")
        name = next(iter(models))
    if name in models:
        return name, models[name]
    candidate = Path(name).expanduser()
    if candidate.is_dir() and (candidate / "codebook.bin").is_file():
        return candidate.name or name, candidate.resolve()
    raise APIError(404, f"Modelo não encontrado: {requested}", "invalid_request_error",
                   param="model", code="model_not_found")


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def build_prompt(messages: list[dict[str, Any]]) -> str:
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        text = content_text(message.get("content"))
        if not text:
            continue
        if role == "system" or role == "developer":
            system_parts.append(text)
        else:
            turns.append((role, text))
    if not turns:
        return "\n".join(system_parts)
    if INCLUDE_HISTORY:
        lines = []
        if system_parts and INCLUDE_SYSTEM:
            lines.append("Instruções: " + "\n".join(system_parts))
        for role, text in turns:
            prefix = "Assistente" if role == "assistant" else "Usuário"
            lines.append(f"{prefix}: {text}")
        lines.append("Usuário:")
        return "\n".join(lines)
    prompt = turns[-1][1]
    if system_parts and INCLUDE_SYSTEM:
        prompt = "Instruções: " + "\n".join(system_parts) + "\n\n" + prompt
    return prompt


def run_model(model_path: Path, prompt: str, realizer: str, timeout: float) -> tuple[str, float]:
    cli = Path(os.environ.get("COGNICORE_CLI", DEFAULT_CLI)).expanduser()
    if not cli.is_file():
        raise APIError(500, f"CLI CogniCore não encontrado: {cli}", "server_error",
                       code="cli_not_found")
    command = [str(cli), "chat", "--model", str(model_path), "--realizer", realizer,
               "--prompt", prompt, "--no-save"]
    if DEBUG:
        command.append("--debug")
    start = time.monotonic()
    with SEMAPHORE:
        try:
            result = subprocess.run(command, text=True, capture_output=True,
                                    timeout=timeout, cwd=str(ROOT))
        except subprocess.TimeoutExpired as error:
            raise APIError(504, f"Tempo esgotado após {timeout:.0f}s", "server_error",
                           code="timeout") from error
    elapsed = time.monotonic() - start
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise APIError(500, "Falha no CogniCore: " + (detail[-2000:] or "erro desconhecido"),
                       "server_error", code="cognicore_error")
    text = result.stdout.strip()
    # If a caller enables --debug, the CLI writes debug lines before the answer.
    if DEBUG and "\n" in text:
        lines = text.splitlines()
        last = 0
        for i, line in enumerate(lines):
            if line.startswith("[PLANO ANTES DO REALIZADOR]"):
                last = i + 1
        text = "\n".join(lines[last:]).strip()
    return text, elapsed


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def embedding_for(text: str, dimension: int = EMBED_DIM) -> list[float]:
    vector = [0.0] * dimension
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = sum(v * v for v in vector) ** 0.5
    if norm:
        vector = [v / norm for v in vector]
    return vector


def completion_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class Handler(BaseHTTPRequestHandler):
    server_version = "CogniCoreOpenAI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _auth(self) -> None:
        if not API_KEY:
            return
        header = self.headers.get("Authorization", "")
        key = self.headers.get("X-API-Key", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else key
        if supplied != API_KEY:
            raise APIError(401, "Chave de API inválida", "authentication_error",
                           code="invalid_api_key")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise APIError(400, "Corpo JSON ausente", "invalid_request_error")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise APIError(400, "JSON inválido: " + str(error), "invalid_request_error") from error
        if not isinstance(value, dict):
            raise APIError(400, "O corpo deve ser um objeto JSON", "invalid_request_error")
        return value

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json_dumps(value).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, error: APIError) -> None:
        self._json(error.status, {"error": {"message": error.message, "type": error.type,
                                           "param": error.param, "code": error.code}})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._auth()
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path in ("/", "/health"):
                self._json(200, {"status": "ok", "models": list(MODELS),
                                 "default_model": DEFAULT_MODEL, "engine": "cognicore",
                                 "neural": False, "uptime": int(time.time()) - START})
                return
            if path == "/v1/models":
                self._json(200, {"object": "list", "data": [model_record(name, path)
                                                           for name, path in MODELS.items()]})
                return
            if path.startswith("/v1/models/"):
                name = unquote(path[len("/v1/models/"):])
                if name not in MODELS:
                    raise APIError(404, f"Modelo não encontrado: {name}", "invalid_request_error",
                                   param="model", code="model_not_found")
                self._json(200, model_record(name, MODELS[name]))
                return
            raise APIError(404, f"Rota não encontrada: {path}", "invalid_request_error",
                           code="not_found")
        except APIError as error:
            self._error(error)
        except Exception as error:  # noqa: BLE001
            self._error(APIError(500, str(error), "server_error"))

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._auth()
            path = urlparse(self.path).path.rstrip("/")
            payload = self._read_json()
            if path == "/v1/chat/completions":
                self._chat_completions(payload)
            elif path == "/v1/completions":
                self._completions(payload)
            elif path == "/v1/embeddings":
                self._embeddings(payload)
            else:
                raise APIError(404, f"Rota não encontrada: {path}", "invalid_request_error",
                               code="not_found")
        except APIError as error:
            self._error(error)
        except Exception as error:  # noqa: BLE001
            self._error(APIError(500, str(error), "server_error"))

    def _chat_completions(self, payload: dict[str, Any]) -> None:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise APIError(400, "messages deve ser uma lista não vazia", "invalid_request_error",
                           param="messages")
        prompt = build_prompt(messages)
        if not prompt.strip():
            raise APIError(400, "Nenhum texto encontrado nas mensagens", "invalid_request_error",
                           param="messages")
        name, path = resolve_model(str(payload.get("model", "")), MODELS_DIR, DEFAULT_MODEL)
        realizer = str(payload.get("cognicore_realizer") or payload.get("realizer") or DEFAULT_REALIZER)
        text, elapsed = run_model(path, prompt, realizer, DEFAULT_TIMEOUT)
        created = int(time.time())
        usage = {"prompt_tokens": estimate_tokens(prompt),
                 "completion_tokens": estimate_tokens(text),
                 "total_tokens": estimate_tokens(prompt) + estimate_tokens(text)}
        completion = {
            "id": completion_id("chatcmpl"), "object": "chat.completion",
            "created": created, "model": name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": usage,
            "cognicore": {"elapsed_seconds": round(elapsed, 3), "realizer": realizer,
                          "neural": False},
        }
        if payload.get("stream"):
            self._stream_chat(completion, text, name, created)
        else:
            self._json(200, completion)

    def _stream_chat(self, completion: dict[str, Any], text: str, model: str,
                     created: int) -> None:
        base = {"id": completion["id"], "object": "chat.completion.chunk",
                "created": created, "model": model}
        first = dict(base)
        first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": text},
                             "finish_reason": None}]
        last = dict(base)
        last["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        body = ("data: " + json_dumps(first) + "\n\n" +
                "data: " + json_dumps(last) + "\n\n" +
                "data: [DONE]\n\n").encode("utf-8")
        self._send(200, body, "text/event-stream; charset=utf-8")

    def _completions(self, payload: dict[str, Any]) -> None:
        prompt_value = payload.get("prompt", "")
        if isinstance(prompt_value, list):
            prompt = "\n".join(content_text(item) for item in prompt_value)
        else:
            prompt = content_text(prompt_value)
        if not prompt.strip():
            raise APIError(400, "prompt vazio", "invalid_request_error", param="prompt")
        name, path = resolve_model(str(payload.get("model", "")), MODELS_DIR, DEFAULT_MODEL)
        realizer = str(payload.get("cognicore_realizer") or DEFAULT_REALIZER)
        text, elapsed = run_model(path, prompt, realizer, DEFAULT_TIMEOUT)
        self._json(200, {
            "id": completion_id("cmpl"), "object": "text_completion",
            "created": int(time.time()), "model": name,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": estimate_tokens(prompt),
                      "completion_tokens": estimate_tokens(text),
                      "total_tokens": estimate_tokens(prompt) + estimate_tokens(text)},
            "cognicore": {"elapsed_seconds": round(elapsed, 3), "realizer": realizer,
                          "neural": False},
        })

    def _embeddings(self, payload: dict[str, Any]) -> None:
        input_value = payload.get("input", "")
        inputs = input_value if isinstance(input_value, list) else [input_value]
        if not inputs:
            raise APIError(400, "input vazio", "invalid_request_error", param="input")
        name, _ = resolve_model(str(payload.get("model", "")), MODELS_DIR, DEFAULT_MODEL)
        dimension = int(payload.get("dimensions") or EMBED_DIM)
        dimension = max(8, min(dimension, 4096))
        encoding = str(payload.get("encoding_format", "float"))
        data = []
        tokens = 0
        for index, value in enumerate(inputs):
            text = content_text(value)
            tokens += estimate_tokens(text)
            vector = embedding_for(text, dimension)
            if encoding == "base64":
                embedding = base64.b64encode(struct.pack("<" + "f" * dimension, *vector)).decode("ascii")
            else:
                embedding = vector
            data.append({"object": "embedding", "embedding": embedding, "index": index})
        self._json(200, {"object": "list", "data": data, "model": name,
                         "usage": {"prompt_tokens": tokens, "total_tokens": tokens}})


def model_record(name: str, path: Path) -> dict[str, Any]:
    try:
        created = int(path.stat().st_mtime)
    except OSError:
        created = START
    return {"id": name, "object": "model", "created": created, "owned_by": "cognicore",
            "permission": [], "root": name, "parent": None}


def main() -> int:
    global MODELS_DIR, MODELS, DEFAULT_MODEL
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("COGNICORE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("COGNICORE_PORT", "8000")))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--default-model", default=os.environ.get("COGNICORE_MODEL", ""))
    parser.add_argument("--realizer", default=DEFAULT_REALIZER)
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()
    MODELS_DIR = args.models_dir.expanduser().resolve()
    MODELS = discover_models(MODELS_DIR)
    DEFAULT_MODEL = args.default_model
    if not DEFAULT_MODEL and MODELS:
        preferred = [n for n in MODELS if "v3.2" in n] or list(MODELS)
        DEFAULT_MODEL = preferred[0]
    if args.api_key:
        os.environ["COGNICORE_API_KEY"] = args.api_key
        globals()["API_KEY"] = args.api_key
    if args.realizer:
        os.environ["COGNICORE_REALIZER"] = args.realizer
        globals()["DEFAULT_REALIZER"] = args.realizer
    if not MODELS:
        print(f"[api] nenhum modelo em {MODELS_DIR}", file=sys.stderr)
    else:
        print(f"[api] modelos: {', '.join(MODELS)}", file=sys.stderr)
    print(f"[api] default={DEFAULT_MODEL} cli={os.environ.get('COGNICORE_CLI', DEFAULT_CLI)}",
          file=sys.stderr)
    print(f"[api] escutando em http://{args.host}:{args.port}/v1", file=sys.stderr)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
