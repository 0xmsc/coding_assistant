from __future__ import annotations

import argparse
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any


@dataclass(frozen=True)
class FakeOpenAIServer:
    base_url: str


class _ScriptedResponses:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self._index = 0
        self._lock = Lock()

    def next_response(self) -> dict[str, Any] | None:
        with self._lock:
            if self._index >= len(self._responses):
                return None
            response = self._responses[self._index]
            self._index += 1
            return response


class _FakeOpenAIHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]) -> None:
        super().__init__(server_address, handler_class)
        self.scripted_responses = _ScriptedResponses(_load_scripted_responses())


def _load_scripted_responses() -> list[dict[str, Any]]:
    script_file = os.environ.get("CODING_ASSISTANT_FAKE_OPENAI_RESPONSES_FILE")
    script_json = os.environ.get("CODING_ASSISTANT_FAKE_OPENAI_RESPONSES_JSON")
    if script_file is not None:
        raw = Path(script_file).read_text(encoding="utf-8")
    elif script_json is not None:
        raw = script_json
    else:
        return []

    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("Fake OpenAI responses must be a JSON array.")
    if not all(isinstance(item, dict) for item in decoded):
        raise ValueError("Every fake OpenAI response must be a JSON object.")
    return decoded


def _model_list_payload() -> dict[str, Any]:
    configured = os.environ.get("CODING_ASSISTANT_FAKE_OPENAI_MODELS_JSON")
    if configured is None:
        model_ids = ["fake-model"]
    else:
        decoded = json.loads(configured)
        if not isinstance(decoded, list) or not all(isinstance(model_id, str) for model_id in decoded):
            raise ValueError("Fake OpenAI models must be a JSON string array.")
        model_ids = decoded
    return {
        "object": "list",
        "data": [{"id": model_id, "object": "model"} for model_id in model_ids],
    }


def _messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def _response_text(payload: dict[str, Any]) -> str:
    configured = os.environ.get("CODING_ASSISTANT_FAKE_OPENAI_RESPONSE")
    if configured is not None:
        return configured
    messages = _messages_from_payload(payload)
    if messages:
        last_message = messages[-1]
        content = last_message.get("content")
        if isinstance(content, str) and content.strip():
            return f"fake response: {content}"
    return "fake response"


def _tool_result_response_text(payload: dict[str, Any]) -> str | None:
    messages = _messages_from_payload(payload)
    if not messages:
        return None
    last_message = messages[-1]
    if last_message.get("role") != "tool":
        return None
    content = last_message.get("content")
    if not isinstance(content, str):
        return "tool result"
    return f"tool result: {content.strip()}"


def _chunk_payload(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> str:
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    payload: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            },
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload)


def _scripted_tool_call_payload(tool_call: dict[str, Any], *, index: int) -> dict[str, Any]:
    identifier = tool_call.get("id")
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Scripted tool calls require a non-empty string id.")
    if not isinstance(name, str) or not name:
        raise ValueError("Scripted tool calls require a non-empty string name.")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {
        "index": index,
        "id": identifier,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _content_chunks(text: str) -> list[str]:
    if not text:
        return [""]
    midpoint = max(1, len(text) // 2)
    return [text[:midpoint], text[midpoint:]]


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "FakeOpenAI/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json({"ok": True})
            return
        if self.path == "/v1/models":
            self._write_json(_model_list_payload())
            return
        self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
        except ValueError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        if payload.get("stream") is not True:
            self._write_json({"error": "Only stream=true is supported."}, status=HTTPStatus.BAD_REQUEST)
            return

        response = self._next_scripted_response()
        if response is not None:
            try:
                self._write_sse_scripted_response(response)
            except ValueError as exc:
                self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if tool_result_text := _tool_result_response_text(payload):
            self._write_sse_response(tool_result_text)
            return

        self._write_sse_response(_response_text(payload))

    def _read_json(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Missing Content-Length.")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        decoded = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Request body must be a JSON object.")
        return decoded

    def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _next_scripted_response(self) -> dict[str, Any] | None:
        server = self.server
        if not isinstance(server, _FakeOpenAIHTTPServer):
            return None
        return server.scripted_responses.next_response()

    def _write_sse_response(self, text: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for chunk in _content_chunks(text):
            self._write_sse_event(_chunk_payload(content=chunk))
        self._write_sse_event(
            _chunk_payload(
                finish_reason="stop",
                usage={
                    "prompt_tokens": 1,
                    "completion_tokens": max(1, len(text.split())),
                    "total_tokens": max(2, len(text.split()) + 1),
                    "cost": 0.0,
                },
            ),
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse_scripted_response(self, response: dict[str, Any]) -> None:
        content = response.get("content")
        tool_calls = response.get("tool_calls")
        if isinstance(content, str):
            chunks = [_chunk_payload(content=chunk) for chunk in _content_chunks(content)]
            finish_reason = "stop"
            completion_tokens = max(1, len(content.split()))
        elif isinstance(tool_calls, list):
            if not all(isinstance(tool_call, dict) for tool_call in tool_calls):
                raise ValueError("Scripted tool_calls must be JSON objects.")
            chunks = [
                _chunk_payload(
                    tool_calls=[
                        _scripted_tool_call_payload(tool_call, index=index)
                        for index, tool_call in enumerate(tool_calls)
                    ],
                ),
            ]
            finish_reason = "tool_calls"
            completion_tokens = 1
        else:
            raise ValueError("Scripted fake OpenAI response requires content or tool_calls.")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        for chunk in chunks:
            self._write_sse_event(chunk)
        self._write_sse_event(
            _chunk_payload(finish_reason=finish_reason, usage=_usage(completion_tokens=completion_tokens)),
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse_event(self, payload: str) -> None:
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


def _usage(*, completion_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": 1,
        "completion_tokens": completion_tokens,
        "total_tokens": completion_tokens + 1,
        "cost": 0.0,
    }


@contextmanager
def run_fake_openai_server(*, host: str = "127.0.0.1", port: int = 0) -> Iterator[FakeOpenAIServer]:
    server = _FakeOpenAIHTTPServer((host, port), _FakeOpenAIHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield FakeOpenAIServer(base_url=f"http://{host}:{server.server_port}/v1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description="Fake OpenAI-compatible server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = _FakeOpenAIHTTPServer((args.host, args.port), _FakeOpenAIHandler)
    print(f"Fake OpenAI endpoint: http://{args.host}:{server.server_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
