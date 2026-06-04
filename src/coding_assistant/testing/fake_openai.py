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
from threading import Thread
from typing import Any


@dataclass(frozen=True)
class FakeOpenAIServer:
    base_url: str


def _response_text(payload: dict[str, Any]) -> str:
    configured = os.environ.get("FAKE_OPENAI_RESPONSE")
    if configured is not None:
        return configured
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content")
            if isinstance(content, str) and content.strip():
                return f"fake response: {content}"
    return "fake response"


def _chunk_payload(*, content: str = "", finish_reason: str | None = None, usage: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            },
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return json.dumps(payload)


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

    def _write_sse_event(self, payload: str) -> None:
        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()


@contextmanager
def run_fake_openai_server(*, host: str = "127.0.0.1", port: int = 0) -> Iterator[FakeOpenAIServer]:
    server = ThreadingHTTPServer((host, port), _FakeOpenAIHandler)
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
    server = ThreadingHTTPServer((args.host, args.port), _FakeOpenAIHandler)
    print(f"Fake OpenAI endpoint: http://{args.host}:{server.server_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
