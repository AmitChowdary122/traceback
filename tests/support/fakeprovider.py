"""A scripted, OpenAI-compatible chat-completions server on 127.0.0.1 for offline acceptance tests.

The shapes it returns are copied from the real responses recorded in tests/fixtures/spikes/
(muse-spark-1.3 on Meta and muse-spark-open on Baseten), so `tbx ask` can be driven end to end
with no network and no credits. Every request is recorded for assertions.

    with FakeProvider(responder) as fp:
        env["META_BASE_URL"] = fp.base_url      # http://127.0.0.1:<port>/v1
        ...
        fp.requests  # [{"path", "headers", "json"}]

`responder(request_json, n)` returns either a chat.completion body (dict) or (status, body).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

Responder = Callable[[dict, int], Any]


def completion(message: dict, finish_reason: str, model: str = "muse-spark-1.3",
               usage: dict | None = None) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "refusal": None, **message},
                     "finish_reason": finish_reason, "logprobs": None}],
        "usage": usage or {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                           "completion_tokens_details": {"reasoning_tokens": 100},
                           "prompt_tokens_details": {"cached_tokens": 0}},
    }


def tool_calls_msg(*calls: tuple[str, dict], content: str | None = None, id_style: str = "meta") -> dict:
    def tc_id() -> str:
        return f"call_{uuid.uuid4()}" if id_style == "meta" else f"chatcmpl-tool-{uuid.uuid4().hex}"
    return {"content": content, "tool_calls": [
        {"id": tc_id(), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for name, args in calls]}


def tool_reply(*calls: tuple[str, dict], content: str | None = None, model: str = "muse-spark-1.3", **extra) -> dict:
    body = completion(tool_calls_msg(*calls, content=content, id_style="meta" if "/" not in model else "baseten"),
                      "tool_calls", model=model)
    body["choices"][0]["message"].update(extra)
    return body


def text_reply(text: str, model: str = "muse-spark-1.3") -> dict:
    return completion({"content": text}, "stop", model=model)


def has_tool_results(req: dict) -> bool:
    return any(m.get("role") == "tool" for m in req.get("messages", []))


def tool_messages(req: dict) -> list[dict]:
    return [m for m in req.get("messages", []) if m.get("role") == "tool"]


def _sse(body: dict) -> bytes:
    """Re-encode a chat.completion as a minimal chat.completion.chunk stream."""
    ch = body["choices"][0]
    msg = ch["message"]
    delta: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(msg["tool_calls"])]
    base = {k: body[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    chunks = [dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]),
              dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": ch["finish_reason"]}], usage=body.get("usage"))]
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks) + b"data: [DONE]\n\n"


class FakeProvider:
    def __init__(self, responder: Responder, models: list[str] | None = None):
        self.responder = responder
        self.models = models or ["muse-spark-1.3", "meta-models/muse-spark-open"]
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        fp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _send(self, status: int, body: Any, sse: bool = False):
                data = _sse(body) if sse else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with fp._lock:
                    fp.requests.append({"path": self.path, "method": "GET", "headers": dict(self.headers), "json": None})
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "meta"} for m in fp.models]})
                else:
                    self._send(404, {"error": {"message": f"no route {self.path}"}})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    req = json.loads(raw or b"{}")
                except ValueError:
                    req = {"_unparseable": raw.decode("utf-8", "replace")}
                with fp._lock:
                    fp.requests.append({"path": self.path, "method": "POST", "headers": dict(self.headers), "json": req})
                    n = sum(1 for r in fp.requests if r["path"].endswith("/chat/completions"))
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": f"fake provider only serves /chat/completions, got {self.path}"}})
                out = fp.responder(req, n)
                status, body = out if isinstance(out, tuple) else (200, out)
                if status == 200 and req.get("tool_choice") not in (None, "auto"):
                    # Real Meta behaviour (spike S1): anything but "auto" is a 400.
                    status, body = 400, {"error": {"code": None, "param": "tool_choice", "type": "invalid_request_error",
                                                   "message": "only `\"auto\"` is supported for `tool_choice`."}}
                self._send(status, body, sse=bool(req.get("stream")) and status == 200)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def chat_requests(self) -> list[dict]:
        return [r["json"] for r in self.requests if r["path"].endswith("/chat/completions")]

    def __enter__(self) -> "FakeProvider":
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def closed_port() -> int:
    """A localhost port with nothing listening (for the exit-code-3 test)."""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
