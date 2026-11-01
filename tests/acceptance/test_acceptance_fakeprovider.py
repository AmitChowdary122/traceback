"""Self-test of the offline fake provider used by the CLI acceptance tests. Runs today.

If this fails, failures in test_acceptance_cli_agent.py say nothing about tbx.
"""
from __future__ import annotations

import json
import urllib.request

import pytest
from fakeprovider import FakeProvider, has_tool_results, text_reply, tool_reply

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy


def post(url: str, body: dict, key: str = "k") -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with OPENER.open(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def responder(req, n):
    if has_tool_results(req):
        return tool_reply(("submit_answer", {"claims": []}))
    return tool_reply(("get_calls", {"start": "2023-03-14T01:00:00Z"}), ("get_locations", {}))


def test_scripted_turns_and_recording():
    with FakeProvider(responder) as fp:
        s, b = post(f"{fp.base_url}/chat/completions", {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "q"}]}, key="canary")
        assert s == 200
        msg = json.loads(b)["choices"][0]["message"]
        assert [t["function"]["name"] for t in msg["tool_calls"]] == ["get_calls", "get_locations"]
        assert all(t["id"].startswith("call_") for t in msg["tool_calls"])  # Meta's real id style
        s, b = post(f"{fp.base_url}/chat/completions", {"model": "m", "messages": [{"role": "tool", "tool_call_id": "x", "content": "{}"}]})
        assert json.loads(json.loads(b)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"claims": []}
        assert len(fp.chat_requests) == 2
        assert fp.requests[0]["headers"]["Authorization"] == "Bearer canary"


def test_tool_choice_other_than_auto_is_400_like_meta():
    with FakeProvider(lambda r, n: text_reply("x")) as fp:
        s, b = post(f"{fp.base_url}/chat/completions", {"model": "m", "messages": [], "tool_choice": "required"})
        assert s == 400 and json.loads(b)["error"]["param"] == "tool_choice"
        s, _ = post(f"{fp.base_url}/chat/completions", {"model": "m", "messages": [], "tool_choice": "auto"})
        assert s == 200


def test_streaming_mode_emits_chunks_and_done():
    with FakeProvider(responder) as fp:
        s, b = post(f"{fp.base_url}/chat/completions", {"model": "m", "messages": [], "stream": True})
        lines = [ln for ln in b.decode().splitlines() if ln.startswith("data: ")]
        assert s == 200 and lines[-1] == "data: [DONE]"
        first = json.loads(lines[0][6:])
        assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_openai_sdk_compatibility_if_installed():
    openai = pytest.importorskip("openai")
    import httpx
    with FakeProvider(responder) as fp:
        client = openai.OpenAI(base_url=fp.base_url, api_key="canary", http_client=httpx.Client(trust_env=False))
        r = client.chat.completions.create(model="muse-spark-1.3", messages=[{"role": "user", "content": "q"}])
        assert r.choices[0].finish_reason == "tool_calls" and len(r.choices[0].message.tool_calls) == 2
        stream = client.chat.completions.create(model="m", messages=[{"role": "tool", "tool_call_id": "x", "content": "{}"}], stream=True)
        chunks = list(stream)
        assert chunks[0].choices[0].delta.tool_calls[0].function.name == "submit_answer"
