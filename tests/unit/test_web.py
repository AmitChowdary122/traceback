"""tbx.web offline: background asks, replay, rate limit, media files (CONTRACTS v1.2 section 10)."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from tbx import web
from tbx.llm import ChatResponse, ConfigError, ToolCall
from tbx.store import Store, artifact_id

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
MED = artifact_id("media", "DCIM/a.png", "1")
TAG = artifact_id("image_tags", "DCIM/a.png", f"{MED}|sam-3.1|car|0")
ANSWER = {"claims": [{"text": "Liz Dehner placed an outgoing call.",
                      "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"}]},
                     {"text": "Bob called.", "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Bob"}]}]}


def _turn(name, args):
    return ChatResponse(content=None, reasoning=None, finish_reason="tool_calls", latency_ms=3,
                        usage={"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 0},
                        tool_calls=[ToolCall(id="c1", name=name, arguments=json.dumps(args))])


class FakeClient:
    def __init__(self, provider, model):
        if provider == "nope":
            raise ConfigError("unknown provider 'nope'")
        self.provider, self.model = provider or "meta", model or "muse-spark-1.3"
        self.script = [_turn("get_calls", {}), _turn("submit_answer", ANSWER)]

    def chat(self, messages, tools):
        time.sleep(0.05)
        return self.script.pop(0)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from PIL import Image
    media = tmp_path / "media"
    (media / "DCIM").mkdir(parents=True)
    Image.new("RGB", (100, 50), "white").save(media / "DCIM" / "a.png")
    (tmp_path / "secret.txt").write_text("outside the media root")
    db = tmp_path / "c.db"
    s = Store(str(db))
    s.set_meta("device_tz", NY)
    s.set_meta("media_root", str(media))
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": json.dumps({"Partner": "9195794674"})}
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-07-28T01:40:00Z", **prov, "app": "Phone", "direction": "outgoing",
                       "number": "9195794674", "contact_name": "Liz Dehner", "duration_s": 121})
    s.insert("media", {"artifact_id": MED, "source_path": "DCIM/a.png", "source_row": "1", "ts_utc": None, **prov,
                       "app": "Camera", "kind": "image", "file_path": "DCIM/a.png", "mime_type": "image/png",
                       "sha256": "0" * 64, "size_bytes": 1})
    s.insert("image_tags", {"artifact_id": TAG, "source_path": "DCIM/a.png", "source_row": f"{MED}|sam-3.1|car|0",
                            "ts_utc": None, **prov, "media_id": MED, "model": "sam-3.1", "label": "car", "score": 0.9,
                            "box": "[1, 2, 30, 40]", "prompt_set": "v1"})
    s.close()
    monkeypatch.setenv("TBX_ASK_LIMIT_PER_HOUR", "2")
    app = web.create_app(str(db), runs_dir=str(tmp_path / "runs"), client_factory=FakeClient)
    return TestClient(app), tmp_path


def _wait(client, run_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] != "running":
            return r
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_ask_runs_in_the_background_and_replays(setup):
    client, tmp = setup
    r = client.post("/api/ask", json={"question": "Who called?", "from": "2024-07-27 21:00", "to": "2024-07-27 23:00"})
    assert r.status_code == 200 and set(r.json()) == {"run_id"}
    run = _wait(client, r.json()["run_id"])
    assert run["status"] == "ok" and [c["text"] for c in run["verified"]] == ["Liz Dehner placed an outgoing call."]
    assert run["rejected"][0]["reasons"][0]["code"] == "VALUE_MISMATCH"
    assert run["window"] == {"start_utc": "2024-07-28T01:00:00Z", "end_utc": "2024-07-28T03:00:00Z"}
    assert run["artifacts"][CALL]["ts_local"] == "2024-07-27T21:40:00-04:00"
    assert [e["type"] for e in run["events"]][0] == "run_start" and run["events"][-1]["type"] == "run_end"
    listing = client.get("/api/runs").json()
    assert listing[0]["run_id"] == run["run_id"] and listing[0]["verified"] == 1 and listing[0]["rejected"] == 1
    assert (tmp / "runs" / f"{run['run_id']}.jsonl").is_file()


def test_ask_validation_and_rate_limit(setup):
    client, _ = setup
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "q", "from": "last tuesday"}).status_code == 400
    assert client.post("/api/ask", json={"question": "q", "provider": "nope"}).status_code == 400
    ids = [client.post("/api/ask", json={"question": f"q{i}"}).json()["run_id"] for i in range(2)]
    over = client.post("/api/ask", json={"question": "q3"})
    assert over.status_code == 429 and "Retry-After" in over.headers
    for i in ids:
        _wait(client, i)


def test_runs_listing_status_and_bad_ids(setup):
    client, tmp = setup
    runs = tmp / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "old-crashed.jsonl").write_text(json.dumps({"ts": "2026-01-01T00:00:00Z", "run_id": "old-crashed", "step": 0,
                                                        "type": "run_start", "payload": {"question": "q"}}) + "\n")
    import os
    os.utime(runs / "old-crashed.jsonl", (time.time() - 3600, time.time() - 3600))
    run = client.get("/api/runs/old-crashed").json()
    assert run["status"] == "error" and "run_end" in run["error"]
    assert client.get("/api/runs/..%2Fsecret").status_code == 404
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_media_file_tags_and_containment(setup):
    client, tmp = setup
    r = client.get(f"/api/media/{MED}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    tags = client.get(f"/api/media/{MED}/tags").json()["rows"]
    assert tags[0]["artifact_id"] == TAG and tags[0]["box"] == [1, 2, 30, 40]
    assert client.get(f"/api/media/{CALL}").status_code == 404           # not a media row
    assert client.get(f"/api/media/{CALL}/tags").status_code == 404
    assert client.get(f"/api/media/{MED}/transcripts").json() == {"rows": []}
    s = Store(str(tmp / "c.db"))
    s.conn.execute("UPDATE media SET file_path = '../secret.txt' WHERE artifact_id = ?", (MED,))
    s.conn.commit()
    s.close()
    assert client.get(f"/api/media/{MED}").status_code == 404           # never serves outside the media root


def test_artifact_and_case_endpoints_never_expose_keys(setup, monkeypatch):
    client, _ = setup
    monkeypatch.setenv("META_API_KEY", "tbx-canary-key-123456")
    row = client.get(f"/api/artifacts/{TAG}").json()
    assert row["raw_json"] and row["box_array"] == [1, 2, 30, 40]
    case = client.get("/api/case").json()
    assert case["providers"]["meta"] is True and case["counts"]["image_tags"] == 1
    assert "tbx-canary-key" not in json.dumps(case) and "media_root" not in case["meta"]
    page = client.get("/")
    assert page.status_code == 200 and "Recorded runs" in page.text and "tbx-canary-key" not in page.text


def test_rate_limit_zero_turns_asking_off():
    rl = web.RateLimit(0)
    assert rl.take()[0] is False
    rl = web.RateLimit(1)
    assert rl.take()[0] is True and rl.take()[0] is False and rl.remaining() == 0
