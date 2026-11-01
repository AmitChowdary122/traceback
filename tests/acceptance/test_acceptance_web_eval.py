"""Acceptance (P1): web API (CONTRACTS.md section 10) and eval artifacts (section 11).
Each test skips until its piece exists."""
from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import datetime

import pytest
import tbxcase
from tbxcase import IDS

REPO = tbxcase.REPO


@pytest.fixture
def client(tbx, case_db, monkeypatch, tmp_path):
    """CONTRACTS v1.2 section 10: tbx.web.create_app(db_path); runs are read from ./runs/*.jsonl."""
    testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("TBX_DB", str(case_db))
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("tbx.web", None)
    try:
        mod = importlib.import_module("tbx.web")
    except ImportError as e:
        tbxcase.unavailable(f"tbx.web not importable yet ({e})")
    if not hasattr(mod, "create_app"):
        pytest.fail("tbx.web must expose create_app(db_path=None) (CONTRACTS v1.2 section 10)")
    return testclient.TestClient(mod.create_app(str(case_db)))


@pytest.mark.p1
def test_index_is_html(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


@pytest.mark.p1
def test_artifact_endpoint_returns_full_row(client):
    r = client.get(f"/api/artifacts/{IDS['CALL_ALICE']}")
    assert r.status_code == 200
    row = r.json()
    assert {"raw_json", "source_path", "source_row"} <= set(row)
    assert row["source_path"] == tbxcase.CALLS_SRC


@pytest.mark.p1
def test_unknown_artifact_is_404(client):
    assert client.get("/api/artifacts/call_0000000000").status_code == 404


@pytest.mark.p1
def test_media_tags_endpoint(client):
    r = client.get(f"/api/media/{IDS['MED_PHOTO']}/tags")
    assert r.status_code == 200
    rows = r.json()["rows"]
    car = [t for t in rows if t.get("artifact_id") == IDS["TAG_CAR"]]
    assert car and car[0]["label"] == "car" and car[0]["box"] == [420, 300, 520, 400], rows


@pytest.mark.p1
def test_recorded_runs_are_replayable(client, tmp_path):
    """v1.2: GET /api/runs and /api/runs/{id} serve runs/*.jsonl with no model call."""
    run_id = "20231114T020000Z-replay"
    claim = {"text": "Alice placed a call.", "time_start": None, "time_end": None,
             "citations": [{"artifact_id": IDS["CALL_ALICE"], "field": "contact_name", "value": "Alice"}]}
    events = [
        ("run_start", {"question": "Who called?", "window": None, "provider": "meta", "model": "muse-spark-1.3", "repair": 0}),
        ("final_answer", {"claims": [claim], "note": None}),
        ("check_result", {"verified": [claim], "rejected": []}),
        ("run_end", {"status": "ok", "totals": {"tokens": 10, "latency_ms": 5, "est_cost_usd": 0.0}}),
    ]
    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "runs" / f"{run_id}.jsonl").write_text("".join(
        json.dumps({"ts": "2026-11-01T00:00:00Z", "run_id": run_id, "step": i, "type": t, "payload": p}) + "\n"
        for i, (t, p) in enumerate(events)))
    listing = client.get("/api/runs")
    assert listing.status_code == 200
    assert any(r["run_id"] == run_id and r["question"] == "Who called?" and r["status"] == "ok" for r in listing.json())
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "ok" and [c["text"] for c in run["verified"]] == ["Alice placed a call."]
    assert [e["type"] for e in run["events"]][:1] == ["run_start"]
    assert client.get("/api/runs/does-not-exist").status_code == 404


# ------------------------------------------------------------------------------- eval
def load_questions():
    path = REPO / "eval" / "questions.yaml"
    if not path.exists():
        pytest.skip("eval/questions.yaml not written yet")
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text())


@pytest.mark.p1
def test_eval_questions_schema():
    qs = load_questions()
    assert 20 <= len(qs) <= 25, len(qs)
    assert len({q["id"] for q in qs}) == len(qs)
    negatives = [q for q in qs if "negative" in q.get("tags", [])]
    assert 3 <= len(negatives) <= 4, len(negatives)
    for q in qs:
        assert isinstance(q["question"], str) and q["question"].strip()
        for k in ("from", "to"):
            datetime.strptime(q["window"][k], "%Y-%m-%d %H:%M")
        if q in negatives:
            assert not q.get("expected_entities") and not q.get("expected_artifact_ids"), q["id"]
        for aid in q.get("expected_artifact_ids", []):
            assert re.fullmatch(r"(msg|call|loc|med|trn|tag)_[0-9a-f]{10}", aid), (q["id"], aid)


@pytest.mark.p1
def test_eval_results_have_zero_uncited_claims():
    files = sorted((REPO / "eval" / "results").glob("*.json"))
    if not files:
        pytest.skip("no eval/results/*.json yet")
    for f in files:
        res = json.loads(f.read_text())
        totals = res.get("totals", res)
        assert totals.get("uncited_in_report") == 0, f.name
        for k in ("fact_recall", "artifact_recall", "citation_precision", "rejected_by_reason", "abstention_accuracy",
                  "latency_p50_s", "latency_p95_s", "tokens", "est_cost_usd"):
            assert k in totals, (f.name, k)
