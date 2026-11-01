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
APP_MODULES = ("tbx.web", "tbx.api", "tbx.server", "tbx.app", "tbx.webapp")


@pytest.fixture
def client(tbx, case_db, monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("TBX_DB", str(case_db))
    for name in APP_MODULES:
        sys.modules.pop(name, None)
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        app = mod.create_app() if hasattr(mod, "create_app") else getattr(mod, "app", None)
        if app is not None:
            return testclient.TestClient(app)
    tbxcase.unavailable(f"no FastAPI app found in {APP_MODULES}")


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
    body = r.json()
    tags = body if isinstance(body, list) else body.get("rows", body.get("tags", []))
    assert any(t.get("label") == "car" and t.get("artifact_id") == IDS["TAG_CAR"] for t in tags), body


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
