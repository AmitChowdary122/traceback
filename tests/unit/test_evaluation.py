"""tbx.evaluation offline: scoring, totals, the run loop with a fake client, the README table (CONTRACTS v1.2 §11)."""
import json
import pathlib
import subprocess
import sys

import pytest

from tbx import evaluation as ev
from tbx.llm import ChatResponse, ToolCall
from tbx.store import Store, artifact_id

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
LOC = artifact_id("locations", "loc.tsv", "1")
ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    s.set_meta("device_tz", NY)
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}"}
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-07-28T01:40:00Z", **prov, "app": "Phone", "direction": "outgoing",
                       "number": "+1 (919) 579-4674", "contact_name": "Josh Hickman", "duration_s": 121})
    s.insert("locations", {"artifact_id": LOC, "source_path": "loc.tsv", "source_row": "1",
                           "ts_utc": "2024-07-28T01:45:00Z", **prov, "source": "life360", "lat": 35.6596081,
                           "lon": -78.8730535, "accuracy_m": 5})
    return s


V = [{"text": "Josh Hickman was called at 9:40 PM.", "citations": [
        {"artifact_id": CALL, "field": "contact_name", "value": "Josh Hickman"},
        {"artifact_id": CALL, "field": "number", "value": "+1 (919) 579-4674"},
        {"artifact_id": CALL, "field": "ts_local", "value": "2024-07-27T21:40:00-04:00"}]},
     {"text": "The phone was at 35.6596, -78.8731.", "citations": [
        {"artifact_id": LOC, "field": "lat", "value": "35.6596081"},
        {"artifact_id": LOC, "field": "lon", "value": "-78.8730535"}]}]


@pytest.mark.parametrize("entity, found", [
    ("+19195794674", True), ("9195794674", True), ("+19195794675", False),      # phones as digits, phone rule
    ("Josh Hickman", True), ("josh", True), ("Alice", False),                     # text containment, folded
    ("35.6596081,-78.8730535", True), ("35.70,-78.87", False),                    # coordinates within 1e-4
    ("2024-07-27T21:40:00-04:00", True), ("2024-07-28 01:40:00Z", True),         # timestamps by instant
    ("2024-07-27T21:41:00-04:00", False),
])
def test_entity_found(store, entity, found):
    assert ev.entity_found(entity, V, store) is found


def test_score_question_and_totals(store):
    q1 = {"id": "q1", "question": "?", "expected_entities": ["+19195794674", "Alice"],
          "expected_artifact_ids": [CALL, "loc_0000000000"], "tags": ["calls"]}
    q2 = {"id": "q2", "question": "?", "tags": ["negative"]}
    q3 = {"id": "q3", "question": "?", "tags": ["negative"]}
    rej = [{"claim": {"text": "x", "citations": []}, "reasons": [{"code": "NO_CITATIONS", "detail": "", "citation_index": None}]}]
    s1 = ev.score_question(q1, V, rej, store)
    assert s1["fact_recall"] == 0.5 and s1["artifact_recall"] == 0.5 and s1["citation_precision"] == pytest.approx(2 / 3)
    assert s1["missing_entities"] == ["Alice"] and s1["missing_artifact_ids"] == ["loc_0000000000"]
    per_q = [{"id": "q1", "status": "ok", "verified": V, "rejected": rej, **s1, "latency_s": 10.0, "tokens": 100,
              "est_cost_usd": 0.01},
             {"id": "q2", "status": "ok", "verified": [], "rejected": [], **ev.score_question(q2, [], [], store),
              "latency_s": 2.0, "tokens": 50, "est_cost_usd": 0.005},
             {"id": "q3", "status": "ok", "verified": V[:1], "rejected": [], **ev.score_question(q3, V[:1], [], store),
              "latency_s": 4.0, "tokens": 50, "est_cost_usd": 0.005}]
    t = ev.totals(per_q, [q1, q2, q3])
    assert t["fact_recall"] == 0.5 and t["artifact_recall"] == 0.5
    assert t["citation_precision"] == 0.75 and t["uncited_in_report"] == 0
    assert t["rejected_by_reason"] == {"NO_CITATIONS": 1}
    assert t["abstention_accuracy"] == 0.5 and t["negative_questions"] == 2
    assert t["latency_p50_s"] == 4.0 and t["latency_p95_s"] == pytest.approx(9.4)
    assert t["tokens"] == 200 and t["est_cost_usd"] == 0.02
    for k in ("fact_recall", "artifact_recall", "citation_precision", "uncited_in_report", "rejected_by_reason",
              "abstention_accuracy", "latency_p50_s", "latency_p95_s", "tokens", "est_cost_usd"):
        assert k in t


def test_result_name():
    assert ev.result_name("baseten", "meta-models/muse-spark-open", 1) == "baseten-meta-models_muse-spark-open-repair.json"
    assert ev.result_name("meta", "muse-spark-1.3", 0) == "meta-muse-spark-1.3.json"


class FakeClient:
    provider, model = "fake", "fake-model"

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools):
        self.n += 1
        q = messages[1]["content"]
        args = {"claims": []} if "nothing" in q else {"claims": [V[0]]}
        return ChatResponse(content=None, reasoning=None, finish_reason="tool_calls", latency_ms=5,
                            usage={"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0},
                            tool_calls=[ToolCall(id="s", name="submit_answer", arguments=json.dumps(args))])


def test_run_eval_uses_the_agent_and_keeps_traces(store, tmp_path):
    qs = [{"id": "q1", "question": "Who was called?", "window": {"from": "2024-07-27 21:00", "to": "2024-07-27 23:00"},
           "expected_entities": ["Josh Hickman"], "expected_artifact_ids": [CALL]},
          {"id": "q2", "question": "nothing here?", "window": {"from": "2024-07-27 21:00", "to": "2024-07-27 23:00"},
           "tags": ["negative"]}]
    res = ev.run_eval(qs, store, FakeClient(), runs_dir=tmp_path / "runs")
    assert [q["id"] for q in res["questions"]] == ["q1", "q2"]
    for q in res["questions"]:
        assert (tmp_path / "runs" / f"{q['run_id']}.jsonl").is_file()
    t = res["totals"]
    assert t["fact_recall"] == 1.0 and t["artifact_recall"] == 1.0 and t["abstention_accuracy"] == 1.0
    assert t["uncited_in_report"] == 0 and t["tokens"] == 2200 and t["est_cost_usd"] > 0
    assert ev.window_of(qs[0], NY).as_dict() == {"start_utc": "2024-07-28T01:00:00Z", "end_utc": "2024-07-28T03:00:00Z"}


def test_report_renders_and_splices(tmp_path):
    res = {"provider": "meta", "model": "muse-spark-1.3", "repair": 0, "questions_file": "q.yaml",
           "totals": {"questions": 2, "errors": 0, "fact_recall": 0.5, "artifact_recall": 1.0, "citation_precision": 0.75,
                      "uncited_in_report": 0, "rejected_by_reason": {"UNGROUNDED_ENTITY:clock_time": 2},
                      "rejected_claims": 2, "abstention_accuracy": 1.0, "latency_p50_s": 12.0, "latency_p95_s": 20.5,
                      "tokens": 12345, "est_cost_usd": 0.0912}}
    table = ev.render_table([res])
    assert "| meta | `muse-spark-1.3` | no | 2 | 50% | 100% | 75% | 100% | 0 |" in table
    assert "UNGROUNDED_ENTITY:clock_time 2" in table and "$0.09" in table
    readme = "# T\n\nintro\n"
    once = ev.splice_readme(readme, table)
    twice = ev.splice_readme(once, table.replace("50%", "60%"))
    assert once.count(ev.REPORT_START) == 1 and twice.count(ev.REPORT_START) == 1 and "60%" in twice and "50%" not in twice
    results = tmp_path / "results"
    results.mkdir()
    (results / "meta-muse-spark-1.3.json").write_text(json.dumps(res))
    out = subprocess.run([sys.executable, str(ROOT / "eval" / "report.py"), "--results", str(results)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and "muse-spark-1.3" in out.stdout


def test_load_questions_rejects_duplicates(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- {id: q1, question: a}\n- {id: q1, question: b}\n")
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_questions(p)
