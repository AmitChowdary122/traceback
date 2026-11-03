"""tbx.runview: recorded runs re-verified on load (re-check + entity spans for the UI)."""
import copy

import pytest

from tbx import runview
from tbx.checker import check
from tbx.models import Answer
from tbx.store import Store, artifact_id
from tbx.timeutil import Window, parse_utc

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
WINDOW = {"start_utc": "2024-07-28T01:00:00Z", "end_utc": "2024-07-28T03:00:00Z"}
GOOD = {"text": "Liz Dehner placed an outgoing call at 9:40 PM.",
        "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"},
                      {"artifact_id": CALL, "field": "ts_local", "value": "2024-07-27T21:40:00-04:00"}]}
BAD = {"text": "Bob called at 9:15 PM.", "citations": [{"artifact_id": CALL, "field": "direction", "value": "outgoing"}]}


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    s.set_meta("device_tz", NY)
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-07-28T01:40:00Z", "ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}",
                       "app": "Phone", "direction": "outgoing", "number": "9195794674", "contact_name": "Liz Dehner",
                       "duration_s": 121})
    s.insert("calls", {"artifact_id": artifact_id("calls", "calls.tsv", "2"), "source_path": "calls.tsv",
                       "source_row": "2", "ts_utc": "2024-07-28T02:00:00Z", "ts_original": "x", "tz_basis": "utc_string",
                       "raw_json": "{}", "app": "Phone", "direction": "incoming", "contact_name": "Bob"})
    return s


def trace(store, *rounds, window=WINDOW, legacy=False):
    """Events for a finished run: one (final_answer, check_result) pair per round, from the real checker."""
    w = Window(parse_utc(window["start_utc"]), parse_utc(window["end_utc"])) if window else None
    ev = [{"ts": "2026-11-03T10:00:00Z", "step": 0, "type": "run_start",
           "payload": {"question": "q", "window": window, "provider": "meta", "model": "muse-spark-1.3",
                       "repair": int(len(rounds) > 1), "case_tz": NY}}]
    for n, claims in enumerate(rounds, 1):
        answer = Answer.model_validate({"claims": claims})
        result = check(answer, store, w).model_dump()
        if legacy:
            result.pop("entities")       # a trace recorded before contract v1.3
        ev.append({"step": n, "type": "final_answer", "payload": {**answer.model_dump(), "round": n}})
        ev.append({"step": n, "type": "check_result", "payload": {**result, "round": n}})
    ev.append({"step": 9, "type": "run_end", "payload": {"status": "ok", "totals": {}}})
    return ev


def test_matching_run_gets_entities_aligned_with_its_lists(store):
    out = runview.reverify(trace(store, [BAD, GOOD]), store)
    assert out["recheck"]["status"] == "match", out["recheck"]
    ann = out["annotations"]
    assert [a["index"] for a in ann["verified"]] == [1] and [a["index"] for a in ann["rejected"]] == [0]
    assert [(e["text"], e["grounded"]) for e in ann["verified"][0]["entities"]] == [("Liz Dehner", True), ("9:40 PM", True)]
    assert [(e["text"], e["grounded"]) for e in ann["rejected"][0]["entities"]] == [("Bob", False), ("9:15 PM", False)]


def test_runs_recorded_before_v13_get_entities_from_the_recheck(store):
    out = runview.reverify(trace(store, [GOOD], legacy=True), store)
    assert out["recheck"]["status"] == "match"
    assert out["recheck"]["entities_recorded"] is False
    assert [e["type"] for e in out["annotations"]["verified"][0]["entities"]] == ["contact", "clock_time"]


def test_a_tampered_record_is_flagged(store):
    ev = trace(store, [BAD, GOOD])
    rec = next(e["payload"] for e in ev if e["type"] == "check_result")
    rec["verified"].insert(0, copy.deepcopy(rec["rejected"][0]["claim"]))   # claim 0 recorded as verified
    rec["rejected"] = []
    out = runview.reverify(ev, store)
    assert out["recheck"]["status"] == "differs"
    changes = out["recheck"]["rounds"][0]["changes"]
    assert any("Claim 1" in c and "verified" in c and "rejects" in c for c in changes), changes
    assert out["annotations"]["verified"][0]["index"] == 0      # still aligned with what the record shows


def test_changed_reasons_are_flagged(store):
    ev = trace(store, [BAD])
    rec = next(e["payload"] for e in ev if e["type"] == "check_result")
    rec["rejected"][0]["reasons"].pop()
    out = runview.reverify(ev, store)
    assert out["recheck"]["status"] == "differs"
    assert "reasons differ" in " ".join(out["recheck"]["rounds"][0]["changes"])


def test_the_window_is_the_recorded_one(store):
    narrow = {"start_utc": "2024-07-28T01:50:00Z", "end_utc": "2024-07-28T03:00:00Z"}
    out = runview.reverify(trace(store, [GOOD], window=narrow), store)
    assert out["recheck"]["status"] == "match"
    assert out["annotations"]["rejected"][0]["index"] == 0      # OUTSIDE_WINDOW under the recorded window


def test_repaired_run_rechecks_both_rounds(store):
    out = runview.reverify(trace(store, [BAD, GOOD], [GOOD]), store)
    assert out["recheck"]["status"] == "match"
    assert [r["round"] for r in out["recheck"]["rounds"]] == [1, 2]
    assert [a["index"] for a in out["annotations"]["verified"]] == [0]
    assert [a["index"] for a in out["first_round_annotations"]["rejected"]] == [0]


def test_running_or_unanswered_runs_are_not_rechecked(store):
    ev = trace(store, [GOOD])
    running = [e for e in ev if e["type"] == "run_start"]
    out = runview.reverify(running, store)
    assert out["recheck"]["status"] == "none"
    assert out["annotations"] == {"verified": [], "rejected": []}


def test_a_recheck_that_cannot_run_is_an_error_not_a_crash(store):
    ev = trace(store, [GOOD])
    next(e for e in ev if e["type"] == "final_answer")["payload"]["claims"] = "not a list"
    out = runview.reverify(ev, store)
    assert out["recheck"]["status"] == "error" and out["recheck"]["message"]


def test_a_different_store_is_a_difference(store, tmp_path):
    other = Store(str(tmp_path / "other.db"))
    other.set_meta("device_tz", NY)
    out = runview.reverify(trace(store, [GOOD]), other)
    assert out["recheck"]["status"] == "differs"
