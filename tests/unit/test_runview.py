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
        if n < len(rounds):   # as tbx.agent writes it: the rejected claims go back once
            ev.append({"step": n, "type": "repair", "payload": {"rejected": result["rejected"]}})
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


# ------------------------------------------------------------------------------ U1: sentences and the repair view
FIXED = {"text": "Liz Dehner placed an outgoing call at 9:40 PM, as the call log shows.",
         "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"},
                       {"artifact_id": CALL, "field": "ts_local", "value": "2024-07-27T21:40:00-04:00"}]}
WRONG_TIME = {"text": "Liz Dehner placed an outgoing call at 9:15 PM.",
              "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"}]}
GHOST = {"text": "A message arrived at 10:00 PM.", "citations": [{"artifact_id": "msg_ffffffffff", "field": "body", "value": "hi"}]}
BOB = {"text": "Bob called.", "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Bob"}]}
BOB_AGAIN = {"text": "Bob called.", "citations": [{"artifact_id": CALL, "field": "direction", "value": "incoming"}]}


def test_rejected_annotations_carry_one_sentence_per_reason(store):
    out = runview.reverify(trace(store, [BAD]), store)
    (ann,) = out["annotations"]["rejected"]
    assert ann["sentences"] == ["'9:15 PM' doesn't match the cited record's time (9:40 PM).",
                                "'Bob' doesn't appear in any cited record."]


def test_repair_view_pairs_round_one_rejections_with_round_two(store):
    out = runview.reverify(trace(store, [WRONG_TIME, GOOD, GHOST, BOB], [GOOD, FIXED, BOB_AGAIN]), store)
    rep = out["repair"]
    assert rep["status"] == "repaired"
    got = [(i["before"]["claim"]["text"], i["outcome"], i["after"] and i["after"]["claim"]["text"]) for i in rep["items"]]
    assert got == [(WRONG_TIME["text"], "corrected", FIXED["text"]), (GHOST["text"], "dropped", None),
                   (BOB["text"], "still rejected", BOB_AGAIN["text"])]
    assert rep["kept"] == 1 and rep["new"] == 0
    assert rep["items"][0]["before"]["sentences"] == ["'9:15 PM' doesn't match the cited record's time (9:40 PM)."]
    assert rep["items"][0]["after"]["verdict"] == "verified"
    assert [e["text"] for e in rep["items"][0]["after"]["entities"]] == ["Liz Dehner", "9:40 PM"]
    assert rep["items"][2]["after"]["sentences"][0] == "Cited direction = 'incoming'; the record says 'outgoing'."
    assert rep["summary"] == "Round 1 rejected 3 claims. Round 2: 1 corrected, 1 dropped, 1 still rejected."


def test_a_resubmitted_claim_counts_as_still_rejected_and_extra_claims_as_new(store):
    out = runview.reverify(trace(store, [BOB], [BOB, GOOD]), store)
    (item,) = out["repair"]["items"]
    assert item["outcome"] == "still rejected" and item["after"]["index"] == 0
    assert out["repair"]["new"] == 1


def test_a_failed_repair_round_keeps_round_one(store):
    ev = trace(store, [BAD])
    ev.insert(-1, {"step": 5, "type": "repair", "payload": {"rejected": []}})
    ev.insert(-1, {"step": 6, "type": "validation_error",
                   "payload": {"error": "repair round failed: step cap reached; keeping round 1"}})
    rep = runview.reverify(ev, store)["repair"]
    assert rep["status"] == "failed" and "keeping round 1" in rep["summary"]
    assert [i["outcome"] for i in rep["items"]] == ["still rejected"]


def test_no_repair_view_without_a_repair_round(store):
    assert runview.reverify(trace(store, [BAD, GOOD]), store)["repair"] is None


# ------------------------------------------------------------------------------ F1: cited-value spans
def cv(text, cites, reasons=(), entities=()):
    claim = {"text": text, "citations": [{"artifact_id": CALL, "field": f, "value": v} for f, v in cites]}
    return [(e["text"], e["field"], e["citation_index"])
            for e in runview.cited_spans(claim, [dict(r) for r in reasons], [dict(e) for e in entities])]


def test_cited_text_values_quotes_and_numbers_are_spans():
    assert cv("A photo shows a Cell Phone.", [("label", "cell phone")]) == [("Cell Phone", "label", 0)]
    assert cv("It said “meet me at the north gate.” Then nothing.",
              [("body", "Meet me at the north gate after the game")]) == [("meet me at the north gate", "body", 0)]
    assert cv("At 35.65961, -78.8730535 and lat 35.6596081.", [("lat", "35.6596081"), ("lon", "-78.8730535")]) == [
        ("35.65961", "lat", 0), ("-78.8730535", "lon", 1), ("35.6596081", "lat", 0)]
    assert cv("It lasted 121 seconds, scored 0.9004.", [("duration_s", "121.0"), ("score", "0.9")]) == [
        ("121", "duration_s", 0), ("0.9004", "score", 1)]
    assert cv("Quoted 'the north gate' and “north…after the game”.", [("body", "Meet me at the north gate after the game")]) == [
        ("the north gate", "body", 0), ("after the game", "body", 0)]   # "north" alone is too short for a quote


def test_cited_value_spans_need_a_passing_citation_whole_words_and_minimum_lengths():
    assert cv("A photo shows a car.", [("label", "car")], reasons=[{"code": "VALUE_MISMATCH", "citation_index": 0}]) == []
    assert cv("A photo shows a car.", [("label", "car")], reasons=[{"code": "OUTSIDE_WINDOW", "citation_index": 0}]) == []
    assert cv("A photo shows a car.", [("label", "car")], reasons=[{"code": "UNGROUNDED_ENTITY:date",
                                                                     "citation_index": None}]) == [("car", "label", 0)]
    assert cv("The card was scanned.", [("label", "car")]) == []          # whole words only
    assert cv("It was an ox.", [("label", "ox")]) == []                    # shorter than 3
    assert cv('He said "hi you".', [("body", "hi you there")]) == []      # a quote shorter than 8
    assert cv("Direction: outgoing.", [("direction", "outgoing")]) == []  # not a text-like field
    assert cv("At 35.6598 it stopped.", [("lat", "35.6596081")]) == []   # outside 1e-4
    assert cv("At 35.6596081, -78.8730535.", [("body", "35.6596081")]) == [("35.6596081", "body", 0)]


def test_r4_spans_take_precedence_over_cited_value_spans():
    quote = 'It said "Liz Dehner will call at 9:15 PM tonight".'
    ents = [{"type": "contact", "start": quote.index("Liz"), "end": quote.index("Liz") + 10},
            {"type": "clock_time", "start": quote.index("9:15"), "end": quote.index("9:15") + 7}]
    assert cv(quote, [("body", "Liz Dehner will call at 9:15 PM tonight, ok?")], entities=ents) == [
        ("will call at", "body", 0), ("tonight", "body", 0)]


def test_annotations_carry_cited_value_spans_for_verified_and_rejected_claims(store):
    good = {"text": "Liz Dehner placed an outgoing Phone call at 9:40 PM.",
            "citations": [{"artifact_id": CALL, "field": "app", "value": "Phone"},
                          {"artifact_id": CALL, "field": "ts_local", "value": "2024-07-27T21:40:00-04:00"},
                          {"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"}]}
    bad = {"text": "The Phone call to Liz Dehner lasted 99 seconds.",
           "citations": [{"artifact_id": CALL, "field": "app", "value": "Phone"},
                         {"artifact_id": CALL, "field": "duration_s", "value": "99"}]}
    ann = runview.reverify(trace(store, [good, bad]), store)["annotations"]
    (v,) = ann["verified"]
    assert [(e["type"], e["text"]) for e in v["entities"]] == [
        ("contact", "Liz Dehner"), ("cited_value", "Phone"), ("clock_time", "9:40 PM")]
    (r,) = ann["rejected"]
    assert [(e["type"], e["text"], e["grounded"]) for e in r["entities"]] == [
        ("cited_value", "Phone", True), ("contact", "Liz Dehner", False)]      # "99" failed: VALUE_MISMATCH
    assert r["sentences"][0].startswith("Cited duration_s = '99'")
