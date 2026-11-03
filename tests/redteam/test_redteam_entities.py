"""CONTRACTS.md v1.3 section 7: CheckResult.entities, the checked spans of each claim's text.

The UI uses these to show which words code verified (green), which failed (red), and that
everything else is the model's own wording. These tests pin the shape and the link to R4.
"""
from __future__ import annotations

import json

import pytest
from redteam_cases import CASES, META_CLAIMS, OPEN_CLAIMS, REAL_QUOTE_CLAIM
from tbxcase import EVENING, make_window

ENTITY_TYPES = {"phone", "email", "url", "timestamp", "date", "clock_time", "coordinate", "contact"}
TRAILING = tuple(".,;:!?")


def _dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def entities_of(result) -> list[list[dict]]:
    ents = getattr(result, "entities", None)
    assert ents is not None, "CheckResult.entities is missing (CONTRACTS v1.3 section 7)"
    return [[_dump(e) for e in per_claim] for per_claim in ents]


def check_claims(tbx, store, claims, window=EVENING):
    answer = tbx.Answer.model_validate({"claims": claims})
    return tbx.check(answer, store, make_window(tbx, *window) if window else None)


def reasons_of(result) -> list[dict]:
    out = []
    for entry in result.rejected:
        for r in (entry["reasons"] if isinstance(entry, dict) else entry.reasons):
            out.append(_dump(r))
    return out


def test_entities_align_with_answer_claims(tbx, case_store):
    claims = [c.claim for c in CASES if c.window == EVENING]
    result = check_claims(tbx, case_store, claims)
    ents = entities_of(result)
    assert len(ents) == len(claims), "one entity list per claim, in answer order"
    for claim, per_claim in zip(claims, ents):
        text = claim["text"]
        assert [(e["start"], e["end"]) for e in per_claim] == sorted((e["start"], e["end"]) for e in per_claim)
        for e in per_claim:
            assert set(e) >= {"type", "text", "start", "end", "grounded"}, e
            assert e["type"] in ENTITY_TYPES and isinstance(e["grounded"], bool), e
            assert 0 <= e["start"] < e["end"] <= len(text) and text[e["start"]:e["end"]] == e["text"], (text, e)
            assert e["text"] == e["text"].strip() and not e["text"].endswith(TRAILING), e
    assert '"entities"' in result.model_dump_json()


def test_zero_claims_gives_empty_entities(tbx, case_store):
    result = check_claims(tbx, case_store, [])
    assert entities_of(result) == []


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_ungrounded_entities_match_reason_codes(tbx, case_store, case):
    """grounded is false exactly when that entity produced UNGROUNDED_ENTITY:<type>."""
    result = check_claims(tbx, case_store, [case.claim], case.window)
    (per_claim,) = entities_of(result)
    ungrounded_types = {e["type"] for e in per_claim if not e["grounded"]}
    reason_types = {r["code"].split(":", 1)[1] for r in reasons_of(result) if r["code"].startswith("UNGROUNDED_ENTITY:")}
    assert ungrounded_types == reason_types, (case.id, per_claim, reason_types)
    if case.expect == "verified":
        assert all(e["grounded"] for e in per_claim), (case.id, per_claim)


def _find(per_claim, typ, contains):
    hits = [e for e in per_claim if e["type"] == typ and contains in e["text"]]
    assert hits, f"no {typ} entity containing {contains!r} in {json.dumps(per_claim)}"
    return hits


def test_real_open_weights_window_restatement_spans(tbx, case_store):
    """R27: the window bounds are the two red spans; the call's own time, number and name are green."""
    (per_claim,) = entities_of(check_claims(tbx, case_store, [OPEN_CLAIMS[0]]))
    red = sorted(e["text"] for e in per_claim if not e["grounded"])
    assert red == ["2023-03-14T01:00:00Z", "2023-03-14T03:00:00Z"], per_claim
    assert all(e["grounded"] for e in _find(per_claim, "timestamp", "2023-03-14T01:42:07Z"))
    assert all(e["grounded"] for e in _find(per_claim, "phone", "010-1234"))
    assert [e["grounded"] for e in _find(per_claim, "contact", "Alice")] == [True]


def test_real_meta_claim_has_no_double_extraction(tbx, case_store):
    """A01 (C1): the ISO timestamp is one entity, not also a date, a clock time or a phone."""
    (per_claim,) = entities_of(check_claims(tbx, case_store, [META_CLAIMS[0]]))
    assert {e["type"] for e in per_claim} == {"phone", "timestamp", "contact"}, per_claim
    assert all(e["grounded"] for e in per_claim)


def test_case_local_clock_and_date_spans(tbx, case_store):
    claim = next(c.claim for c in CASES if c.id == "A06_ts_local_clock_and_month_day_in_case_tz")
    (per_claim,) = entities_of(check_claims(tbx, case_store, [claim]))
    assert all(e["grounded"] for e in _find(per_claim, "clock_time", "9:42"))
    assert all(e["grounded"] for e in _find(per_claim, "date", "March 13"))
    assert [e["text"] for e in per_claim if e["type"] == "contact"] == ["Alice"]


def test_wrong_clock_time_is_the_red_span(tbx, case_store):
    claim = next(c.claim for c in CASES if c.id == "R20_wrong_clock_time")
    (per_claim,) = entities_of(check_claims(tbx, case_store, [claim]))
    assert [e["grounded"] for e in _find(per_claim, "clock_time", "9:15")] == [False]
    assert [e["grounded"] for e in _find(per_claim, "contact", "Alice")] == [True]


def test_every_contact_occurrence_is_listed(tbx, case_store):
    claim = {"text": "Alice called; later Alice and Bob texted.", "time_start": None, "time_end": None,
             "citations": [{"artifact_id": "call_3f9a1c2b7d", "field": "contact_name", "value": "Alice"}]}
    (per_claim,) = entities_of(check_claims(tbx, case_store, [claim]))
    contacts = [(e["text"], e["grounded"]) for e in per_claim if e["type"] == "contact"]
    assert contacts == [("Alice", True), ("Alice", True), ("Bob", False)], contacts


def test_times_inside_a_verified_quote_are_grounded(tbx, case_store):
    """v1.4 (C9): the real q07 claim. '8:00' and '9:30' are inside the verified transcript quote, so they are
    grounded; the claim's own '9:28 AM' is checked against the cited timestamp and also grounded."""
    (per_claim,) = entities_of(check_claims(tbx, case_store, [REAL_QUOTE_CLAIM], window=None))
    clocks = {e["text"]: e["grounded"] for e in per_claim if e["type"] == "clock_time"}
    assert clocks.get("8:00") is True and clocks.get("9:30") is True, per_claim
    assert any(k.startswith("9:28") and v for k, v in clocks.items()), per_claim

