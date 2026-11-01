"""Red-team suite for tbx.checker.check (CONTRACTS.md section 7). Offline, no LLM.

Run:  pytest tests/redteam -v
Gate: TBX_STRICT_TESTS=1 pytest tests/redteam   (skips become failures)
"""
from __future__ import annotations

import json
import re
import socket

import pytest
import tbxcase
from redteam_cases import ACCEPT, CASES, REJECT, claim
from tbxcase import EVENING, IDS, KNOWN_CONTACTS, cite, make_window

STABLE_CODES = {"UNKNOWN_ARTIFACT", "UNKNOWN_FIELD", "VALUE_MISMATCH", "OUTSIDE_WINDOW", "TIME_RANGE_UNSUPPORTED", "NO_CITATIONS"}
ENTITY_TYPES = {"phone", "email", "url", "timestamp", "date", "clock_time", "coordinate", "contact"}
CITATION_LEVEL = {"UNKNOWN_ARTIFACT", "UNKNOWN_FIELD", "VALUE_MISMATCH", "OUTSIDE_WINDOW"}


def _dump(obj):
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _reason(r) -> dict:
    r = _dump(r)
    assert isinstance(r, dict) and {"code", "detail", "citation_index"} <= set(r), f"reason shape: {r!r}"
    return r


def run_case(tbx, store, case: tbxcase.Case):
    answer = tbx.Answer.model_validate({"claims": [case.claim]})
    window = make_window(tbx, *case.window) if case.window else None
    return tbx.check(answer, store, window)


def explain(case, result) -> str:
    rej = [{"claim": _dump(r["claim"] if isinstance(r, dict) else r.claim)["text"],
            "reasons": [_reason(x) for x in (r["reasons"] if isinstance(r, dict) else r.reasons)]}
           for r in result.rejected]
    return f"\ncase {case.id}: {case.note}\nclaim: {json.dumps(case.claim)}\nrejected: {json.dumps(rej, indent=1)}"


def reasons_of(result) -> list[dict]:
    assert len(result.rejected) == 1
    entry = result.rejected[0]
    reasons = entry["reasons"] if isinstance(entry, dict) else entry.reasons
    return [_reason(r) for r in reasons]


# ---------------------------------------------------------------------------- preconditions
def test_store_preconditions(case_store):
    """If these fail, every case below is meaningless: fix the Store first."""
    assert case_store.case_tz == tbxcase.CASE_TZ
    assert case_store.known_contacts() == KNOWN_CONTACTS
    for key, artifact_id in IDS.items():
        row = case_store.get(artifact_id)
        assert row is not None, f"{key} {artifact_id} not readable back from the store"
        assert "table" in row
    assert case_store.get("call_0000000000") is None


# ---------------------------------------------------------------------------- the cases
@pytest.mark.parametrize("case", ACCEPT, ids=lambda c: c.id)
def test_must_accept(tbx, case_store, case):
    result = run_case(tbx, case_store, case)
    assert len(result.verified) == 1 and len(result.rejected) == 0, explain(case, result)
    assert _dump(result.verified[0])["text"] == case.claim["text"]


@pytest.mark.parametrize("case", REJECT, ids=lambda c: c.id)
def test_must_reject(tbx, case_store, case):
    result = run_case(tbx, case_store, case)
    assert len(result.verified) == 0 and len(result.rejected) == 1, explain(case, result)
    entry = result.rejected[0]
    assert _dump(entry["claim"] if isinstance(entry, dict) else entry.claim)["text"] == case.claim["text"]
    reasons = reasons_of(result)
    got = {r["code"] for r in reasons}
    assert got == set(case.codes), f"expected exactly {sorted(case.codes)}, got {sorted(got)}" + explain(case, result)
    for code, want in (case.indexes or {}).items():
        idx = {r["citation_index"] for r in reasons if r["code"] == code}
        assert idx == want, f"{code}: citation_index {sorted(idx, key=str)} != {sorted(want)}" + explain(case, result)
    for code, text in (case.detail_contains or {}).items():
        assert any(text in r["detail"] for r in reasons if r["code"] == code), explain(case, result)


# ---------------------------------------------------------------------------- reason shape
def test_reason_codes_are_stable_api(tbx, case_store):
    """Every code is one of the 7 stable codes; citation-level codes carry a valid citation_index,
    claim-level codes carry None; details are non-empty strings."""
    for case in REJECT:
        result = run_case(tbx, case_store, case)
        n = len(case.claim["citations"])
        for r in reasons_of(result):
            code = r["code"]
            if code.startswith("UNGROUNDED_ENTITY:"):
                assert code.split(":", 1)[1] in ENTITY_TYPES, f"{case.id}: unknown entity type in {code}"
            else:
                assert code in STABLE_CODES, f"{case.id}: unstable code {code}"
            if code in CITATION_LEVEL:
                assert isinstance(r["citation_index"], int) and 0 <= r["citation_index"] < n, f"{case.id}: {r}"
            else:
                assert r["citation_index"] is None, f"{case.id}: claim-level {code} should have citation_index None: {r}"
            assert isinstance(r["detail"], str) and r["detail"].strip(), f"{case.id}: empty detail for {code}"


# ---------------------------------------------------------------------------- whole-answer properties
def _all_claims_answer(tbx):
    """Every evening-window case in one Answer: exercises partitioning and determinism."""
    claims = [c.claim for c in CASES if c.window == EVENING]
    return tbx.Answer.model_validate({"claims": claims, "note": "red-team batch"}), claims


def test_batch_partition_matches_single_claim_results(tbx, case_store):
    answer, claims = _all_claims_answer(tbx)
    result = tbx.check(answer, case_store, make_window(tbx, *EVENING))
    expected_verified = [c.claim["text"] for c in CASES if c.window == EVENING and c.expect == "verified"]
    expected_rejected = [c.claim["text"] for c in CASES if c.window == EVENING and c.expect == "rejected"]
    got_verified = [_dump(c)["text"] for c in result.verified]
    got_rejected = [_dump(r["claim"] if isinstance(r, dict) else r.claim)["text"] for r in result.rejected]
    assert sorted(got_verified) == sorted(expected_verified)
    assert sorted(got_rejected) == sorted(expected_rejected)
    assert len(got_verified) + len(got_rejected) == len(claims)


def test_check_is_deterministic(tbx, case_store):
    answer, _ = _all_claims_answer(tbx)
    window = make_window(tbx, *EVENING)
    first = tbx.check(answer, case_store, window).model_dump_json()
    for _ in range(3):
        assert tbx.check(answer, case_store, window).model_dump_json() == first


def test_check_makes_no_network_calls(tbx, case_store, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("checker attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    answer, _ = _all_claims_answer(tbx)
    tbx.check(answer, case_store, make_window(tbx, *EVENING))


def test_zero_claims_is_a_valid_answer(tbx, case_store):
    answer = tbx.Answer.model_validate({"claims": [], "note": "no evidence found in window"})
    result = tbx.check(answer, case_store, make_window(tbx, *EVENING))
    assert list(result.verified) == [] and list(result.rejected) == []


def test_check_does_not_mutate_the_answer(tbx, case_store):
    answer, _ = _all_claims_answer(tbx)
    before = answer.model_dump_json()
    tbx.check(answer, case_store, make_window(tbx, *EVENING))
    assert answer.model_dump_json() == before


def test_every_rejection_lists_a_reason(tbx, case_store):
    """No silent rejections and no empty reason lists."""
    answer, _ = _all_claims_answer(tbx)
    result = tbx.check(answer, case_store, make_window(tbx, *EVENING))
    for entry in result.rejected:
        reasons = entry["reasons"] if isinstance(entry, dict) else entry.reasons
        assert len(reasons) >= 1


def test_ids_in_case_store_follow_the_contract_formula():
    """Self-check of the fixture (not of tbx): derived IDs use prefix + 10 hex chars."""
    for key, artifact_id in IDS.items():
        assert re.fullmatch(r"(msg|call|loc|med|trn|tag)_[0-9a-f]{10}", artifact_id), (key, artifact_id)


def test_case_table_size():
    """The suite is meant to stay around 30+ cases; this guards against a cases file that silently empties."""
    assert len(ACCEPT) >= 15 and len(REJECT) >= 25, (len(ACCEPT), len(REJECT))
    assert claim("x")["citations"] == [] and cite("CALL_ALICE", "number", "1")["artifact_id"] == IDS["CALL_ALICE"]
