"""Recorded runs, re-verified on load: the extra fields of the run view.

`reverify(events, store)` re-runs the deterministic checker on each recorded final_answer, with the run's recorded
window, against the store, and compares the result with the recorded check_result. It also gives the v1.3 entity
spans for every claim the page shows, including runs recorded before v1.3. No model is called.

The web app calls it for GET /api/runs/{id}; `tbx export-static` saves that response, so a static copy carries the
re-check made at export time.
"""
from __future__ import annotations

import json
from typing import Any

from tbx import timeutil as tu
from tbx.checker import check
from tbx.models import Answer
from tbx.store import Store

MAX_CHANGES = 8


def _canon(x: Any) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False, default=str)


def recorded_window(events: list[dict]) -> tu.Window | None:
    start = next((e.get("payload") or {} for e in events if e.get("type") == "run_start"), {})
    w = start.get("window")
    if not w:
        return None
    return tu.Window(tu.parse_utc(w["start_utc"]), tu.parse_utc(w["end_utc"]))


def recorded_rounds(events: list[dict]) -> list[dict]:
    """[{round, answer, check}] in round order: each final_answer payload and the check_result of the same round."""
    answers = [e.get("payload") or {} for e in events if e.get("type") == "final_answer"]
    checks = [e.get("payload") or {} for e in events if e.get("type") == "check_result"]
    out = []
    for i, a in enumerate(answers):
        n = a.get("round", i + 1)
        c = next((c for c in checks if c.get("round") == n), None)
        if c is None and i < len(checks) and "round" not in checks[i]:
            c = checks[i]
        out.append({"round": n, "answer": a, "check": c})
    return out


def place(claims: list[dict], record: dict) -> list[tuple[str, int] | None]:
    """Where each answer claim sits in a check result: ("verified", j), ("rejected", j), or None.

    The checker keeps answer order inside both lists, so one in-order walk pairs them up."""
    ver = record.get("verified") or []
    rej = record.get("rejected") or []
    jv = jr = 0
    out: list[tuple[str, int] | None] = []
    for c in claims:
        cc = _canon(c)
        if jv < len(ver) and _canon(ver[jv]) == cc:
            out.append(("verified", jv))
            jv += 1
        elif jr < len(rej) and _canon((rej[jr] or {}).get("claim")) == cc:
            out.append(("rejected", jr))
            jr += 1
        else:
            out.append(None)
    return out


def _codes(entry: dict | None) -> list[str]:
    return [r.get("code", "?") for r in (entry or {}).get("reasons", [])]


def _compare(claims: list[dict], record: dict, now: dict) -> list[str]:
    """Plain-English differences between a recorded check and the re-check, claim by claim (1-based)."""
    if _canon([record.get("verified", []), record.get("rejected", [])]) == _canon([now["verified"], now["rejected"]]):
        changes = []
    else:
        was, nows = place(claims, record), place(claims, now)
        changes = []
        for i, (a, b) in enumerate(zip(was, nows), 1):
            if a is None:
                changes.append(f"Claim {i} is not in the recorded check result.")
                continue
            if b is None:  # cannot happen: the re-check is of these claims
                changes.append(f"Claim {i} is missing from the re-check.")
                continue
            if a[0] == b[0] == "rejected":
                old, new = record["rejected"][a[1]], now["rejected"][b[1]]
                if _canon(old.get("reasons")) != _canon(new.get("reasons")):
                    changes.append(f"Claim {i}: the reasons differ (recorded {', '.join(_codes(old)) or 'none'}; "
                                   f"re-check {', '.join(_codes(new)) or 'none'}).")
            elif a[0] != b[0]:
                if b[0] == "rejected":
                    why = ", ".join(_codes(now["rejected"][b[1]]))
                    changes.append(f"Claim {i} was recorded as verified; the re-check rejects it ({why}).")
                else:
                    changes.append(f"Claim {i} was recorded as rejected; the re-check verifies it.")
        if len(record.get("verified", [])) + len(record.get("rejected", [])) != len(claims):
            changes.append(f"The recorded check lists {len(record.get('verified', [])) + len(record.get('rejected', []))}"
                           f" claims; the answer has {len(claims)}.")
        if not changes:
            changes.append("The recorded check result differs from the re-check.")
    if "entities" in record and _canon(record["entities"]) != _canon(now["entities"]):
        changes.append("The recorded entity spans differ from the re-check.")
    return changes[:MAX_CHANGES]


def _annotations(claims: list[dict], record: dict, entities: list[list[dict]]) -> dict:
    """Entity spans for the claims the page shows, aligned with record["verified"] and record["rejected"]."""
    out: dict[str, list[dict]] = {"verified": [{"index": None, "entities": []} for _ in record.get("verified", [])],
                                  "rejected": [{"index": None, "entities": []} for _ in record.get("rejected", [])]}
    for i, spot in enumerate(place(claims, record)):
        if spot is not None:
            out[spot[0]][spot[1]] = {"index": i, "entities": entities[i] if i < len(entities) else []}
    return out


def _counts(x: dict) -> str:
    return f"{len(x.get('verified', []))} verified, {len(x.get('rejected', []))} rejected"


def reverify(events: list[dict], store: Store) -> dict:
    """-> {"recheck": {...}, "annotations": {...}, "first_round_annotations": {...} | None}.

    recheck.status is "match" (same result as recorded), "differs" (a visible warning), "error" (the re-check could
    not run) or "none" (no recorded answer yet)."""
    rounds = recorded_rounds(events)
    empty = {"verified": [], "rejected": []}
    out: dict[str, Any] = {"recheck": {"status": "none", "message": "", "rounds": [], "entities_recorded": False},
                           "annotations": empty, "first_round_annotations": None}
    checked = [r for r in rounds if r["check"] is not None]
    if not checked:
        return out
    try:
        window = recorded_window(events)
        results = []
        for r in checked:
            answer = Answer.model_validate({k: v for k, v in r["answer"].items() if k != "round"})
            results.append(check(answer, store, window).model_dump(mode="json"))
    except Exception as e:  # noqa: BLE001 - a bad trace must not break the page; it is reported instead
        out["recheck"].update(status="error", message=f"Could not re-check this run: {type(e).__name__}: {e}"[:400])
        last = checked[-1]
        claims = last["answer"].get("claims") if isinstance(last["answer"].get("claims"), list) else []
        out["annotations"] = _annotations(claims, last["check"], last["check"].get("entities") or [])
        return out

    summary = []
    for r, now in zip(checked, results):
        claims = [c for c in Answer.model_validate({"claims": r["answer"]["claims"]}).model_dump()["claims"]]
        changes = _compare(claims, r["check"], now)
        summary.append({"round": r["round"], "same": not changes, "recorded": _counts(r["check"]),
                        "rechecked": _counts(now), "changes": changes, "_claims": claims, "_now": now})
    same = all(s["same"] for s in summary)
    label = "; ".join(f"round {s['round']}: {s['recorded']}" for s in summary) if len(summary) > 1 else summary[0]["recorded"]
    if same:
        message = f"Re-checked against the store: the same result as recorded ({label})."
    else:
        message = "The re-check differs from the recorded result. " + " ".join(
            c for s in summary for c in s["changes"])
    out["recheck"] = {"status": "match" if same else "differs", "message": message,
                      "entities_recorded": all("entities" in r["check"] for r in checked),
                      "rounds": [{k: v for k, v in s.items() if not k.startswith("_")} for s in summary]}
    last = summary[-1]
    out["annotations"] = _annotations(last["_claims"], checked[-1]["check"], last["_now"]["entities"])
    if len(summary) > 1:
        out["first_round_annotations"] = _annotations(summary[0]["_claims"], checked[0]["check"],
                                                      summary[0]["_now"]["entities"])
    return out
