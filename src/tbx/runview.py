"""Recorded runs, re-verified on load: the extra fields of the run view.

`reverify(events, store)` re-runs the deterministic checker on each recorded final_answer, with the run's recorded
window, against the store, and compares the result with the recorded check_result. It also gives the v1.3 entity
spans for every claim the page shows, including runs recorded before v1.3. No model is called.

The web app calls it for GET /api/runs/{id}; `tbx export-static` saves that response, so a static copy carries the
re-check made at export time.
"""
from __future__ import annotations

import difflib
import json
from typing import Any

from tbx import explain
from tbx import timeutil as tu
from tbx.checker import check, fold
from tbx.models import Answer
from tbx.store import Store

MAX_CHANGES = 8
PAIR_TEXT = 0.5          # a round-2 claim this similar to a round-1 rejected claim is its repair...
PAIR_TEXT_SAME_ROW = 0.3  # ...or this similar when it also cites one of the same records


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


def _annotations(claims: list[dict], record: dict, entities: list[list[dict]], store: Store) -> dict:
    """For the claims the page shows, aligned with record["verified"] and record["rejected"]: each claim's answer
    index and entity spans, plus one plain-English sentence per reason for the rejected ones."""
    out: dict[str, list[dict]] = {"verified": [{"index": None, "entities": []} for _ in record.get("verified", [])],
                                  "rejected": [{"index": None, "entities": []} for _ in record.get("rejected", [])]}
    for i, spot in enumerate(place(claims, record)):
        if spot is not None:
            out[spot[0]][spot[1]] = {"index": i, "entities": entities[i] if i < len(entities) else []}
    tz = store.case_tz
    for entry, ann in zip(record.get("rejected", []), out["rejected"]):
        try:
            ann["sentences"] = explain.sentences(entry, ann["entities"], store, tz)
        except Exception:  # noqa: BLE001 - never lose the codes over a sentence; fall back to the details
            ann["sentences"] = [r.get("detail") or r.get("code", "") for r in entry.get("reasons", [])]
    return out


# ------------------------------------------------------------------------------ the repair view
def _ids(claim: dict) -> set[str]:
    return {c.get("artifact_id") for c in claim.get("citations") or [] if c.get("artifact_id")}


def _pair_score(a: dict, b: dict) -> float | None:
    """How likely round-2 claim b is the repair of round-1 claim a (None: not a candidate)."""
    if _canon(a) == _canon(b):
        return 2.0
    text = difflib.SequenceMatcher(None, fold(a.get("text", "")), fold(b.get("text", "")), autojunk=False).ratio()
    same_row = bool(_ids(a) & _ids(b))
    if text >= PAIR_TEXT or (same_row and text >= PAIR_TEXT_SAME_ROW):
        return text + (0.2 if same_row else 0.0)
    return None


def _side(claims: list[dict], record: dict, ann: dict, i: int) -> dict:
    """One claim of a round, with its verdict, reasons, sentences and spans."""
    spot = place(claims, record)[i]
    verdict, j = spot if spot else ("unknown", -1)
    a = ann.get(verdict, [])[j] if spot and j < len(ann.get(verdict, [])) else {"entities": []}
    entry = record["rejected"][j] if verdict == "rejected" else {}
    return {"index": i, "claim": claims[i], "verdict": verdict, "reasons": entry.get("reasons", []),
            "sentences": a.get("sentences", []), "entities": a.get("entities", [])}


def repair_view(events: list[dict], rounds: list[dict]) -> dict | None:
    """Round-1 rejected claims next to what round 2 submitted: corrected, dropped or still rejected.

    rounds: [{"claims", "record", "ann"}] per checked round. A round-2 claim identical to a round-1 verified claim is
    "kept"; the others are paired with round-1 rejected claims by text similarity (with a bonus for citing one of the
    same records), best pairs first. A round-2 claim paired with nothing is "new"."""
    if not any(e.get("type") == "repair" for e in events) or not rounds:
        return None
    first = rounds[0]
    places1 = place(first["claims"], first["record"])
    rejected1 = [i for i, sp in enumerate(places1) if sp and sp[0] == "rejected"]
    befores = {i: _side(first["claims"], first["record"], first["ann"], i) for i in rejected1}
    if len(rounds) < 2:
        why = next((str((e.get("payload") or {}).get("error", "")) for e in reversed(events)
                    if e.get("type") == "validation_error" and "repair" in str((e.get("payload") or {}).get("error", ""))),
                   "")
        return {"status": "failed", "kept": 0, "new": 0,
                "summary": f"The repair round did not produce a valid answer{f' ({why})' if why else ''}; "
                           "the report keeps round 1.",
                "items": [{"before": befores[i], "outcome": "still rejected", "after": None} for i in rejected1]}
    last = rounds[-1]
    verified1 = {_canon(first["claims"][i]) for i, sp in enumerate(places1) if sp and sp[0] == "verified"}
    kept = {k for k, c in enumerate(last["claims"]) if _canon(c) in verified1}
    pairs = []
    for i in rejected1:
        for k, c in enumerate(last["claims"]):
            if k in kept:
                continue
            score = _pair_score(first["claims"][i], c)
            if score is not None:
                pairs.append((-score, i, k))
    match: dict[int, int] = {}
    used: set[int] = set()
    for _, i, k in sorted(pairs):
        if i not in match and k not in used:
            match[i] = k
            used.add(k)
    items = []
    for i in rejected1:
        if i not in match:
            items.append({"before": befores[i], "outcome": "dropped", "after": None})
            continue
        after = _side(last["claims"], last["record"], last["ann"], match[i])
        items.append({"before": befores[i], "after": after,
                      "outcome": "corrected" if after["verdict"] == "verified" else "still rejected"})
    n = {o: sum(1 for x in items if x["outcome"] == o) for o in ("corrected", "dropped", "still rejected")}
    summary = (f"Round 1 rejected {len(items)} claim{'s' if len(items) != 1 else ''}. Round 2: "
               + ", ".join(f"{v} {k}" for k, v in n.items()) + ".")
    return {"status": "repaired", "summary": summary, "items": items, "kept": len(kept),
            "new": len(last["claims"]) - len(kept) - len(used)}


def _counts(x: dict) -> str:
    return f"{len(x.get('verified', []))} verified, {len(x.get('rejected', []))} rejected"


def reverify(events: list[dict], store: Store) -> dict:
    """-> {"recheck": {...}, "annotations": {...}, "first_round_annotations": {...} | None}.

    recheck.status is "match" (same result as recorded), "differs" (a visible warning), "error" (the re-check could
    not run) or "none" (no recorded answer yet)."""
    rounds = recorded_rounds(events)
    empty = {"verified": [], "rejected": []}
    out: dict[str, Any] = {"recheck": {"status": "none", "message": "", "rounds": [], "entities_recorded": False},
                           "annotations": empty, "first_round_annotations": None, "repair": None}
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
        out["annotations"] = _annotations(claims, last["check"], last["check"].get("entities") or [], store)
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
    views = [{"claims": s["_claims"], "record": r["check"], "ann": _annotations(s["_claims"], r["check"],
                                                                                s["_now"]["entities"], store)}
             for s, r in zip(summary, checked)]
    out["annotations"] = views[-1]["ann"]
    if len(views) > 1:
        out["first_round_annotations"] = views[0]["ann"]
    out["repair"] = repair_view(events, views)
    return out
