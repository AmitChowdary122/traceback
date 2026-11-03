"""Recorded runs, re-verified on load: the extra fields of the run view.

`reverify(events, store)` re-runs the deterministic checker on each recorded final_answer, with the run's recorded
window, against the store, and compares the result with the recorded check_result. It also gives the spans the page
colors for every claim it shows: the v1.3 entity spans (including for runs recorded before v1.3), plus "cited_value"
spans for the words that match the value of a citation that passed (display only; the checker is unchanged). No model
is called.

When a round's re-check differs from its recording, the page shows the re-checked verdicts (`verdicts`, with
annotations aligned to them), and the warning lists the recorded verdicts claim by claim (`recorded_verdicts`).

The web app calls it for GET /api/runs/{id}; `tbx export-static` saves that response, so a static copy carries the
re-check made at export time.
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Any

from tbx import explain
from tbx import timeutil as tu
from tbx.checker import MEASURE_FIELDS, MIN_QUOTE, TEXT_FIELDS, _float, _fold_map, check, contact_pattern, fold
from tbx.models import Answer
from tbx.store import Store

MAX_CHANGES = 8
PAIR_TEXT = 0.5          # a round-2 claim this similar to a round-1 rejected claim is its repair...
PAIR_TEXT_SAME_ROW = 0.3  # ...or this similar when it also cites one of the same records

# Cited-value spans: the words of a claim that match the value of a citation that passed. Display only: the checker
# and CheckResult are unchanged (contract v1.3); these spans are added to the R4 spans the page colors green.
VALUE_FIELDS = frozenset({"body", "text", "transcript", "label", "contact_name", "sender", "recipient", "app", "kind",
                          "source", "mime_type", "file_path"})
NUMBER_TOL = {"lat": 1e-4, "lon": 1e-4, "score": 1e-3, **{f: 1e-6 for f in MEASURE_FIELDS}}   # CONTRACTS section 7
MIN_VALUE = 3
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”|„([^“”]+)[“”]|‘([^’]+)’|«([^»]+)»|(?<!\w)'([^']+)'(?!\w)")
_ELLIPSIS = re.compile(r"\s*(?:…|\.\.\.)\s*")
_NUMBER = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|\.\d)")
_LEAD = " \t\n,;:"


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

    A lookup by canonical claim JSON, so a record whose lists are out of answer order, or that misses a claim, still
    pairs every claim it does have. Identical claims take that claim's entries in turn (verified ones first)."""
    slots: dict[str, list[tuple[str, int]]] = {}
    for j, c in enumerate(record.get("verified") or []):
        slots.setdefault(_canon(c), []).append(("verified", j))
    for j, r in enumerate(record.get("rejected") or []):
        slots.setdefault(_canon((r or {}).get("claim")), []).append(("rejected", j))
    out: list[tuple[str, int] | None] = []
    for c in claims:
        free = slots.get(_canon(c))
        out.append(free.pop(0) if free else None)
    return out


def recorded_verdicts(claims: list[dict], record: dict) -> list[dict]:
    """Each answer claim's verdict in a (recorded) check result: {"claim" (1-based), "text", "verdict", "codes"}.

    verdict is "verified", "rejected" or "not recorded"; codes are the rejected claim's reason codes."""
    out = []
    for i, (c, spot) in enumerate(zip(claims, place(claims, record)), 1):
        verdict = spot[0] if spot else "not recorded"
        codes = _codes(record["rejected"][spot[1]]) if spot and spot[0] == "rejected" else []
        out.append({"claim": i, "text": c.get("text", ""), "verdict": verdict, "codes": codes})
    return out


def _verdict_line(v: dict) -> str:
    if v["verdict"] == "rejected":
        return f"claim {v['claim']} rejected ({', '.join(v['codes']) or 'no reason recorded'})"
    return f"claim {v['claim']} {v['verdict']}"


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


# ------------------------------------------------------------------------------ cited-value spans
def _quotes(text: str) -> list[tuple[int, int]]:
    """The [start, end) of each quoted passage (inside the quotes, trimmed); a passage with an ellipsis gives one
    piece per fragment."""
    out = []
    for m in _QUOTED.finditer(text):
        g = next(k for k in range(1, len(m.groups()) + 1) if m.group(k) is not None)
        a, b = m.start(g), m.end(g)
        for part in _ELLIPSIS.split(text[a:b]):
            if not part.strip():
                continue
            at = text.index(part, a)
            x, y = at + len(part) - len(part.lstrip()), at + len(part.rstrip())
            out.append((x, y))
            a = at + len(part)
    return out


def _number(s: str) -> float | None:
    return _float(s.replace(",", "").replace("−", "-"))


def cited_spans(claim: dict, reasons: list[dict], entities: list[dict]) -> list[dict]:
    """Spans of the claim text that match the value of a citation that passed (no reason carries its index):

    - a text-like field (VALUE_FIELDS): a whole-word, NFKC-folded, case-insensitive occurrence of the value (>= 3);
    - body/text/transcript: a quoted passage (>= 8 characters) that is a substring of the cited value;
    - lat/lon/score/measures: a number equal to the cited value within the field's tolerance.

    Overlapping matches merge; R4 entity spans take precedence, so they are cut out. Each span: {"type":
    "cited_value", "text", "start", "end", "grounded": True, "field", "citation_index"} (Python string offsets)."""
    text = claim.get("text") or ""
    cites = claim.get("citations") or []
    failed = {r.get("citation_index") for r in reasons or [] if isinstance(r.get("citation_index"), int)}
    found: list[tuple[int, int, str, int]] = []
    fmap = None
    quotes = None
    for i, c in enumerate(cites):
        field, value = c.get("field"), str(c.get("value") if c.get("value") is not None else "")
        if i in failed or not field:
            continue
        if field in VALUE_FIELDS and len(fold(value)) >= MIN_VALUE:
            if fmap is None:
                fmap = _fold_map(text)
            folded, starts, ends = fmap
            found += [(starts[m.start()], ends[m.end() - 1], field, i)
                      for m in contact_pattern(value).finditer(folded)]
        if field in TEXT_FIELDS:
            quotes = _quotes(text) if quotes is None else quotes
            v = fold(value)
            for a, b in quotes:
                q = fold(text[a:b])
                while b > a and len(q) >= MIN_QUOTE and q not in v and text[b - 1] in ".,;:!?":
                    b -= 1      # punctuation the claim put inside the quotes
                    q = fold(text[a:b])
                if len(q) >= MIN_QUOTE and q in v:
                    found.append((a, b, field, i))
        if field in NUMBER_TOL and (x := _number(value)) is not None:
            for m in _NUMBER.finditer(text):
                y = _number(m.group(0))
                if y is not None and abs(x - y) <= NUMBER_TOL[field]:
                    found.append((m.start(), m.end(), field, i))
    merged: list[list] = []
    for a, b, field, i in sorted(found):
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, field, i])
    taken = sorted((e["start"], e["end"]) for e in entities or [])
    out = []
    for a, b, field, i in merged:
        pieces = [(a, b)]
        for s, e in taken:
            pieces = [p for x, y in pieces for p in ((x, min(y, s)), (max(x, e), y)) if p[0] < p[1]]
        for x, y in pieces:
            while x < y and text[x] in _LEAD:
                x += 1
            while y > x and text[y - 1].isspace():
                y -= 1
            if any(ch.isalnum() for ch in text[x:y]):
                out.append({"type": "cited_value", "text": text[x:y], "start": x, "end": y, "grounded": True,
                            "field": field, "citation_index": i})
    return out


def _spans(claim: dict, reasons: list[dict], entities: list[dict]) -> list[dict]:
    """The page's spans for one claim: its R4 entities plus its cited-value spans, by start."""
    try:
        extra = cited_spans(claim, reasons, entities)
    except Exception:  # noqa: BLE001 - highlighting is a display aid; never lose the R4 spans over it
        extra = []
    return sorted([*entities, *extra], key=lambda e: (e["start"], e["end"], e["type"] == "cited_value"))


def _annotations(claims: list[dict], record: dict, entities: list[list[dict]], store: Store) -> dict:
    """For the claims the page shows, aligned with record["verified"] and record["rejected"]: each claim's answer
    index and spans (R4 entities + cited values), plus one plain-English sentence per reason for the rejected ones."""
    out: dict[str, list[dict]] = {"verified": [{"index": None, "entities": []} for _ in record.get("verified", [])],
                                  "rejected": [{"index": None, "entities": []} for _ in record.get("rejected", [])]}
    for i, spot in enumerate(place(claims, record)):
        if spot is not None:
            ents = entities[i] if i < len(entities) else []
            reasons = (record["rejected"][spot[1]] or {}).get("reasons", []) if spot[0] == "rejected" else []
            out[spot[0]][spot[1]] = {"index": i, "entities": _spans(claims[i], reasons, ents)}
    tz = store.case_tz
    for entry, ann in zip(record.get("rejected", []), out["rejected"]):
        try:
            ann["sentences"] = explain.sentences(entry, [e for e in ann["entities"] if e["type"] != "cited_value"],
                                                 store, tz)
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
    out: dict[str, Any] = {"recheck": {"status": "none", "message": "", "rounds": [], "entities_recorded": False,
                                       "shown": "recorded"},
                           "annotations": empty, "first_round_annotations": None, "repair": None,
                           "verdicts": None, "first_round_verdicts": None}
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
        out["verdicts"] = _lists(last["check"])
        return out

    summary = []
    for r, now in zip(checked, results):
        claims = [c for c in Answer.model_validate({"claims": r["answer"]["claims"]}).model_dump()["claims"]]
        changes = _compare(claims, r["check"], now)
        summary.append({"round": r["round"], "same": not changes, "recorded": _counts(r["check"]),
                        "rechecked": _counts(now), "changes": changes, "_claims": claims, "_now": now,
                        **({"recorded_verdicts": recorded_verdicts(claims, r["check"])} if changes else {})})
    same = all(s["same"] for s in summary)
    label = "; ".join(f"round {s['round']}: {s['recorded']}" for s in summary) if len(summary) > 1 else summary[0]["recorded"]
    if same:
        message = f"Re-checked against the store: the same result as recorded ({label})."
    else:
        recorded = " ".join(
            ("Recorded" + (f" (round {s['round']})" if len(summary) > 1 else "") + ": "
             + "; ".join(_verdict_line(v) for v in s["recorded_verdicts"]) + ".")
            for s in summary if not s["same"])
        message = ("The re-check differs from the recorded result. " + " ".join(c for s in summary for c in s["changes"])
                   + " The verdicts shown are the re-check's. " + recorded)
    out["recheck"] = {"status": "match" if same else "differs", "message": message,
                      "entities_recorded": all("entities" in r["check"] for r in checked),
                      "shown": "recorded" if same else "rechecked",
                      "rounds": [{k: v for k, v in s.items() if not k.startswith("_")} for s in summary]}
    # A round whose re-check differs is shown as re-checked; its recorded verdicts are listed in the warning.
    views = []
    for s, r in zip(summary, checked):
        record = r["check"] if s["same"] else s["_now"]
        views.append({"claims": s["_claims"], "record": record,
                      "ann": _annotations(s["_claims"], record, s["_now"]["entities"], store)})
    out["annotations"] = views[-1]["ann"]
    out["verdicts"] = _lists(views[-1]["record"])
    if len(views) > 1:
        out["first_round_annotations"] = views[0]["ann"]
        out["first_round_verdicts"] = _lists(views[0]["record"])
    out["repair"] = repair_view(events, views)
    return out


def _lists(record: dict) -> dict:
    return {"verified": record.get("verified", []), "rejected": record.get("rejected", [])}
