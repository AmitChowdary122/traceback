"""Plain-English sentences for the checker's reasons (one per reason, in order).

The reason codes stay the API; these sentences are what a reader sees first. They are built from the claim, the
reason, the artifact row in the store (the stored value, its case-local time) and the claim's v1.3 entity spans
(the red span an UNGROUNDED_ENTITY reason is about). Pure: no network, no model.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta

from tbx import timeutil as tu
from tbx.checker import (NO_CITATIONS, OUTSIDE_WINDOW, TIME_RANGE_UNSUPPORTED, TIMESTAMP_FIELDS, UNKNOWN_ARTIFACT,
                         UNKNOWN_FIELD, VALUE_MISMATCH, fold)
from tbx.store import Store

MAX_VALUE = 160
_ENTITY_DETAIL = re.compile(r"(\w+) (.+) is not supported by the claim's cited values", re.S)
_NOT_ISO = re.compile(r"(time_start|time_end) (.+) is not an ISO time", re.S)
_DOW = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ------------------------------------------------------------------------------ formatting
def _value(v: object) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = " ".join(str(v).split())
    return s if len(s) <= MAX_VALUE else s[:MAX_VALUE - 1] + "…"


def _parts(dt: datetime, tz: str) -> tuple[str, str, str]:
    """-> ("Thu Jun 27, 2024", "9:42:07 PM", "UTC-04:00") in the case timezone."""
    loc = tu.utc_to_local(dt, tz)
    off = loc.strftime("%z")
    clock = f"{(loc.hour + 11) % 12 + 1}:{loc.minute:02d}:{loc.second:02d} {'AM' if loc.hour < 12 else 'PM'}"
    return f"{_DOW[loc.weekday()]} {_MON[loc.month - 1]} {loc.day}, {loc.year}", clock, f"UTC{off[:3]}:{off[3:]}"


def local_text(dt: datetime, tz: str) -> str:
    """A UTC datetime as case-local text, the way the page shows it: "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00)"."""
    day, clock, off = _parts(dt, tz)
    return f"{day} {clock} ({off})"


def span_text(a: datetime, b: datetime, tz: str) -> str:
    """Two instants as one case-local span; the date and offset are written once when they are shared."""
    if a == b:
        return local_text(a, tz)
    (da, ca, oa), (db, cb, ob) = _parts(a, tz), _parts(b, tz)
    if da == db and oa == ob:
        return f"{da} {ca} – {cb} ({oa})"
    return f"{local_text(a, tz)} – {local_text(b, tz)}"


# ------------------------------------------------------------------------------ per reason
def _citation(claim: dict, i: int | None) -> dict:
    cites = claim.get("citations") or []
    return cites[i] if isinstance(i, int) and 0 <= i < len(cites) else {}


def _stored(row: dict, field: str, tz: str) -> object:
    if field == "ts_local":
        return tu.local_iso(row.get("ts_utc"), tz)
    return row.get(field)


def _value_mismatch(reason: dict, claim: dict, store: Store, tz: str) -> str:
    c = _citation(claim, reason.get("citation_index"))
    field, cited = c.get("field", "?"), c.get("value", "")
    row = store.get(c.get("artifact_id", "")) if c else None
    if row is None:
        return reason.get("detail") or VALUE_MISMATCH
    stored = _stored(row, field, tz)
    if stored is None:
        what = "timestamp" if field in TIMESTAMP_FIELDS else field
        return f"Cited {field} = '{_value(cited)}'; the record has no {what}."
    return f"Cited {field} = '{_value(cited)}'; the record says '{_value(stored)}'."


def _outside_window(reason: dict, claim: dict, store: Store, tz: str) -> str:
    c = _citation(claim, reason.get("citation_index"))
    row = store.get(c.get("artifact_id", "")) if c else None
    if row is None:
        return reason.get("detail") or OUTSIDE_WINDOW
    if not row.get("ts_utc"):
        return "This record has no timestamp."
    return f"This record is from {local_text(tu.parse_utc(row['ts_utc']), tz)}, outside the question window."


def _red_span(etype: str, key: str | None, nth: int, spans: list[dict]) -> str | None:
    """The claim's own text of the entity this reason is about (its red span)."""
    red = [e for e in spans or [] if e.get("type") == etype and not e.get("grounded")]
    if etype == "contact":  # one reason per contact name, named in the detail
        return next((e["text"] for e in red if key is not None and fold(e["text"]) == fold(key)), key)
    # other types: one reason per occurrence, in text order, like the spans
    return red[nth]["text"] if nth < len(red) else key


def _ungrounded(reason: dict, spans: list[dict], seen: dict[str, int]) -> str:
    etype = reason["code"].split(":", 1)[1]
    nth = seen.get(etype, 0)
    seen[etype] = nth + 1
    key = None
    m = _ENTITY_DETAIL.fullmatch(reason.get("detail") or "")
    if m:
        try:
            v = ast.literal_eval(m.group(2))
            key = v if isinstance(v, str) else None
        except (ValueError, SyntaxError):
            key = None
    text = _red_span(etype, key, nth, spans)
    if text is None:
        return reason.get("detail") or reason["code"]
    return f"'{_value(text)}' doesn't appear in any cited record."


def _support(claim: dict, store: Store) -> list[datetime]:
    """The cited artifacts' times, as R5 sees them (a call's end counts)."""
    out: list[datetime] = []
    for c in claim.get("citations") or []:
        row = store.get(c.get("artifact_id", ""))
        if row is None or not row.get("ts_utc"):
            continue
        ts = tu.parse_utc(row["ts_utc"])
        out.append(ts)
        if row["table"] == "calls":
            try:
                out.append(ts + timedelta(seconds=float(row.get("duration_s"))))
            except (TypeError, ValueError):
                pass
    return out


def _time_range(reason: dict, claim: dict, store: Store, tz: str) -> str:
    detail = reason.get("detail") or ""
    m = _NOT_ISO.fullmatch(detail)
    if m:
        return f"The claim's {m.group(1)} {m.group(2)} is not a valid time."
    bounds: dict[str, datetime] = {}
    for name in ("time_start", "time_end"):
        try:
            if claim.get(name):
                bounds[name] = tu.parse_ts(claim[name], tz)[0]
        except ValueError:
            pass
    start, end = bounds.get("time_start"), bounds.get("time_end")
    if detail == "time_start is after time_end" and start and end:
        return f"The claim's start, {local_text(start, tz)}, is after its end, {local_text(end, tz)}."
    if start and end:
        says = f"The claim says {span_text(start, end, tz)}"
    elif start:
        says = f"The claim says it starts at {local_text(start, tz)}"
    elif end:
        says = f"The claim says it ends at {local_text(end, tz)}"
    else:
        return detail or TIME_RANGE_UNSUPPORTED
    support = _support(claim, store)
    if not support:
        return f"{says}, but none of the cited records has a timestamp."
    return f"{says}; the cited records span {span_text(min(support), max(support), tz)}."


def sentence(reason: dict, claim: dict, spans: list[dict], store: Store, tz: str, seen: dict[str, int]) -> str:
    code = reason.get("code") or ""
    if code == VALUE_MISMATCH:
        return _value_mismatch(reason, claim, store, tz)
    if code == OUTSIDE_WINDOW:
        return _outside_window(reason, claim, store, tz)
    if code.startswith("UNGROUNDED_ENTITY:"):
        return _ungrounded(reason, spans, seen)
    if code == TIME_RANGE_UNSUPPORTED:
        return _time_range(reason, claim, store, tz)
    if code == UNKNOWN_ARTIFACT:
        return f"No record {_citation(claim, reason.get('citation_index')).get('artifact_id', '?')} exists in this extraction."
    if code == UNKNOWN_FIELD:
        return f"{_citation(claim, reason.get('citation_index')).get('field', '?')} can't be cited."
    if code == NO_CITATIONS:
        return "The claim cites nothing."
    return reason.get("detail") or code


def sentences(entry: dict, spans: list[dict], store: Store, tz: str) -> list[str]:
    """One plain sentence per reason of a rejected claim ({"claim", "reasons"}), in the reasons' order."""
    claim = entry.get("claim") or {}
    seen: dict[str, int] = {}
    return [sentence(r, claim, spans, store, tz, seen) for r in entry.get("reasons") or []]
