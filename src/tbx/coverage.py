"""What the agent searched, in plain English, from the trace's tool_call / tool_result events.

For each evidence family: every search with its time range, filters, how many rows matched (total), how many the
model saw (returned) and whether the list was cut at the limit; then which families were never searched. Direct
lookups by ID (get_artifact) are counted apart. Pure: the trace and the case timezone only.
"""
from __future__ import annotations

from datetime import datetime

from tbx import timeutil as tu

FAMILIES = (("messages", "search_messages", "Messages"), ("calls", "get_calls", "Calls and voicemail"),
            ("locations", "get_locations", "Locations"), ("media", "find_media", "Photos and media"),
            ("transcripts", "get_transcripts", "Voice transcripts"))
LOOKUP = "get_artifact"
DEFAULT_LIMIT, MAX_LIMIT = 50, 200
_FILTERS = (("query", "matching '{}'"), ("contact", "for contact '{}'"), ("number", "for number {}"),
            ("app", "in app {}"), ("direction", "direction {}"), ("source", "from source {}"), ("kind", "of kind {}"),
            ("tag", "tagged '{}'"), ("media_id", "for media {}"))
_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _when(dt: datetime, tz: str) -> tuple[str, str]:
    loc = tu.utc_to_local(dt, tz)
    return (f"{_MON[loc.month - 1]} {loc.day}, {loc.year}",
            f"{(loc.hour + 11) % 12 + 1}:{loc.minute:02d} {'AM' if loc.hour < 12 else 'PM'}")


def short_time(dt: datetime, tz: str) -> str:
    d, c = _when(dt, tz)
    return f"{d} {c}"


def short_range(a: datetime, b: datetime, tz: str) -> str:
    (da, ca), (db, cb) = _when(a, tz), _when(b, tz)
    return f"{da} {ca} – {cb}" if da == db else f"{da} {ca} – {db} {cb}"


def _time(v: object, tz: str) -> datetime | None:
    try:
        return tu.parse_ts(str(v), tz)[0] if v not in (None, "") else None
    except ValueError:
        return None


def _limit(args: dict) -> int:
    try:
        return max(1, min(MAX_LIMIT, int(args.get("limit", DEFAULT_LIMIT))))
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _pairs(events: list[dict]) -> list[dict]:
    """tool_call + its tool_result, in order (tbx.agent writes them back to back). A call still running is skipped."""
    out, pending = [], []
    for e in events:
        p = e.get("payload") or {}
        if e.get("type") == "tool_call":
            pending.append({"step": e.get("step"), "tool": p.get("name"), "args": p.get("args") or {}})
        elif e.get("type") == "tool_result" and pending:
            c = pending.pop(0)
            out.append({**c, "total": int(p.get("total") or 0), "returned": int(p.get("returned") or 0),
                        "truncated": bool(p.get("truncated")), "error": p.get("error")})
    return out


class _Call:
    def __init__(self, raw: dict, window: tu.Window | None, tz: str):
        self.raw, self.tz = raw, tz
        args = raw["args"] if isinstance(raw["args"], dict) else {}
        self.start, self.end = _time(args.get("start"), tz), _time(args.get("end"), tz)
        self.window = window
        self.filters = tuple(fmt.format(args[k]) for k, fmt in _FILTERS if args.get(k) not in (None, ""))
        self.limit = _limit(args)

    @property
    def is_window(self) -> bool:
        w = self.window
        return w is not None and self.start == w.start_utc and self.end == w.end_utc

    def scope(self) -> str:
        if self.start is None and self.end is None:
            return "in the whole extraction"
        if self.is_window:
            return "in the window"
        if self.start and self.end:
            return short_range(self.start, self.end, self.tz)
        return f"from {short_time(self.start, self.tz)} on" if self.start else f"up to {short_time(self.end, self.tz)}"

    def range_text(self) -> str:
        s = self.scope()
        return s.removeprefix("in ") if s.startswith("in the") else s

    def key(self) -> tuple:
        return self.start, self.end, self.filters

    def inside(self, other: "_Call") -> bool:
        """This search's time range lies within the other's (and is not the same)."""
        lo = other.start or datetime.min.replace(tzinfo=tu.UTC)
        hi = other.end or datetime.max.replace(tzinfo=tu.UTC)
        a = self.start or datetime.min.replace(tzinfo=tu.UTC)
        b = self.end or datetime.max.replace(tzinfo=tu.UTC)
        return lo <= a and b <= hi and (a, b) != (lo, hi)

    def row(self) -> dict:
        r = self.raw
        if self.start or self.end:
            if self.start and self.end:
                window = short_range(self.start, self.end, self.tz)
            else:
                window = self.scope()
        else:
            window = "all time"
        return {"tool": r["tool"], "step": r["step"], "window": window, "filters": ", ".join(self.filters),
                "limit": self.limit, "total": r["total"], "returned": r["returned"], "truncated": r["truncated"],
                "error": r["error"]}


def _first(c: _Call) -> str:
    total, returned = c.raw["total"], c.raw["returned"]
    what = " ".join([*([", ".join(c.filters)] if c.filters else []), c.scope()])
    if total == 0:
        return f"none {what}"
    shown = ("all shown" if total > 1 else "shown") if returned >= total else f"{returned} shown"
    return f"{total} {what}, {shown}"


def _later(c: _Call, earlier: list[_Call]) -> str:
    total, returned = c.raw["total"], c.raw["returned"]
    same = [e for e in earlier if e.key() == c.key()]
    if same:
        if returned >= total and any(e.raw["truncated"] for e in same):
            return f"then all {total} with a higher limit"
        if returned >= total:
            return "then the same search again"
        return f"then {returned} of {total} with limit {c.limit}"
    result = "none" if total == 0 else (f"all {total}" if returned >= total else f"{returned} of {total}")
    what = ", ".join([*c.filters]) if c.key()[:2] == earlier[-1].key()[:2] else \
        " ".join([*([", ".join(c.filters)] if c.filters else []), c.range_text()])
    narrowed = any(c.inside(e) or (c.key()[:2] == e.key()[:2] and set(e.filters) < set(c.filters)) for e in earlier)
    return f"then {'narrowed to ' if narrowed else ''}{what}: {result}"


def _sentence(label: str, calls: list[_Call]) -> str:
    ok = [c for c in calls if not c.raw["error"]]
    failed = [c for c in calls if c.raw["error"]]
    if not ok:
        if failed:
            return f"{label}: the search failed ({failed[0].raw['error']})."
        return f"{label}: never searched."
    parts = [_first(ok[0])] + [_later(c, ok[:i]) for i, c in enumerate(ok[1:], 1)]
    if failed:
        parts.append(f"{len(failed)} search{'es' if len(failed) > 1 else ''} failed")
    return f"{label}: {'; '.join(parts)}."


def coverage(events: list[dict], case_tz: str) -> dict:
    """-> {"summary", "families": [{family, label, tool, searched, sentence, calls: [...]}], "lookups": {...}}."""
    start = next((e.get("payload") or {} for e in events if e.get("type") == "run_start"), {})
    w = start.get("window")
    window = tu.Window(tu.parse_utc(w["start_utc"]), tu.parse_utc(w["end_utc"])) if w else None
    calls = _pairs(events)
    families = []
    for family, tool, label in FAMILIES:
        mine = [_Call(c, window, case_tz) for c in calls if c["tool"] == tool]
        sentence = _sentence(label, mine)
        families.append({"family": family, "label": label, "tool": tool,
                         "searched": any(not c.raw["error"] for c in mine), "sentence": sentence,
                         "text": sentence.removeprefix(f"{label}: "), "calls": [c.row() for c in mine]})
    lookups = [c for c in calls if c["tool"] == LOOKUP]
    not_found = sum(1 for c in lookups if c["returned"] == 0)
    searched = [f for f in families if f["searched"]]
    never = [f["label"].lower() for f in families if not f["searched"] and not f["calls"]]
    failed = [f["label"].lower() for f in families if not f["searched"] and f["calls"]]
    if not searched:
        summary = "No evidence family was searched."
    else:
        summary = f"Searched {len(searched)} of {len(families)} evidence families."
        if never:
            summary += f" Never searched: {', '.join(never)}."
        if failed:
            summary += f" The search of {', '.join(failed)} failed."
    if lookups:
        summary += (f" Opened {len(lookups)} record{'s' if len(lookups) != 1 else ''} directly by ID"
                    + (f" ({not_found} not found)." if not_found else "."))
    return {"summary": summary, "families": families,
            "lookups": {"count": len(lookups), "not_found": not_found}}
