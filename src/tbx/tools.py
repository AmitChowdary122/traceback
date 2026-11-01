"""The six typed tools the agent calls (CONTRACTS.md section 5).

Every tool returns {"rows": [...], "total": int, "truncated": bool}; bad arguments add an
"error" string instead of raising, so the model can correct itself.
"""
from __future__ import annotations

from typing import Any, Callable

from tbx import timeutil as tu
from tbx.checker import fold, phones_match
from tbx.store import FAMILY, Store

DEFAULT_LIMIT, MAX_LIMIT, LONG_FIELD = 50, 200, 300
LONG_FIELDS = ("body", "text", "transcript")
MEASURE = {"duration_s", "accuracy_m", "size_bytes"}
HIDDEN = {"raw_json", "tz_basis", "ts_original", "source_row"}  # not citable; kept out of list results


class ToolArgError(ValueError):
    pass


def clamp_limit(limit: Any) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    try:
        n = int(limit)
    except (TypeError, ValueError) as e:
        raise ToolArgError(f"limit must be an integer, got {limit!r}") from e
    return max(1, min(MAX_LIMIT, n))


def _register(store: Store) -> None:
    if getattr(store, "_tbx_funcs", False):
        return
    store.conn.create_function("tbx_phone", 2, lambda a, b: int(a is not None and phones_match(str(a), str(b))),
                               deterministic=True)
    store.conn.create_function("tbx_has", 2, lambda hay, needle: int(hay is not None and fold(needle) in fold(hay)),
                               deterministic=True)
    store.conn.create_function("tbx_eq", 2, lambda a, b: int(a is not None and fold(a) == fold(b)), deterministic=True)
    store._tbx_funcs = True  # type: ignore[attr-defined]


def _time(value: Any, name: str, case_tz: str) -> str | None:
    if value in (None, ""):
        return None
    try:
        return tu.fmt_utc(tu.parse_user_time(str(value), case_tz))
    except ValueError as e:
        raise ToolArgError(f"{name} must be an ISO time with Z or an offset, got {value!r}") from e


def _shape(row: dict, case_tz: str, full: bool = False) -> dict:
    out: dict[str, Any] = {"artifact_id": row["artifact_id"], "ts_utc": row["ts_utc"],
                           "ts_local": tu.local_iso(row["ts_utc"], case_tz)}
    for k, v in row.items():
        if k in out or k == "table" or k == "raw_json" or (not full and k in HIDDEN):
            continue
        if k in MEASURE and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    if not full:
        for f in LONG_FIELDS:
            if isinstance(out.get(f), str) and len(out[f]) > LONG_FIELD:
                out[f] = out[f][:LONG_FIELD]
                out[f"{f}_truncated"] = True
    return out


class _Q:
    """A tiny WHERE builder over one family table."""

    def __init__(self, table: str, case_tz: str):
        self.table, self.case_tz = table, case_tz
        self.where: list[str] = []
        self.params: list[Any] = []

    def add(self, sql: str, *params: Any) -> None:
        self.where.append(sql)
        self.params.extend(params)

    def window(self, start: Any, end: Any) -> None:
        s, e = _time(start, "start", self.case_tz), _time(end, "end", self.case_tz)
        if s:
            self.add("ts_utc >= ?", s)
        if e:
            self.add("ts_utc <= ?", e)

    def terms(self, query: Any, cols: tuple[str, ...]) -> None:
        for term in str(query or "").split():
            self.add("(" + " OR ".join(f"tbx_has({c}, ?)" for c in cols) + ")", *([term] * len(cols)))

    def run(self, store: Store, limit: Any) -> dict:
        n = clamp_limit(limit)
        where = f" WHERE {' AND '.join(self.where)}" if self.where else ""
        total = store.query(f"SELECT COUNT(*) AS n FROM {self.table}{where}", self.params)[0]["n"]
        rows = store.query(f"SELECT * FROM {self.table}{where} "
                           f"ORDER BY ts_utc IS NULL, ts_utc, artifact_id LIMIT ?", [*self.params, n])
        return {"rows": [_shape(r, self.case_tz) for r in rows], "total": total, "truncated": total > len(rows)}


def search_messages(store: Store, query=None, contact=None, number=None, app=None, start=None, end=None, limit=None):
    q = _Q("messages", store.case_tz)
    q.terms(query, ("body", "sender", "recipient"))
    if contact:
        q.add("(tbx_has(sender, ?) OR tbx_has(recipient, ?))", contact, contact)
    if number:
        q.add("(tbx_phone(sender_number, ?) OR tbx_phone(recipient_number, ?))", number, number)
    if app:
        q.add("tbx_has(app, ?)", app)
    q.window(start, end)
    return q.run(store, limit)


def get_calls(store: Store, query=None, number=None, contact=None, direction=None, start=None, end=None, limit=None):
    q = _Q("calls", store.case_tz)
    if query not in (None, ""):
        q.add("tbx_has(transcript, ?)", str(query))  # v1.2: one case-insensitive substring of the transcript
    if number:
        q.add("tbx_phone(number, ?)", number)
    if contact:
        q.add("tbx_has(contact_name, ?)", contact)
    if direction:
        q.add("tbx_eq(direction, ?)", direction)
    q.window(start, end)
    return q.run(store, limit)


def get_locations(store: Store, start=None, end=None, source=None, limit=None):
    q = _Q("locations", store.case_tz)
    if source:
        q.add("tbx_has(source, ?)", source)
    q.window(start, end)
    return q.run(store, limit)


def find_media(store: Store, kind=None, tag=None, start=None, end=None, limit=None):
    q = _Q("media", store.case_tz)
    if kind:
        q.add("tbx_eq(kind, ?)", kind)
    if tag:
        q.add("artifact_id IN (SELECT media_id FROM image_tags WHERE tbx_has(label, ?))", tag)
    q.window(start, end)
    out = q.run(store, limit)
    _attach_tags(store, out["rows"])
    return out


def _attach_tags(store: Store, rows: list[dict]) -> None:
    """Each media row lists its image_tags rows ({artifact_id, label, score}), so they can be cited (v1.2 enrich)."""
    ids = [r["artifact_id"] for r in rows]
    if not ids:
        return
    tags: dict[str, list[dict]] = {i: [] for i in ids}
    for t in store.query(f"SELECT artifact_id, media_id, label, score FROM image_tags WHERE media_id IN "
                         f"({', '.join('?' * len(ids))}) ORDER BY media_id, label, score DESC, artifact_id", ids):
        tags[t["media_id"]].append({"artifact_id": t["artifact_id"], "label": t["label"], "score": t["score"]})
    for r in rows:
        r["tags"] = tags[r["artifact_id"]]


def get_transcripts(store: Store, query=None, media_id=None, start=None, end=None, limit=None):
    q = _Q("transcripts", store.case_tz)
    q.terms(query, ("text",))
    if media_id:
        q.add("media_id = ?", media_id)
    q.window(start, end)
    return q.run(store, limit)


def get_artifact(store: Store, artifact_id=None):
    if not artifact_id:
        raise ToolArgError("artifact_id is required")
    row = store.get(str(artifact_id))
    if row is None:
        return {"rows": [], "total": 0, "truncated": False}
    return {"rows": [_shape(row, store.case_tz, full=True)], "total": 1, "truncated": False}


TOOLS: dict[str, Callable[..., dict]] = {f.__name__: f for f in (
    search_messages, get_calls, get_locations, find_media, get_transcripts, get_artifact)}


def run_tool(store: Store, name: str, args: dict | None) -> dict:
    fn = TOOLS.get(name)
    if fn is None:
        return {"rows": [], "total": 0, "truncated": False, "error": f"unknown tool {name!r}"}
    _register(store)
    params = fn.__code__.co_varnames[1:fn.__code__.co_argcount]
    clean = {k: v for k, v in (args or {}).items() if k in params}
    try:
        return fn(store, **clean)
    except ToolArgError as e:
        return {"rows": [], "total": 0, "truncated": False, "error": str(e)}


# ------------------------------------------------------------------------------ OpenAI tool specs
_TIME = {"type": "string", "description": "ISO 8601 time with Z or an offset, e.g. 2024-07-27T21:00:00-04:00"}
_LIMIT = {"type": "integer", "description": f"max rows (default {DEFAULT_LIMIT}, cap {MAX_LIMIT})"}


def _spec(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required or [], "additionalProperties": False}}}


_S = {"type": "string"}
TOOL_SPECS: list[dict] = [
    _spec("search_messages", "Search chat/SMS messages. Every word of `query` must appear in the body, sender or "
          "recipient (case-insensitive).",
          {"query": _S, "contact": {"type": "string", "description": "name; matches sender or recipient"},
           "number": {"type": "string", "description": "phone number in any format"},
           "app": _S, "start": _TIME, "end": _TIME, "limit": _LIMIT}),
    _spec("get_calls", "Call log entries (phone and app calls). Voicemails are their own direction, not missed "
          "calls; a voicemail row may carry the carrier's transcription in `transcript`.",
          {"query": {"type": "string", "description": "case-insensitive substring of the voicemail transcript"},
           "number": {"type": "string", "description": "phone number in any format"}, "contact": _S,
           "direction": {"type": "string",
                         "enum": ["incoming", "outgoing", "missed", "rejected", "voicemail", "unknown"]},
           "start": _TIME, "end": _TIME, "limit": _LIMIT}),
    _spec("get_locations", "Device location fixes (lat/lon) from any source.",
          {"start": _TIME, "end": _TIME, "source": _S, "limit": _LIMIT}),
    _spec("find_media", "Photos, audio and video files. `tag` filters by an image tag label (e.g. 'car'). Each row "
          "lists its image tags (SAM 3.1 detections: artifact_id, label, score); cite a tag's own artifact_id for "
          "what a photo shows. Spoken audio in videos is in get_transcripts.",
          {"kind": {"type": "string", "enum": ["image", "audio", "video", "other"]}, "tag": _S,
           "start": _TIME, "end": _TIME, "limit": _LIMIT}),
    _spec("get_transcripts", "Transcripts of voice notes and other audio.",
          {"query": _S, "media_id": _S, "start": _TIME, "end": _TIME, "limit": _LIMIT}),
    _spec("get_artifact", "One artifact's full row by artifact_id (full message or transcript text).",
          {"artifact_id": _S}, ["artifact_id"]),
]
assert {t["function"]["name"] for t in TOOL_SPECS} == set(TOOLS)
assert set(FAMILY) >= {"messages", "calls", "locations", "media", "transcripts", "image_tags"}
