"""The deterministic citation checker (CONTRACTS.md v1.3 section 7).

Pure: no network, no LLM, no clock. Same inputs give byte-identical model_dump_json().
Strict: a claim with any reason is rejected, and every reason is listed.
v1.3: CheckResult.entities lists the R4 entities of each claim as spans of its text, so a reader can tell the words
code checked (grounded or not) from the model's own wording, which is not verified.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from tbx import timeutil as tu
from tbx.models import Answer, CheckResult, Claim, EntitySpan, Reason, RejectedClaim
from tbx.store import CITABLE, Store

UNKNOWN_ARTIFACT = "UNKNOWN_ARTIFACT"
UNKNOWN_FIELD = "UNKNOWN_FIELD"
VALUE_MISMATCH = "VALUE_MISMATCH"
OUTSIDE_WINDOW = "OUTSIDE_WINDOW"
TIME_RANGE_UNSUPPORTED = "TIME_RANGE_UNSUPPORTED"
NO_CITATIONS = "NO_CITATIONS"
SLACK = timedelta(seconds=60)

TIMESTAMP_FIELDS = {"ts_utc", "ts_local"}
PHONE_FIELDS = {"number", "sender_number", "recipient_number"}
TEXT_FIELDS = {"body", "text", "transcript"}
COORD_FIELDS = {"lat", "lon"}
MEASURE_FIELDS = {"duration_s", "accuracy_m", "size_bytes"}
MIN_QUOTE = 8


# ------------------------------------------------------------------------------ normalization
def fold(s: object) -> str:
    """NFKC, casefold, whitespace collapsed and stripped."""
    return " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())


def digits(s: object) -> str:
    return re.sub(r"\D", "", str(s))


def phones_match(a: str, b: str) -> bool:
    da, db = digits(a), digits(b)
    if not da or not db:
        return False
    if len(da) >= 10 and len(db) >= 10:
        return da[-10:] == db[-10:]
    return da == db


def _float(s: object) -> float | None:
    try:
        v = float(str(s).strip())
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN never matches


def _num_close(cited: str, stored: object, tol: float) -> bool:
    a, b = _float(cited), _float(stored)
    return a is not None and b is not None and abs(a - b) <= tol


def ts_match(cited: str, stored_utc: str | None, case_tz: str) -> bool:
    if stored_utc is None:
        return False
    try:
        c, has_seconds = tu.parse_ts(cited, case_tz)
    except ValueError:
        return False
    s = tu.parse_utc(stored_utc)
    if has_seconds:
        return c.replace(microsecond=0) == s.replace(microsecond=0)
    return c.replace(second=0, microsecond=0) == s.replace(second=0, microsecond=0)


def value_matches(field: str, cited: str, row: dict, case_tz: str) -> bool:
    if field in TIMESTAMP_FIELDS:
        return ts_match(cited, row.get("ts_utc"), case_tz)
    stored = row.get(field)
    if stored is None:
        return False
    if field in PHONE_FIELDS:
        return phones_match(cited, stored)
    if field in TEXT_FIELDS:
        c, s = fold(cited), fold(stored)
        return c == s or (len(c) >= MIN_QUOTE and c in s)
    if field in COORD_FIELDS:
        return _num_close(cited, stored, 1e-4)
    if field == "score":
        return _num_close(cited, stored, 1e-3)
    if field in MEASURE_FIELDS:
        return _num_close(cited, stored, 1e-6)
    return fold(cited) == fold(stored)


# ------------------------------------------------------------------------------ entities (R4)
@dataclass(frozen=True)
class Entity:
    """One R4 entity. text == the claim text[start:end], trimmed (v1.3); key is the value R4 grounds: the cleaned
    match for most types (as in v1.2), the known contact's name for contacts."""
    type: str
    text: str
    start: int
    end: int
    key: str


_URL = re.compile(r"https?://[^\s<>\"'`]+", re.I)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
                        r"|[ ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2}))(?![\w:])")
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
           "nov": 11, "dec": 12}
_MONTH_NAMES = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
                r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b"
                   rf"|\b(?:{_MONTH_NAMES}|{_MONTH_NAMES.upper()})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}}\b)?")
_CLOCK = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp]\.?\s?[Mm]\b\.?)?(?![\d:])")
_COORD = re.compile(r"(?<![\d.])-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}(?!\d|\.\d)")
_PHONE = re.compile(r"\+?\(?\d[\d\s().\-]{5,}\d\)?")
_DECIMAL = re.compile(r"-?\d+\.\d+")
_BLANK = "\x00"
ORDER = ("url", "email", "timestamp", "date", "clock_time", "coordinate", "phone")
_RANK = {t: i for i, t in enumerate((*ORDER, "contact"))}
_PATTERNS = {"url": _URL, "email": _EMAIL, "timestamp": _TIMESTAMP, "date": _DATE, "clock_time": _CLOCK,
             "coordinate": _COORD, "phone": _PHONE}


def _trim(s: str, a: int, b: int, chars: str | None = None, left: bool = True) -> tuple[int, int]:
    """The bounds of s[a:b] after str.strip(chars) (str.rstrip(chars) when left is False); None means whitespace."""
    out = str.isspace if chars is None else chars.__contains__
    while left and a < b and out(s[a]):
        a += 1
    while b > a and out(s[b - 1]):
        b -= 1
    return a, b


SPAN_TRAILING = ".,;:!?"   # v1.3: a span never ends with these (nor with whitespace)


def _span_end(s: str, a: int, b: int) -> int:
    while b > a and (s[b - 1] in SPAN_TRAILING or s[b - 1].isspace()):
        b -= 1
    return b


def _clean(etype: str, s: str) -> tuple[int, int, str] | None:
    """A raw match -> (start, end, key): the trimmed span's bounds within s and the value R4 grounds, or None."""
    a, b = _trim(s, 0, len(s))
    if etype in ("url", "email"):
        a, b = _trim(s, a, b, ".,;:!?)]}'\"", left=False)
    if etype == "phone":
        a, b = _trim(s, a, b, " .-")
    key = s[a:b]
    if etype == "phone" and (len(digits(key)) < 7 or _DECIMAL.fullmatch(key.lstrip("+"))):
        return None
    if etype == "clock_time" and parse_clock(key) is None:
        return None
    if etype == "date" and parse_date(key) is None:
        return None
    if etype == "coordinate":
        lat, lon = (float(x) for x in key.split(","))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
    b = _span_end(s, a, b)
    return (a, b, key) if key and b > a else None


def contact_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(fold(name))}(?!\w)")


def _fold_map(text: str) -> tuple[str, list[int], list[int]]:
    """fold(text), plus the [start, end) in text of the characters each folded character came from.

    A character is normalized together with the combining marks after it, so offsets map back to whole characters."""
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    gap = False
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(text[j]):
            j += 1
        for ch in unicodedata.normalize("NFKC", text[i:j]).casefold():
            if ch.isspace():
                gap = True
                continue
            if gap and out:
                out.append(" ")
                starts.append(i)
                ends.append(i)
            gap = False
            out.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    return "".join(out), starts, ends


def extract_entities(text: str, contacts: set[str]) -> list[Entity]:
    """C1 order; a span taken by an earlier type is blanked before later types scan. Contacts come last, matched
    against the original text, by name (sorted), with one entity per whole-word occurrence (v1.3)."""
    out: list[Entity] = []
    work = text
    for etype in ORDER:
        for m in _PATTERNS[etype].finditer(work):
            cleaned = _clean(etype, m.group(0))
            if cleaned is None:
                continue
            a, b, key = cleaned
            out.append(Entity(etype, text[m.start() + a:m.start() + b], m.start() + a, m.start() + b, key))
            work = work[:m.start()] + _BLANK * (m.end() - m.start()) + work[m.end():]
    folded, starts, ends = _fold_map(text)
    for name in sorted(contacts):
        if len(name.strip()) < 3:
            continue
        for m in contact_pattern(name).finditer(folded):
            a, b = _trim(text, starts[m.start()], ends[m.end() - 1])
            b = _span_end(text, a, b)
            if b > a:
                out.append(Entity("contact", text[a:b], a, b, name))
    return out


def parse_clock(s: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:([AaPp])\.?\s?[Mm]\.?)?", s.strip())
    if not m:
        return None
    h, mi, mer = int(m.group(1)), int(m.group(2)), m.group(4)
    if mi > 59 or (m.group(3) and int(m.group(3)) > 59):
        return None
    if mer:
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if mer.lower() == "p" else 0)
    elif h > 23:
        return None
    return h, mi


def parse_date(s: str) -> tuple[int | None, int, int] | None:
    """-> (year or None, month, day)."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?", s)
        if not m:
            return None
        mo, d = _MONTHS[m.group(1)[:3].lower()], int(m.group(2))
        y = int(m.group(3)) if m.group(3) else None
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


@dataclass
class _Pool:
    """What a claim's valid citations can ground."""
    values: list[str]
    timestamps: list[datetime]
    coords: list[tuple[float, float]]


def _grounded(e: Entity, pool: _Pool, case_tz: str) -> bool:
    if e.type == "phone":
        cands = []
        for v in pool.values:
            cands.append(v)
            cands += [m.group(0) for m in _PHONE.finditer(v)]
        return any(phones_match(e.key, c) for c in cands)
    if e.type == "contact":
        pat = contact_pattern(e.key)
        return any(pat.search(fold(v)) for v in pool.values)
    if e.type == "date":
        y, mo, d = parse_date(e.key)  # type: ignore[misc]
        for ts in pool.timestamps:
            loc = tu.utc_to_local(ts, case_tz)
            if loc.month == mo and loc.day == d and (y is None or loc.year == y):
                return True
        return False
    if e.type == "clock_time":
        hm = parse_clock(e.key)
        return any(tu.case_hhmm(ts, case_tz) == hm for ts in pool.timestamps)
    if e.type == "coordinate":
        lat, lon = (float(x) for x in e.key.split(","))
        return any(abs(lat - a) <= 1e-4 and abs(lon - b) <= 1e-4 for a, b in pool.coords)
    needle = fold(e.key)  # url, email, timestamp: general rule (containment after folding)
    return any(needle in fold(v) for v in pool.values)


# ------------------------------------------------------------------------------ the check
def _check_claim(claim: Claim, store: Store, window: tu.Window | None, case_tz: str,
                 contacts: set[str]) -> tuple[list[Reason], list[EntitySpan]]:
    reasons: list[Reason] = []
    pool = _Pool([], [], [])
    support: list[datetime] = []
    coord_parts: dict[str, dict[str, float]] = {}

    if not claim.citations:
        reasons.append(Reason(code=NO_CITATIONS, detail="the claim cites no artifacts", citation_index=None))

    for i, c in enumerate(claim.citations):
        row = store.get(c.artifact_id)
        if row is None:
            reasons.append(Reason(code=UNKNOWN_ARTIFACT, detail=f"{c.artifact_id!r} is not in the store",
                                  citation_index=i))
            continue
        table = row["table"]
        ts = tu.parse_utc(row["ts_utc"]) if row.get("ts_utc") else None
        if ts is not None:
            support.append(ts)
            if table == "calls" and _float(row.get("duration_s")) is not None:
                support.append(ts + timedelta(seconds=_float(row["duration_s"])))  # type: ignore[arg-type]
        if c.field not in CITABLE[table]:
            reasons.append(Reason(code=UNKNOWN_FIELD, detail=f"{c.field!r} is not a citable field of {table}",
                                  citation_index=i))
        elif not value_matches(c.field, c.value, row, case_tz):
            shown = row.get("ts_utc") if c.field in TIMESTAMP_FIELDS else row.get(c.field)
            reasons.append(Reason(code=VALUE_MISMATCH, citation_index=i,
                                  detail=f"{c.field}: cited {c.value!r}, stored {shown!r}"))
        else:
            pool.values.append(c.value)
            if c.field in TIMESTAMP_FIELDS and ts is not None:
                pool.timestamps.append(ts)
            if c.field in COORD_FIELDS:
                coord_parts.setdefault(c.artifact_id, {})[c.field] = float(c.value)
        if window is not None:
            if ts is None:
                reasons.append(Reason(code=OUTSIDE_WINDOW, detail="no timestamp", citation_index=i))
            elif not window.contains(ts):
                reasons.append(Reason(code=OUTSIDE_WINDOW, citation_index=i,
                                      detail=f"{row['ts_utc']} is outside {tu.fmt_utc(window.start_utc)}"
                                             f"..{tu.fmt_utc(window.end_utc)}"))
    pool.coords = [(p["lat"], p["lon"]) for p in coord_parts.values() if {"lat", "lon"} <= set(p)]

    spans: list[EntitySpan] = []
    named: set[str] = set()
    for e in extract_entities(claim.text, contacts):
        grounded = _grounded(e, pool, case_tz)
        spans.append(EntitySpan(type=e.type, text=e.text, start=e.start, end=e.end, grounded=grounded))
        if e.type == "contact":       # one reason per contact name, however often it occurs (as in v1.2)
            if e.key in named:
                continue
            named.add(e.key)
        if not grounded:
            reasons.append(Reason(code=f"UNGROUNDED_ENTITY:{e.type}", citation_index=None,
                                  detail=f"{e.type} {e.key!r} is not supported by the claim's cited values"))
    spans.sort(key=lambda x: (x.start, x.end, _RANK[x.type], x.text))

    reasons += _check_range(claim, support, case_tz)
    return reasons, spans


def _check_range(claim: Claim, support: list[datetime], case_tz: str) -> list[Reason]:
    bounds: dict[str, datetime] = {}
    out: list[Reason] = []
    for name in ("time_start", "time_end"):
        raw = getattr(claim, name)
        if raw is None:
            continue
        try:
            bounds[name] = tu.parse_ts(raw, case_tz)[0]
        except ValueError:
            out.append(Reason(code=TIME_RANGE_UNSUPPORTED, detail=f"{name} {raw!r} is not an ISO time",
                              citation_index=None))
    if not bounds:
        return out
    if "time_start" in bounds and "time_end" in bounds and bounds["time_start"] > bounds["time_end"]:
        out.append(Reason(code=TIME_RANGE_UNSUPPORTED, detail="time_start is after time_end", citation_index=None))
    if not support:
        out.append(Reason(code=TIME_RANGE_UNSUPPORTED, citation_index=None,
                          detail="no cited artifact has a timestamp to support the time range"))
        return out
    lo, hi = min(support) - SLACK, max(support) + SLACK
    for name, dt in bounds.items():
        if not lo <= dt <= hi:
            out.append(Reason(code=TIME_RANGE_UNSUPPORTED, citation_index=None,
                              detail=f"{name} {tu.fmt_utc(dt)} is outside the cited support "
                                     f"{tu.fmt_utc(lo)}..{tu.fmt_utc(hi)}"))
    return out


def check(answer: Answer, store: Store, window: tu.Window | None) -> CheckResult:
    case_tz = store.case_tz
    contacts = store.known_contacts()
    result = CheckResult()
    for claim in answer.claims:
        reasons, spans = _check_claim(claim, store, window, case_tz, contacts)
        result.entities.append(spans)
        if reasons:
            result.rejected.append(RejectedClaim(claim=claim.model_copy(deep=True), reasons=reasons))
        else:
            result.verified.append(claim.model_copy(deep=True))
    return result
