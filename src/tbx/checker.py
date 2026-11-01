"""The deterministic citation checker (CONTRACTS.md v1.2 section 7).

Pure: no network, no LLM, no clock. Same inputs give byte-identical model_dump_json().
Strict: a claim with any reason is rejected, and every reason is listed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta

from tbx import timeutil as tu
from tbx.models import Answer, CheckResult, Claim, Reason, RejectedClaim
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
    type: str
    text: str


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
_PATTERNS = {"url": _URL, "email": _EMAIL, "timestamp": _TIMESTAMP, "date": _DATE, "clock_time": _CLOCK,
             "coordinate": _COORD, "phone": _PHONE}


def _clean(etype: str, s: str) -> str | None:
    s = s.strip()
    if etype in ("url", "email"):
        s = s.rstrip(".,;:!?)]}'\"")
    if etype == "phone":
        s = s.strip(" .-")
        if len(digits(s)) < 7 or _DECIMAL.fullmatch(s.lstrip("+")):
            return None
    if etype == "clock_time" and parse_clock(s) is None:
        return None
    if etype == "date" and parse_date(s) is None:
        return None
    if etype == "coordinate":
        lat, lon = (float(x) for x in s.split(","))
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
    return s or None


def contact_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(fold(name))}(?!\w)")


def extract_entities(text: str, contacts: set[str]) -> list[Entity]:
    """C1 order; a span taken by an earlier type is blanked before later types scan."""
    out: list[Entity] = []
    work = text
    for etype in ORDER:
        for m in _PATTERNS[etype].finditer(work):
            val = _clean(etype, m.group(0))
            if val is None:
                continue
            out.append(Entity(etype, val))
            work = work[:m.start()] + _BLANK * (m.end() - m.start()) + work[m.end():]
    folded = fold(text)
    for name in sorted(contacts):
        if len(name.strip()) >= 3 and contact_pattern(name).search(folded):
            out.append(Entity("contact", name))
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
        return any(phones_match(e.text, c) for c in cands)
    if e.type == "contact":
        pat = contact_pattern(e.text)
        return any(pat.search(fold(v)) for v in pool.values)
    if e.type == "date":
        y, mo, d = parse_date(e.text)  # type: ignore[misc]
        for ts in pool.timestamps:
            loc = tu.utc_to_local(ts, case_tz)
            if loc.month == mo and loc.day == d and (y is None or loc.year == y):
                return True
        return False
    if e.type == "clock_time":
        hm = parse_clock(e.text)
        return any(tu.case_hhmm(ts, case_tz) == hm for ts in pool.timestamps)
    if e.type == "coordinate":
        lat, lon = (float(x) for x in e.text.split(","))
        return any(abs(lat - a) <= 1e-4 and abs(lon - b) <= 1e-4 for a, b in pool.coords)
    needle = fold(e.text)  # url, email, timestamp: general rule (containment after folding)
    return any(needle in fold(v) for v in pool.values)


# ------------------------------------------------------------------------------ the check
def _check_claim(claim: Claim, store: Store, window: tu.Window | None, case_tz: str,
                 contacts: set[str]) -> list[Reason]:
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

    for e in extract_entities(claim.text, contacts):
        if not _grounded(e, pool, case_tz):
            reasons.append(Reason(code=f"UNGROUNDED_ENTITY:{e.type}", citation_index=None,
                                  detail=f"{e.type} {e.text!r} is not supported by the claim's cited values"))

    reasons += _check_range(claim, support, case_tz)
    return reasons


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
        reasons = _check_claim(claim, store, window, case_tz, contacts)
        if reasons:
            result.rejected.append(RejectedClaim(claim=claim.model_copy(deep=True), reasons=reasons))
        else:
            result.verified.append(claim.model_copy(deep=True))
    return result
