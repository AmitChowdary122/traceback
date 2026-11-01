"""The one module that converts between case-local time and UTC (CONTRACTS.md section 3).

Every other module calls these functions; nothing else touches zoneinfo or offsets.
- Stored times are `ts_utc` strings: ISO 8601, UTC, second precision, `Z` suffix.
- A value without an offset is read in the case timezone (`case_meta.device_tz`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc
_SECONDS = re.compile(r"T?\d{1,2}:\d{2}:\d{2}")
_DATE_TIME = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?", re.I)


@lru_cache(maxsize=64)
def zone(tz: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"unknown timezone: {tz!r}") from e


def valid_tz(tz: str) -> bool:
    try:
        zone(tz)
        return True
    except ValueError:
        return False


def _require_aware(dt: datetime) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive datetime not allowed here: {dt!r}")


def fmt_utc(dt: datetime) -> str:
    """Aware datetime -> 'YYYY-MM-DDTHH:MM:SSZ' (sub-seconds truncated)."""
    _require_aware(dt)
    return dt.astimezone(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_to_utc(naive: datetime, tz: str) -> datetime:
    """THE local -> UTC conversion. Ambiguous wall times (DST fall-back) take the first occurrence."""
    if naive.tzinfo is not None:
        return naive.astimezone(UTC)
    return naive.replace(tzinfo=zone(tz), fold=0).astimezone(UTC)


def utc_to_local(dt: datetime, tz: str) -> datetime:
    _require_aware(dt)
    return dt.astimezone(zone(tz))


def parse_ts(value: str, case_tz: str) -> tuple[datetime, bool]:
    """Parse an ISO date-time (Z, offset, or none = case tz). Returns (UTC datetime, had_seconds)."""
    s = (value or "").strip()
    if not _DATE_TIME.fullmatch(s) or len(s) <= 10:
        raise ValueError(f"not an ISO date-time: {value!r}")
    try:
        dt = datetime.fromisoformat(s[:-1] + "+00:00" if s[-1] in "zZ" else s)
    except ValueError as e:
        raise ValueError(f"not an ISO date-time: {value!r}") from e
    has_seconds = bool(_SECONDS.search(s))
    if dt.tzinfo is None:
        return local_to_utc(dt, case_tz), has_seconds
    return dt.astimezone(UTC), has_seconds


def parse_utc(value: str) -> datetime:
    """Parse a stored ts_utc string."""
    dt, _ = parse_ts(value, "UTC")
    return dt


def parse_user_time(value: str, case_tz: str) -> datetime:
    """CLI/UI time: 'YYYY-MM-DD HH:MM' in the case tz, or ISO with Z/offset. Returns UTC."""
    return parse_ts(value, case_tz)[0]


def local_iso(ts_utc: str | datetime | None, tz: str) -> str | None:
    """ts_utc rendered in the case tz with its offset, e.g. 2023-03-13T21:42:07-04:00."""
    if ts_utc is None:
        return None
    dt = parse_utc(ts_utc) if isinstance(ts_utc, str) else ts_utc
    return utc_to_local(dt, tz).replace(microsecond=0).isoformat()


def from_epoch(value: int | float | str, unit: str) -> datetime:
    """Epoch seconds ('s') or milliseconds ('ms') -> aware UTC datetime."""
    v = float(value)
    if unit == "s":
        return datetime.fromtimestamp(v, UTC)
    if unit == "ms":
        return datetime.fromtimestamp(v / 1000.0, UTC)
    raise ValueError(f"unknown epoch unit: {unit!r}")


def case_date(dt: datetime, tz: str) -> str:
    return utc_to_local(dt, tz).strftime("%Y-%m-%d")


def case_hhmm(dt: datetime, tz: str) -> tuple[int, int]:
    loc = utc_to_local(dt, tz)
    return loc.hour, loc.minute


def now_utc() -> str:
    return fmt_utc(datetime.now(UTC))


@dataclass(frozen=True)
class Window:
    """A question window. Both ends are aware datetimes in UTC, both inclusive."""

    start_utc: datetime
    end_utc: datetime

    def __post_init__(self):
        _require_aware(self.start_utc)
        _require_aware(self.end_utc)
        object.__setattr__(self, "start_utc", self.start_utc.astimezone(UTC))
        object.__setattr__(self, "end_utc", self.end_utc.astimezone(UTC))
        if self.start_utc > self.end_utc:
            raise ValueError("window start is after its end")

    @classmethod
    def from_user(cls, start: str, end: str, case_tz: str) -> "Window":
        return cls(parse_user_time(start, case_tz), parse_user_time(end, case_tz))

    def contains(self, dt: datetime) -> bool:
        return self.start_utc <= dt <= self.end_utc

    def as_dict(self) -> dict:
        return {"start_utc": fmt_utc(self.start_utc), "end_utc": fmt_utc(self.end_utc)}
