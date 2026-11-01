"""tbx.timeutil: the one module that converts local <-> UTC (CONTRACTS.md section 3)."""
from datetime import datetime, timedelta, timezone

import pytest

from tbx import timeutil as tu

NY = "America/New_York"
UTC = timezone.utc


def test_fmt_utc_is_second_precision_with_z():
    dt = datetime(2023, 3, 14, 1, 42, 7, 999_999, tzinfo=UTC)
    assert tu.fmt_utc(dt) == "2023-03-14T01:42:07Z"


def test_fmt_utc_converts_offsets_to_utc():
    dt = datetime(2023, 3, 13, 21, 42, 7, tzinfo=timezone(timedelta(hours=-4)))
    assert tu.fmt_utc(dt) == "2023-03-14T01:42:07Z"


def test_fmt_utc_rejects_naive():
    with pytest.raises(ValueError):
        tu.fmt_utc(datetime(2023, 3, 14, 1, 42, 7))


@pytest.mark.parametrize("value,expected,has_seconds", [
    ("2023-03-14T01:42:07Z", "2023-03-14T01:42:07Z", True),
    ("2023-03-13T21:42:07-04:00", "2023-03-14T01:42:07Z", True),
    ("2023-03-14T01:42Z", "2023-03-14T01:42:00Z", False),
    ("2023-03-13T21:42:07", "2023-03-14T01:42:07Z", True),      # no offset -> case tz (EDT)
    ("2023-03-13 21:42", "2023-03-14T01:42:00Z", False),
    ("2023-03-14T01:42:07.250Z", "2023-03-14T01:42:07Z", True),
    ("2023-03-14T01:42:07+00:00", "2023-03-14T01:42:07Z", True),
])
def test_parse_ts(value, expected, has_seconds):
    dt, secs = tu.parse_ts(value, NY)
    assert tu.fmt_utc(dt) == expected and secs is has_seconds


@pytest.mark.parametrize("bad", ["", "not-a-time", "2023-13-01T00:00:00Z", "21:42"])
def test_parse_ts_rejects_garbage(bad):
    with pytest.raises(ValueError):
        tu.parse_ts(bad, NY)


def test_local_iso_uses_case_tz_with_offset_across_dst():
    assert tu.local_iso("2023-03-14T01:42:07Z", NY) == "2023-03-13T21:42:07-04:00"   # EDT
    assert tu.local_iso("2023-03-11T01:42:07Z", NY) == "2023-03-10T20:42:07-05:00"   # EST
    assert tu.local_iso(None, NY) is None


def test_local_to_utc_is_the_one_conversion():
    assert tu.fmt_utc(tu.local_to_utc(datetime(2023, 3, 13, 21, 0), NY)) == "2023-03-14T01:00:00Z"
    assert tu.fmt_utc(tu.local_to_utc(datetime(2023, 1, 13, 21, 0), NY)) == "2023-01-14T02:00:00Z"


def test_parse_user_time_case_local_and_explicit():
    assert tu.fmt_utc(tu.parse_user_time("2023-03-13 21:00", NY)) == "2023-03-14T01:00:00Z"
    assert tu.fmt_utc(tu.parse_user_time("2023-03-14T01:00:00Z", NY)) == "2023-03-14T01:00:00Z"
    assert tu.fmt_utc(tu.parse_user_time("2023-03-13T21:00:00-05:00", NY)) == "2023-03-14T02:00:00Z"
    with pytest.raises(ValueError):
        tu.parse_user_time("tonight", NY)


def test_window_inclusive_and_utc_only():
    w = tu.Window(datetime(2023, 3, 14, 1, tzinfo=UTC), datetime(2023, 3, 14, 3, tzinfo=UTC))
    assert w.contains(datetime(2023, 3, 14, 1, tzinfo=UTC))
    assert w.contains(datetime(2023, 3, 14, 3, tzinfo=UTC))
    assert not w.contains(datetime(2023, 3, 14, 3, 0, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        tu.Window(datetime(2023, 3, 14, 1), datetime(2023, 3, 14, 3, tzinfo=UTC))
    with pytest.raises(ValueError):
        tu.Window(datetime(2023, 3, 14, 3, tzinfo=UTC), datetime(2023, 3, 14, 1, tzinfo=UTC))


def test_window_from_user_strings_and_back():
    w = tu.Window.from_user("2023-03-13 21:00", "2023-03-13 23:00", NY)
    assert (tu.fmt_utc(w.start_utc), tu.fmt_utc(w.end_utc)) == ("2023-03-14T01:00:00Z", "2023-03-14T03:00:00Z")
    assert w.as_dict() == {"start_utc": "2023-03-14T01:00:00Z", "end_utc": "2023-03-14T03:00:00Z"}


def test_epoch_conversions():
    assert tu.fmt_utc(tu.from_epoch(1678758127, "s")) == "2023-03-14T01:42:07Z"
    assert tu.fmt_utc(tu.from_epoch(1678758127000, "ms")) == "2023-03-14T01:42:07Z"
    with pytest.raises(ValueError):
        tu.from_epoch(1, "us")


def test_case_local_date_and_clock():
    dt = datetime(2023, 3, 14, 1, 42, 7, tzinfo=UTC)
    assert tu.case_date(dt, NY) == "2023-03-13"
    assert tu.case_hhmm(dt, NY) == (21, 42)


def test_valid_tz():
    assert tu.valid_tz(NY) and not tu.valid_tz("Mars/Olympus")
