"""tbx.checker beyond the red-team suite: entity extraction, normalization, edge inputs."""
import json

import pytest

from tbx import checker
from tbx.models import Answer
from tbx.store import Store, artifact_id
from tbx.timeutil import Window, parse_utc

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
LOC = artifact_id("locations", "loc.tsv", "1")
MSG = artifact_id("messages", "sms.tsv", "1")


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    s.set_meta("device_tz", NY)
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}"}
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2023-03-14T04:05:00Z", **prov, "app": "Phone", "direction": "incoming",
                       "number": "9195551234", "contact_name": "Liz Dehner", "duration_s": 61})
    s.insert("locations", {"artifact_id": LOC, "source_path": "loc.tsv", "source_row": "1",
                           "ts_utc": "2023-03-14T16:30:00Z", **prov, "source": "wifi",
                           "lat": 35.591383, "lon": -78.775345, "accuracy_m": 5})
    s.insert("messages", {"artifact_id": MSG, "source_path": "sms.tsv", "source_row": "1",
                          "ts_utc": "2023-03-14T16:31:00Z", **prov, "app": "SMS", "thread_id": "1",
                          "direction": "incoming", "sender": "Liz Dehner", "sender_number": "9195551234",
                          "body": "Call 919-555-0000 or 919 555 1234 after 12:30 PM"})
    return s


def run(store, text, *cites, window=None, **kw):
    a = Answer.model_validate({"claims": [{"text": text, "citations": [
        {"artifact_id": i, "field": f, "value": v} for i, f, v in cites], **kw}]})
    return checker.check(a, store, window)


def codes(res):
    return sorted({r.code for r in res.rejected[0].reasons}) if res.rejected else []


def test_extraction_order_removes_spans():
    ents = checker.extract_entities("At 2023-03-14T01:42:07Z see https://x.example/a?b=2023-03-14 or a@b.org", set())
    assert [(e.type, e.text) for e in ents] == [("url", "https://x.example/a?b=2023-03-14"), ("email", "a@b.org"),
                                               ("timestamp", "2023-03-14T01:42:07Z")]


def test_decimal_numbers_are_not_phones():
    ents = checker.extract_entities("longitude -78.7753450 and score 0.97260272", set())
    assert [e.type for e in ents] == []


def test_coordinate_pair_is_not_a_phone():
    ents = checker.extract_entities("at 35.591383, -78.775345 today", set())
    assert [e.type for e in ents] == ["coordinate"]


def test_clock_time_12h_and_24h():
    assert checker.parse_clock("12:05 AM") == (0, 5)
    assert checker.parse_clock("12:30 p.m.") == (12, 30)
    assert checker.parse_clock("21:42") == (21, 42)
    assert checker.parse_clock("13:00 PM") is None
    assert checker.parse_clock("25:00") is None


def test_midnight_clock_time_grounded_in_case_tz(store):
    res = run(store, "Liz Dehner called at 12:05 AM on March 14.", (CALL, "ts_local", "2023-03-14T00:05:00-04:00"),
              (CALL, "contact_name", "Liz Dehner"))
    assert res.verified and not res.rejected


def test_multiword_contact_must_be_cited(store):
    res = run(store, "Liz Dehner called.", (CALL, "direction", "incoming"))
    assert codes(res) == ["UNGROUNDED_ENTITY:contact"]


def test_phone_grounded_inside_cited_body(store):
    res = run(store, "The message mentioned 919-555-0000.", (MSG, "body", "Call 919-555-0000 or 919 555 1234"))
    assert res.verified and not res.rejected


def test_coordinates_need_same_artifact_lat_and_lon(store):
    ok = run(store, "Phone at 35.5914, -78.7753.", (LOC, "lat", "35.591383"), (LOC, "lon", "-78.775345"))
    assert ok.verified
    bad = run(store, "Phone at 35.5914, -78.7753.", (LOC, "lat", "35.591383"))
    assert codes(bad) == ["UNGROUNDED_ENTITY:coordinate"]


def test_unparseable_time_range_is_rejected(store):
    res = run(store, "A call came in.", (CALL, "direction", "incoming"), time_start="tonight")
    assert codes(res) == ["TIME_RANGE_UNSUPPORTED"]


def test_only_time_end_checked(store):
    res = run(store, "A call came in.", (CALL, "direction", "incoming"), time_end="2023-03-14T04:07:01Z")
    assert res.verified  # 04:05:00 + 61 s + 60 s slack = 04:07:01, inclusive


def test_time_range_with_no_timestamped_citation(store, tmp_path):
    res = run(store, "Something happened.", ("call_ffffffffff", "direction", "incoming"),
              time_start="2023-03-14T04:05:00Z")
    assert set(codes(res)) == {"UNKNOWN_ARTIFACT", "TIME_RANGE_UNSUPPORTED"}


def test_null_stored_value_never_matches(store):
    res = run(store, "Message recipient.", (MSG, "recipient", "None"))
    assert codes(res) == ["VALUE_MISMATCH"]


def test_mismatched_citation_does_not_ground_entities(store):
    res = run(store, "Liz Dehner called.", (CALL, "contact_name", "Liz Dehner "), (CALL, "direction", "outgoing"))
    assert codes(res) == ["VALUE_MISMATCH"]   # name cited correctly (whitespace folded)
    res = run(store, "Bob Smith called.", (CALL, "contact_name", "Bob Smith"))
    assert codes(res) == ["VALUE_MISMATCH"]   # not a known contact, so no R4 entity


def test_result_json_is_byte_identical(store):
    w = Window(parse_utc("2023-03-14T00:00:00Z"), parse_utc("2023-03-15T00:00:00Z"))
    a = run(store, "Liz Dehner called 919-555-1234 at 12:05 AM.", (CALL, "number", "919 555 1234"), window=w)
    b = run(store, "Liz Dehner called 919-555-1234 at 12:05 AM.", (CALL, "number", "919 555 1234"), window=w)
    assert a.model_dump_json() == b.model_dump_json()
    json.loads(a.model_dump_json())


def test_transcript_follows_the_text_rule(tmp_path):
    s = Store(str(tmp_path / "vm.db"))
    s.set_meta("device_tz", NY)
    vm = artifact_id("calls", "calls.tsv", "9")
    s.insert("calls", {"artifact_id": vm, "source_path": "calls.tsv", "source_row": "9",
                       "ts_utc": "2024-02-01T15:00:00Z", "ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}",
                       "app": "Phone", "direction": "voicemail", "number": "+18005551212",
                       "transcript": "Hi,  your U-Haul\nreservation is ready."})
    ok = run(s, "A voicemail mentioned a reservation.", (vm, "transcript", "u-haul reservation IS READY"))
    assert ok.verified and not ok.rejected                     # casefold + whitespace collapse + substring
    short = run(s, "A voicemail mentioned a reservation.", (vm, "transcript", "U-Haul"))
    assert codes(short) == ["VALUE_MISMATCH"]                  # substrings need >= 8 characters
    assert codes(run(s, "A call was missed.", (vm, "direction", "missed"))) == ["VALUE_MISMATCH"]
