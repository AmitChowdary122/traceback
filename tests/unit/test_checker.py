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


# ------------------------------------------------------------------------------ v1.3 entities (spans)
def spans(res, i=0):
    return [(e.type, e.text, e.start, e.end, e.grounded) for e in res.entities[i]]


def test_entity_spans_are_trimmed_offsets_into_the_claim_text(store):
    text = "Liz Dehner called at 12:05 a.m. on March 14."
    res = run(store, text, (CALL, "ts_local", "2023-03-14T00:05:00-04:00"), (CALL, "contact_name", "Liz Dehner"))
    assert res.verified
    got = spans(res)
    assert got == [("contact", "Liz Dehner", 0, 10, True), ("clock_time", "12:05 a.m", 21, 30, True),
                   ("date", "March 14", 35, 43, True)]
    for _, t, a, b, _ in got:
        assert text[a:b] == t


def test_contact_span_maps_back_through_case_and_whitespace_folding(store):
    text = "Then LIZ   DEHNER called."
    res = run(store, text, (CALL, "contact_name", "Liz Dehner"))
    assert res.verified
    assert spans(res) == [("contact", "LIZ   DEHNER", 5, 17, True)]


def test_one_reason_per_contact_name_but_every_occurrence_is_a_span(store):
    res = run(store, "Liz Dehner called; Liz Dehner hung up.", (CALL, "direction", "incoming"))
    assert [r.code for r in res.rejected[0].reasons] == ["UNGROUNDED_ENTITY:contact"]  # unchanged from v1.2
    assert spans(res) == [("contact", "Liz Dehner", 0, 10, False), ("contact", "Liz Dehner", 19, 29, False)]


def test_entities_follow_answer_order_across_verified_and_rejected(store):
    a = Answer.model_validate({"claims": [
        {"text": "Bob called at 9:15 PM.", "citations": [{"artifact_id": CALL, "field": "direction", "value": "incoming"}]},
        {"text": "Liz Dehner called.", "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"}]},
        {"text": "No entities here.", "citations": []}]})
    res = checker.check(a, store, None)
    assert [len(x) for x in res.entities] == [1, 1, 0]
    assert spans(res, 0) == [("clock_time", "9:15 PM", 14, 21, False)]
    assert spans(res, 1) == [("contact", "Liz Dehner", 0, 10, True)]


def test_overlapping_contact_spans_sort_by_start_then_end(store):
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}"}
    store.insert("calls", {"artifact_id": artifact_id("calls", "calls.tsv", "2"), "source_path": "calls.tsv",
                           "source_row": "2", "ts_utc": None, **prov, "contact_name": "Liz"})
    res = run(store, "Liz Dehner called.", (CALL, "contact_name", "Liz Dehner"))
    assert spans(res) == [("contact", "Liz", 0, 3, True), ("contact", "Liz Dehner", 0, 10, True)]


def test_entities_round_trip_and_old_results_still_load(store):
    from tbx.models import CheckResult
    res = run(store, "Liz Dehner called 919-555-1234.", (CALL, "number", "9195551234"))
    again = CheckResult.model_validate(json.loads(res.model_dump_json()))
    assert again == res and again.model_dump_json() == res.model_dump_json()
    assert CheckResult.model_validate({"verified": [], "rejected": []}).entities == []   # traces recorded before v1.3


@pytest.mark.parametrize("s", ["Liz  Dehner", "\tLiz Dehner\n", "ﬁle Straße", "José Ñuñez", "ＬＩＺ dehner",
                               "İstanbul", "a​b", "", "   ", "x — y", "📞 Liz Dehner called"])
def test_offset_tracking_fold_equals_fold(s):
    folded, starts, ends = checker._fold_map(s)
    assert folded == checker.fold(s)
    assert len(starts) == len(ends) == len(folded)
    assert all(0 <= a <= b <= len(s) for a, b in zip(starts, ends))


def test_contact_span_after_an_emoji_uses_code_point_offsets(store):
    text = "📞 Liz Dehner called."
    res = run(store, text, (CALL, "contact_name", "Liz Dehner"))
    (e,) = res.entities[0]
    assert (e.start, e.end, text[e.start:e.end]) == (2, 12, "Liz Dehner")
