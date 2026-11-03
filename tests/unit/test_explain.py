"""tbx.explain: one plain-English sentence per checker reason, from fixtures (no demo runs)."""
import pytest

from tbx import explain
from tbx.checker import check
from tbx.models import Answer
from tbx.store import Store, artifact_id
from tbx.timeutil import Window, parse_utc

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
MSG = artifact_id("messages", "sms.tsv", "1")
LOC = artifact_id("locations", "loc.tsv", "1")
NOTS = artifact_id("messages", "sms.tsv", "2")
WINDOW = Window(parse_utc("2024-06-28T01:00:00Z"), parse_utc("2024-06-28T03:00:00Z"))   # 9-11 PM EDT, Jun 27


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    s.set_meta("device_tz", NY)
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": "{}"}
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-06-28T01:42:07Z", **prov, "app": "Phone", "direction": "outgoing",
                       "number": "+1 (919) 555-0134", "contact_name": "Liz Dehner", "duration_s": 121})
    s.insert("messages", {"artifact_id": MSG, "source_path": "sms.tsv", "source_row": "1",
                          "ts_utc": "2024-06-27T23:15:03Z", **prov, "app": "SMS", "thread_id": "7",
                          "direction": "incoming", "sender": "Bob", "sender_number": "9195550199",
                          "body": "Meet me at the north gate after the game tonight, around ten or so."})
    s.insert("messages", {"artifact_id": NOTS, "source_path": "sms.tsv", "source_row": "2", "ts_utc": None, **prov,
                          "app": "SMS", "thread_id": "7", "direction": "outgoing", "body": "ok"})
    s.insert("locations", {"artifact_id": LOC, "source_path": "loc.tsv", "source_row": "1",
                           "ts_utc": "2024-06-28T01:50:00Z", **prov, "source": "wifi", "lat": 35.6596081,
                           "lon": -78.8730535, "accuracy_m": 5})
    return s


def rejected(store, text, *cites, window=WINDOW, **kw):
    """Check one claim with the real checker; return its rejected entry and its entity spans (as JSON)."""
    claim = {"text": text, "citations": [{"artifact_id": a, "field": f, "value": v} for a, f, v in cites], **kw}
    res = check(Answer.model_validate({"claims": [claim]}), store, window).model_dump(mode="json")
    assert res["rejected"], res
    return res["rejected"][0], res["entities"][0]


def say(store, *args, **kw):
    entry, spans = rejected(store, *args, **kw)
    return explain.sentences(entry, spans, store, NY)


def test_value_mismatch_quotes_the_stored_value(store):
    assert say(store, "A call went out.", (CALL, "direction", "incoming")) == [
        "Cited direction = 'incoming'; the record says 'outgoing'."]


def test_value_mismatch_on_a_timestamp_shows_both_times_case_local(store):
    assert say(store, "A call.", (CALL, "ts_local", "2024-06-27T21:15:00-04:00")) == [
        "The cited time (ts_local) is Thu Jun 27, 2024 9:15:00 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "ts_utc", "2024-06-28T01:15:00Z")) == [
        "The cited time (ts_utc) is Thu Jun 27, 2024 9:15:00 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "ts_utc", "2024-06-28T01:15Z")) == [        # no seconds cited, none shown
        "The cited time (ts_utc) is Thu Jun 27, 2024 9:15 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "ts_local", "2024-06-27 21:15")) == [        # no offset: the case timezone
        "The cited time (ts_local) is Thu Jun 27, 2024 9:15 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "ts_local", "last night")) == [
        "The cited ts_local 'last night' is not a valid time; the record's time is Thu Jun 27, 2024 9:42:07 PM "
        "(UTC-04:00)."]
    assert say(store, "A reply.", (NOTS, "ts_utc", "2024-06-28T01:15:00Z"), window=None) == [
        "Cited ts_utc = '2024-06-28T01:15:00Z'; the record has no timestamp."]


def test_value_mismatch_numbers_long_text_and_missing_values(store):
    assert say(store, "It lasted a while.", (CALL, "duration_s", "120")) == [
        "Cited duration_s = '120'; the record says '121'."]
    (s,) = say(store, "Bob wrote.", (MSG, "body", "Meet me at the south gate"), (MSG, "sender", "Bob"), window=None)
    assert s == ("Cited body = 'Meet me at the south gate'; the record says 'Meet me at the north gate after the "
                 "game tonight, around ten or so.'.")
    assert say(store, "A message.", (MSG, "recipient", "Alice"), window=None) == [
        "Cited recipient = 'Alice'; the record has no recipient."]


def test_outside_window_gives_the_case_local_time_or_says_there_is_none(store):
    assert say(store, "Bob wrote.", (MSG, "sender", "Bob")) == [
        "This record is from Thu Jun 27, 2024 7:15:03 PM (UTC-04:00), outside the question window."]
    assert say(store, "A reply.", (NOTS, "direction", "outgoing")) == ["This record has no timestamp."]


def test_ungrounded_entity_quotes_the_red_span(store):
    assert say(store, "Liz Dehner called at 9:15 PM.", (CALL, "contact_name", "Liz Dehner")) == [
        "'9:15 PM' doesn't match the cited record's time (9:42 PM)."]
    assert say(store, "bob called.", (CALL, "direction", "outgoing")) == [
        "'bob' doesn't appear in any cited record."]          # the claim's own spelling of the contact Bob


def test_two_ungrounded_entities_of_one_type_keep_their_order(store):
    entry, spans = rejected(store, "Calls at 9:01 PM and 9:02 PM, then Liz Dehner and Bob.",
                            (CALL, "direction", "outgoing"))
    assert [r["code"] for r in entry["reasons"]] == ["UNGROUNDED_ENTITY:clock_time"] * 2 + ["UNGROUNDED_ENTITY:contact"] * 2
    assert explain.sentences(entry, spans, store, NY) == [
        "'9:01 PM' doesn't match the cited record's time (9:42 PM).",
        "'9:02 PM' doesn't match the cited record's time (9:42 PM).",
        "'Bob' doesn't appear in any cited record.", "'Liz Dehner' doesn't appear in any cited record."]


# ------------------------------------------------------------------------------ F2: the true value
def test_a_wrong_clock_time_names_the_cited_records_time(store):
    # the citations pass (ts cited correctly); only the time written in the text is wrong
    assert say(store, "Liz Dehner called at 9:36 PM.", (CALL, "contact_name", "Liz Dehner"),
               (CALL, "ts_local", "2024-06-27T21:42:07-04:00")) == [
        "'9:36 PM' doesn't match the cited record's time (9:42 PM)."]
    assert say(store, "Liz Dehner called at 21:36.", (CALL, "contact_name", "Liz Dehner")) == [
        "'21:36' doesn't match the cited record's time (21:42)."]                  # in the claim's own style


def test_with_several_cited_records_the_nearest_time_is_named(store):
    # the call is at 9:42:07 PM, the location fix at 9:50:00 PM
    assert say(store, "Liz Dehner called at 9:48 PM from the gym.", (CALL, "contact_name", "Liz Dehner"),
               (LOC, "source", "wifi")) == ["'9:48 PM' doesn't match the cited record's time (9:50 PM)."]
    assert say(store, "Liz Dehner called at 9:30 PM from the gym.", (CALL, "contact_name", "Liz Dehner"),
               (LOC, "source", "wifi")) == ["'9:30 PM' doesn't match the cited record's time (9:42 PM)."]


def test_a_wrong_date_names_the_cited_records_date(store):
    assert say(store, "On June 28 Liz Dehner called.", (CALL, "contact_name", "Liz Dehner")) == [
        "'June 28' doesn't match the cited record's date (Thu Jun 27, 2024)."]


def test_a_right_time_that_is_not_cited_says_so(store):
    assert say(store, "Liz Dehner called at 9:42 PM.", (CALL, "contact_name", "Liz Dehner")) == [
        "'9:42 PM' matches the cited record's time, but the claim doesn't cite that record's timestamp."]
    assert say(store, "A reply at 9:15 PM.", (NOTS, "direction", "outgoing"), window=None) == [
        "'9:15 PM' doesn't appear in any cited record; none of them has a timestamp."]


def test_after_a_failed_citation_the_entity_is_not_backed_by_a_correctly_cited_value(store):
    # the classic: a wrong time in the text, cited with the same wrong time
    assert say(store, "Liz Dehner called at 9:15 PM.", (CALL, "contact_name", "Liz Dehner"),
               (CALL, "ts_local", "2024-06-27T21:15:00-04:00")) == [
        "The cited time (ts_local) is Thu Jun 27, 2024 9:15:00 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:42:07 PM (UTC-04:00).",
        "'9:15 PM' isn't backed by a correctly cited time (the cited record's time is 9:42 PM)."]
    assert say(store, "On June 28 at 9:15 PM.", ("call_ffffffffff", "ts_utc", "2024-06-29T01:15:00Z")) == [
        "No record call_ffffffffff exists in this extraction.",
        "'June 28' isn't backed by a correctly cited time.", "'9:15 PM' isn't backed by a correctly cited time."]
    assert say(store, "bob wrote.", (MSG, "sender", "Alice"), window=None) == [
        "Cited sender = 'Alice'; the record says 'Bob'.", "'bob' isn't backed by a correctly cited value."]
    assert say(store, "Liz Dehner called.", (CALL, "raw_json", "{}")) == [
        "raw_json can't be cited.", "'Liz Dehner' isn't backed by a correctly cited value."]


def test_time_range_sentences(store):
    # the call's support: 9:42:07 PM to 9:44:08 PM (+121 s)
    assert say(store, "A call.", (CALL, "direction", "outgoing"), time_start="2024-06-27T21:30:00-04:00",
               time_end="2024-06-27T21:44:00-04:00") == [
        "The claim says Thu Jun 27, 2024 9:30:00 PM – 9:44:00 PM (UTC-04:00); the cited records span "
        "Thu Jun 27, 2024 9:42:07 PM – 9:44:08 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "direction", "outgoing"), time_start="2024-06-27T22:30:00-04:00") == [
        "The claim says it starts at Thu Jun 27, 2024 10:30:00 PM (UTC-04:00); the cited records span "
        "Thu Jun 27, 2024 9:42:07 PM – 9:44:08 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "direction", "outgoing"), time_start="2024-06-27T21:43:00-04:00",
               time_end="2024-06-27T21:42:30-04:00") == [
        "The claim's start, Thu Jun 27, 2024 9:43:00 PM (UTC-04:00), is after its end, "
        "Thu Jun 27, 2024 9:42:30 PM (UTC-04:00)."]
    assert say(store, "A call.", (CALL, "direction", "outgoing"), time_start="tonight") == [
        "The claim's time_start 'tonight' is not a valid time."]
    assert say(store, "A reply.", (NOTS, "direction", "outgoing"), time_end="2024-06-27T21:43:00-04:00", window=None) == [
        "The claim says it ends at Thu Jun 27, 2024 9:43:00 PM (UTC-04:00), but none of the cited records has a "
        "timestamp."]


def test_unknown_artifact_field_and_no_citations(store):
    assert say(store, "A call.", ("call_0d96ca43-8907", "direction", "outgoing")) == [
        "No record call_0d96ca43-8907 exists in this extraction."]
    assert say(store, "A call.", (CALL, "raw_json", "{}")) == ["raw_json can't be cited."]
    assert say(store, "Something happened.") == ["The claim cites nothing."]


def test_every_reason_gets_a_sentence_in_order(store):
    entry, spans = rejected(store, "Bob called at 9:15 PM.", (CALL, "direction", "incoming"),
                            ("msg_ffffffffff", "body", "x"), (CALL, "tz_basis", "utc_string"))
    out = explain.sentences(entry, spans, store, NY)
    assert len(out) == len(entry["reasons"]) and all(out)
    assert [r["code"] for r in entry["reasons"]][:3] == ["VALUE_MISMATCH", "UNKNOWN_ARTIFACT", "UNKNOWN_FIELD"]


def test_an_unknown_code_falls_back_to_its_detail(store):
    entry = {"claim": {"text": "x", "citations": []}, "reasons": [{"code": "NEW_RULE", "detail": "why", "citation_index": None}]}
    assert explain.sentences(entry, [], store, NY) == ["why"]
