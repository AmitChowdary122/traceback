"""tbx.store and tbx.models (CONTRACTS.md v1.1 sections 2, 4, 6)."""
import json
import sqlite3

import pytest
from pydantic import ValidationError

from tbx.models import Answer, Citation
from tbx.store import CITABLE, PREFIX, Store, artifact_id


def _call(aid="call_0123456789", **kw):
    row = {"artifact_id": aid, "source_path": "Call Logs/calls.tsv", "source_row": "1",
           "ts_utc": "2023-03-14T01:42:07Z", "ts_original": "1678758127000", "tz_basis": "utc_epoch_ms",
           "raw_json": "{}", "app": "Phone", "direction": "outgoing", "number": "+15550101234",
           "contact_name": "Alice", "duration_s": 95}
    row.update(kw)
    return row


def test_artifact_id_formula():
    import hashlib
    want = "msg_" + hashlib.sha1(b"SMS/sms.tsv|101").hexdigest()[:10]
    assert artifact_id("messages", "SMS/sms.tsv", "101") == want
    assert PREFIX == {"messages": "msg", "calls": "call", "locations": "loc", "media": "med",
                      "transcripts": "trn", "image_tags": "tag"}


def test_artifact_id_rejects_absolute_or_windows_paths():
    for bad in ("/abs/path.tsv", "C:/x.tsv", "a\\b.tsv", "../up.tsv"):
        with pytest.raises(ValueError):
            artifact_id("calls", bad, "1")


def test_schema_created_and_roundtrip(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.insert("calls", _call())
    got = s.get("call_0123456789")
    assert got["table"] == "calls" and got["contact_name"] == "Alice" and got["raw_json"] == "{}"
    assert s.get("call_9999999999") is None
    assert s.get("call_0d96ca43-8907-4d91-aac2-346156bd69d6") is None
    assert s.get("nonsense") is None


def test_insert_collision_fails_loudly(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.insert("calls", _call())
    with pytest.raises(ValueError, match="collision"):
        s.insert("calls", _call())


def test_insert_rejects_unknown_table_and_column(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    with pytest.raises(ValueError):
        s.insert("bogus", _call())
    with pytest.raises(ValueError):
        s.insert("calls", _call(evil="x"))


def test_meta_and_case_tz(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    assert s.case_tz == "UTC"
    s.set_meta("device_tz", "America/New_York")
    assert s.case_tz == "America/New_York"
    s.set_meta("device_tz", "Europe/London")
    assert s.case_tz == "Europe/London"
    assert s.meta()["device_tz"] == "Europe/London"


def test_known_contacts_distinct_non_empty(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.insert("calls", _call())
    s.insert("calls", _call("call_1111111111", contact_name=""))
    s.insert("calls", _call("call_2222222222", contact_name=None))
    base = {"source_path": "SMS/sms.tsv", "ts_utc": None, "ts_original": None, "tz_basis": "unknown", "raw_json": "{}",
            "app": "SMS", "thread_id": "1", "direction": "incoming", "body": "hi"}
    s.insert("messages", dict(base, artifact_id="msg_0000000001", source_row="1", sender="Bob", recipient=" "))
    s.insert("messages", dict(base, artifact_id="msg_0000000002", source_row="2", sender=None, recipient="Carol"))
    assert s.known_contacts() == {"Alice", "Bob", "Carol"}


def test_store_reopens_existing_db(tmp_path):
    p = str(tmp_path / "t.db")
    Store(p).insert("calls", _call())
    assert Store(p).get("call_0123456789")["number"] == "+15550101234"


def test_citable_fields_follow_contract():
    assert CITABLE["calls"] == {"ts_utc", "ts_local", "source_path", "app", "direction", "number", "contact_name", "duration_s",
                                "transcript"}  # v1.2
    assert "raw_json" not in CITABLE["messages"] and "tz_basis" not in CITABLE["messages"]
    assert "ts_original" not in CITABLE["media"]


def test_provenance_columns_not_null(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        s.insert("calls", _call(raw_json=None))


def test_models_validate_answer_shape():
    a = Answer.model_validate({"claims": [{"text": "x", "citations": [{"artifact_id": "a", "field": "f", "value": "v"}]}]})
    assert a.claims[0].time_start is None and a.note is None
    assert Answer.model_validate({"claims": []}).claims == []
    with pytest.raises(ValidationError):
        Answer.model_validate({"claims": [{"citations": []}]})
    with pytest.raises(ValidationError):
        Answer.model_validate({"claims": [{"text": "x", "citations": [{"artifact_id": "a", "field": "f"}]}]})


def test_citation_value_numbers_are_kept_as_strings():
    assert Citation.model_validate({"artifact_id": "a", "field": "duration_s", "value": 95}).value == "95"
    assert Citation.model_validate({"artifact_id": "a", "field": "lat", "value": 40.7484}).value == "40.7484"


def test_rows_are_json_serialisable(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.insert("calls", _call())
    json.dumps(s.get("call_0123456789"))
