"""tbx.tools: the six typed tools (CONTRACTS.md section 5)."""
import json

import pytest

from tbx import tools
from tbx.store import Store, artifact_id

NY = "America/New_York"


def _prov(src, key, ts):
    return {"artifact_id": None, "source_path": src, "source_row": key, "ts_utc": ts, "ts_original": ts,
            "tz_basis": "utc_string", "raw_json": json.dumps({"k": key})}


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.db"))
    s.set_meta("device_tz", NY)
    for i, (ts, body, sender, num) in enumerate([
        ("2024-07-26T19:03:00Z", "Are you here yet?", "Liz Dehner", "+1 919-555-1234"),
        ("2024-07-26T19:11:00Z", "Yes I am. " + "x" * 400, None, None),
        ("2024-07-26T19:03:00Z", "Same second, other id", "Bob", "5550209876"),
        (None, "draft with no time", None, None),
    ], 1):
        row = _prov("Messages/sms.tsv", str(i), ts)
        row.update(artifact_id=artifact_id("messages", "Messages/sms.tsv", str(i)), app="Google Messages",
                   thread_id="1", direction="incoming" if sender else "outgoing", sender=sender,
                   sender_number=num, recipient=None if sender else "Liz Dehner", recipient_number=None,
                   body=body, attachment_media_id=None)
        s.insert("messages", row)
    for i, (ts, d, num, name, dur) in enumerate([
        ("2024-07-27T21:40:00Z", "outgoing", "919-579-4674", "Josh", 121.0),
        ("2024-07-27T21:45:00Z", "incoming", "(919) 579-4674", "Josh", 105.0),
        ("2024-02-01T20:07:00Z", "outgoing", "828-705-8820", None, 13.0),
    ], 1):
        row = _prov("Call Logs/calls.tsv", str(i), ts)
        row.update(artifact_id=artifact_id("calls", "Call Logs/calls.tsv", str(i)), app="Phone", direction=d,
                   number=num, contact_name=name, duration_s=dur)
        s.insert("calls", row)
    med = artifact_id("media", "DCIM/Camera/a.jpg", "1")
    row = _prov("DCIM/Camera/a.jpg", "1", "2024-07-27T16:00:00Z")
    row.update(artifact_id=med, app="Camera", kind="image", file_path="DCIM/Camera/a.jpg", mime_type="image/jpeg",
               sha256="0" * 64, size_bytes=1000, lat=35.5, lon=-78.7)
    s.insert("media", row)
    row = _prov("DCIM/Camera/a.jpg", f"{med}|sam-3.1|car|0", "2024-07-27T16:00:00Z")
    row.update(artifact_id=artifact_id("image_tags", "DCIM/Camera/a.jpg", f"{med}|sam-3.1|car|0"), media_id=med,
               model="sam-3.1", label="car", score=0.9, box="[1, 2, 3, 4]", prompt_set="v1")
    s.insert("image_tags", row)
    for i, (ts, lat) in enumerate([("2024-07-27T21:00:00Z", 35.1), ("2024-07-27T22:00:00Z", 35.2)], 1):
        row = _prov("Locations/loc.tsv", str(i), ts)
        row.update(artifact_id=artifact_id("locations", "Locations/loc.tsv", str(i)), source="google_maps",
                   lat=lat, lon=-78.0, accuracy_m=10.0)
        s.insert("locations", row)
    return s


def call(store, name, **args):
    out = tools.run_tool(store, name, args)
    json.dumps(out)
    return out


def test_shape_order_and_ts_local(store):
    out = call(store, "search_messages")
    assert set(out) >= {"rows", "total", "truncated"} and out["total"] == 4 and out["truncated"] is False
    keys = [(r["ts_utc"] or "~", r["artifact_id"]) for r in out["rows"]]
    assert keys == sorted(keys) and out["rows"][-1]["ts_utc"] is None   # NULL timestamps last
    first = out["rows"][0]
    assert first["ts_local"] == "2024-07-26T15:03:00-04:00"
    assert "raw_json" not in first and "tz_basis" not in first and "ts_original" not in first


def test_long_body_truncated(store):
    out = call(store, "search_messages", query="yes i am")
    row = out["rows"][0]
    assert len(row["body"]) == 300 and row["body_truncated"] is True
    full = call(store, "get_artifact", artifact_id=row["artifact_id"])["rows"][0]
    assert len(full["body"]) > 300 and "raw_json" not in full and "body_truncated" not in full


def test_limit_total_truncated_and_cap(store):
    out = call(store, "search_messages", limit=1)
    assert len(out["rows"]) == 1 and out["total"] == 4 and out["truncated"] is True
    assert call(store, "search_messages", limit=10_000)["total"] == 4
    assert tools.clamp_limit(10_000) == 200 and tools.clamp_limit(0) == 1 and tools.clamp_limit(None) == 50


def test_time_filters_inclusive_and_offsets(store):
    out = call(store, "get_calls", start="2024-07-27T17:40:00-04:00", end="2024-07-27T21:45:00Z")
    assert [r["direction"] for r in out["rows"]] == ["outgoing", "incoming"]
    out = call(store, "get_calls", start="2024-07-27T21:40:01Z")
    assert out["total"] == 1


def test_number_filter_uses_phone_rule(store):
    assert call(store, "get_calls", number="+19195794674")["total"] == 2
    assert call(store, "get_calls", number="9195794675")["total"] == 0
    assert call(store, "search_messages", number="919 555 1234")["total"] == 1


def test_contact_direction_app_query_filters(store):
    assert call(store, "get_calls", contact="josh", direction="incoming")["total"] == 1
    assert call(store, "search_messages", contact="liz")["total"] == 3          # sender or recipient
    assert call(store, "search_messages", app="google messages", query="here YET")["total"] == 1
    assert call(store, "search_messages", query="nothing-like-this")["rows"] == []


def test_locations_media_and_tags(store):
    assert call(store, "get_locations", source="google_maps", start="2024-07-27T21:30:00Z")["total"] == 1
    assert call(store, "find_media", kind="image")["total"] == 1
    assert call(store, "find_media", tag="car")["total"] == 1
    assert call(store, "find_media", tag="dog")["total"] == 0
    assert call(store, "get_transcripts")["total"] == 0


def test_measure_numbers_are_ints_when_integral(store):
    row = call(store, "get_calls", number="828-705-8820")["rows"][0]
    assert row["duration_s"] == 13 and isinstance(row["duration_s"], int)


def test_bad_args_return_error_not_exception(store):
    out = call(store, "get_calls", start="last tuesday")
    assert out["rows"] == [] and out["total"] == 0 and "error" in out
    assert "error" in call(store, "get_artifact")
    assert call(store, "get_artifact", artifact_id="call_0000000000")["rows"] == []
    assert "error" in call(store, "no_such_tool")


def test_tool_specs_cover_contract():
    names = {t["function"]["name"] for t in tools.TOOL_SPECS}
    assert names == {"search_messages", "get_calls", "get_locations", "find_media", "get_transcripts", "get_artifact"}
    ga = next(t for t in tools.TOOL_SPECS if t["function"]["name"] == "get_artifact")
    assert ga["function"]["parameters"]["required"] == ["artifact_id"]
