"""tbx.ingest on a tiny ALEAPP-shaped report (real header names, synthetic rows)."""
import csv
import hashlib
import json
import sqlite3

import pytest

from tbx import ingest
from tbx.store import Store

TSV = "_TSV Exports"


def write_tsv(root, name, header, rows):
    d = root / TSV
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def report(tmp_path):
    root = tmp_path / "ALEAPP_Output"
    write_tsv(root, "GoogleMessages.tsv",
              ["Message Timestamp", "Direction", "Message Sender", "Other Participant/Conversation Name", "Message",
               "Message Type", "Attachment Byte Size", "Attachment Location", "Conversation ID"],
              [["2024-07-26 19:03:11.805000+00:00", "Incoming", "(984) 291-5610", "Thom De'Fer", "Are you here yet?",
                "text/plain", "N/A", "", "174"],
               ["2024-07-26 19:11:00.100000+00:00", "Outgoing", "", "Thom De'Fer", "Yes I am.\nSecond line",
                "text/plain", "N/A", "", "174"],
               ["2024-01-27 14:06:13.591000+00:00", "Incoming", "32665", "32665", "946663 is your code", "text/plain",
                "N/A", "", "1"]])
    write_tsv(root, "Call logs .tsv",
              ["Call Date", "Phone Account Address", "SIM ICCID", "Phone Account Package", "Partner",
               "Raw Call Type (as stored)", "Type", "Duration in Secs", "Partner Location", "Country ISO", "Data",
               "Mime Type", "Transcription", "Deleted"],
              [["2024-07-27 21:40:04.825000+00:00", "19199282177", "x", "com.android.phone", "9195794674", "2",
                'Outgoing <i data-feather="phone-outgoing"></i>', "121", "North Carolina", "US", " ", " ", " ", "0"],
               ["2024-01-27 16:50:18+00:00", " ", "x", "com.android.phone", "+14722000026", "4", "Voicemail", "58",
                "", "US", "/data/x", " ", "Transcript not available", "0"],
               ["2024-02-01 15:00:00+00:00", " ", "x", "com.android.phone", "+18005551212", "4", "Voicemail", "31",
                "", "US", "/data/y", " ", "  Hi, your U-Haul reservation is ready.  ", "0"],
               ["2024-02-02 15:00:00+00:00", " ", "x", "com.android.phone", "+18005550000", "3", "Missed", "0",
                "", "US", " ", " ", "", "0"]])
    write_tsv(root, "Contacts.tsv", ["Mimetype", "Data 1", "Display Name", "Phone Number", "Email Address", "Source File"],
              [["vnd.android.cursor.item/phone_v2", "+1 919-579-4674", "Josh Hickman", "+1 919-579-4674", "", "c.db"]])
    write_tsv(root, "Google Voice - Messages.tsv",
              ["Timestamp", "Direction", "Sender", "Message", "Image", "Account Number", "Conversation ID",
               "Recipient(s)", "Read Status"],
              [["2024/02/26 15:59:13", "Outgoing", "+19104773625", "Are you there?", "", "1", "t.+19105576476",
                "+19105576476", ""]])
    write_tsv(root, "Life360 - Locations.tsv",
              ["Timestamp", "Latitude", "Longitude", "Altitude", "Speed (as stored)", "Course", "Bearing",
               "Vertical Accuracy (as stored)", "Horizontal Accuracy (as stored)", "Location Mode",
               "Connected Access Point BSSID", "Connected Access Point SSID", "ID", "Source File"],
              [["2024-06-21 22:37:59.856000+00:00", "34.6803928", "-76.9265727", "-32.9", "0.0", "0.0", "0.0", "100.0",
                "9.947", "move", "", "", "623cd5ca-03af-4173-b622-40f30531fefa", "L360EventStore.db"]])
    write_tsv(root, "Build.tsv", ["Label", "Value"], [["Manufacturer", "Google"], ["Model", "Pixel 7a"]])
    write_tsv(root, "Unrelated Artifact.tsv", ["A", "B"], [["1", "2"]])
    (root / "_lava_data.lava").write_text(json.dumps({"parser_info": {"leapp_version": "2026.4.5-dev"}}))
    return root


def _pb(field, payload: bytes) -> bytes:
    return bytes([field << 3 | 2, len(payload)]) + payload


@pytest.fixture
def media(tmp_path):
    """Extraction-root mirror: persistent_properties (protobuf) + one geotagged JPEG."""
    from PIL import Image
    root = tmp_path / "media"
    prop = root / "data/property/persistent_properties"
    prop.parent.mkdir(parents=True)
    entry = _pb(1, b"persist.sys.timezone") + _pb(2, b"America/New_York")
    prop.write_bytes(_pb(1, _pb(1, b"persist.sys.locale") + _pb(2, b"en-US")) + _pb(1, entry))
    cam = root / "data/media/0/DCIM/Camera"
    cam.mkdir(parents=True)
    im = Image.new("RGB", (64, 48), "red")
    exif = Image.Exif()
    exif[0x0132] = "2024:07:16 19:59:52"
    ifd = exif.get_ifd(0x8769)
    ifd[36867] = "2024:07:16 19:59:52"
    ifd[36881] = "-04:00"
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N", 2: (38.0, 54.0, 8.64), 3: "W", 4: (77.0, 2.0, 2.616)})
    im.save(cam / "PXL_20240716_235952437.jpg", exif=exif)
    (cam / "notes.txt").write_text("not media")
    return root


def rows(db, table):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute(f"SELECT * FROM {table} ORDER BY artifact_id")]
    con.close()
    return out


def test_ingest_maps_families_with_provenance(report, tmp_path):
    db = tmp_path / "t.db"
    summary = ingest.ingest_report(str(report), str(db))
    assert summary["counts"]["messages"] == 4 and summary["counts"]["calls"] == 4 and summary["counts"]["locations"] == 1
    msgs = {r["body"]: r for r in rows(db, "messages")}
    m = msgs["Are you here yet?"]
    assert m["source_path"] == f"{TSV}/GoogleMessages.tsv" and m["source_row"] == "1"
    assert m["artifact_id"] == "msg_" + hashlib.sha1(f"{TSV}/GoogleMessages.tsv|1".encode()).hexdigest()[:10]
    assert m["ts_utc"] == "2024-07-26T19:03:11Z" and m["ts_original"] == "2024-07-26 19:03:11.805000+00:00"
    assert m["tz_basis"] == "utc_string" and m["direction"] == "incoming"
    assert m["sender"] == "Thom De'Fer" and m["sender_number"] == "(984) 291-5610" and m["thread_id"] == "174"
    assert json.loads(m["raw_json"])["Message"] == "Are you here yet?"
    out = msgs["Yes I am.\nSecond line"]
    assert out["direction"] == "outgoing" and out["recipient"] == "Thom De'Fer" and out["sender"] is None
    code = msgs["946663 is your code"]
    assert code["sender"] is None and code["sender_number"] == "32665"      # numbers never become names
    gv = msgs["Are you there?"]
    assert gv["ts_utc"] == "2024-02-26T15:59:13Z" and gv["app"] == "Google Voice"
    assert gv["recipient_number"] == "+19105576476"


def test_calls_direction_duration_and_contact_lookup(report, tmp_path):
    db = tmp_path / "t.db"
    ingest.ingest_report(str(report), str(db))
    calls = {r["number"]: r for r in rows(db, "calls")}
    out = calls["9195794674"]
    assert out["direction"] == "outgoing" and out["duration_s"] == 121 and out["contact_name"] == "Josh Hickman"
    raw = json.loads(out["raw_json"])
    assert "Contacts.tsv" in raw["_tbx"]["contact_name_from"]
    assert out["transcript"] is None                                     # " " is not a transcript
    vm = calls["+14722000026"]
    assert vm["direction"] == "voicemail" and vm["contact_name"] is None  # raw type 4 (v1.2)
    assert vm["transcript"] is None                                      # placeholder -> NULL
    vm2 = calls["+18005551212"]
    assert vm2["direction"] == "voicemail" and vm2["transcript"] == "Hi, your U-Haul reservation is ready."
    assert calls["+18005550000"]["direction"] == "missed" and calls["+18005550000"]["transcript"] is None


@pytest.mark.parametrize("raw, want", [("Transcript not available", None), ("transcript not available.", None),
                                       ("", None), ("   ", None), (None, None), (" Call me back. ", "Call me back.")])
def test_voicemail_transcript_placeholders(raw, want):
    assert ingest.voicemail_transcript(raw) == want


def test_locations_and_case_meta(report, media, tmp_path):
    db = tmp_path / "t.db"
    ingest.ingest_report(str(report), str(db), media_root=str(media))
    loc = rows(db, "locations")[0]
    assert loc["source"] == "life360" and loc["lat"] == pytest.approx(34.6803928) and loc["accuracy_m"] == pytest.approx(9.947)
    assert loc["source_row"] == "623cd5ca-03af-4173-b622-40f30531fefa"     # natural key
    s = Store(str(db))
    meta = s.meta()
    assert s.case_tz == "America/New_York" and meta["report_root"] == "ALEAPP_Output"
    assert meta["aleapp_version"] == "2026.4.5-dev" and meta["device_model"] == "Google Pixel 7a"
    assert "ingested_at" in meta


def test_device_tz_defaults_to_utc_with_warning(report, tmp_path):
    summary = ingest.ingest_report(str(report), str(tmp_path / "t.db"))
    assert Store(str(tmp_path / "t.db")).case_tz == "UTC" and any("device_tz" in w for w in summary["warnings"])


def test_media_exif_time_gps_and_photo_location(report, media, tmp_path):
    db = tmp_path / "t.db"
    ingest.ingest_report(str(report), str(db), media_root=str(media))
    med = rows(db, "media")
    assert len(med) == 1
    m = med[0]
    rel = "data/media/0/DCIM/Camera/PXL_20240716_235952437.jpg"
    assert m["source_path"] == rel and m["file_path"] == rel and m["source_row"] == "1"
    assert m["kind"] == "image" and m["mime_type"] == "image/jpeg" and m["app"] == "Camera"
    assert m["ts_utc"] == "2024-07-16T23:59:52Z" and m["ts_original"] == "2024:07:16 19:59:52 -04:00"
    assert m["tz_basis"] == "offset_string"                               # v1.2
    assert m["lat"] == pytest.approx(38.9024, abs=1e-4) and m["lon"] == pytest.approx(-77.03406, abs=1e-4)
    assert len(m["sha256"]) == 64 and m["size_bytes"] > 0
    loc = [r for r in rows(db, "locations") if r["source"] == "photo_exif"]
    assert len(loc) == 1 and loc[0]["source_path"] == rel and loc[0]["ts_utc"] == m["ts_utc"]


def test_ingest_twice_identical_ids(report, media, tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    ingest.ingest_report(str(report), str(a), media_root=str(media))
    ingest.ingest_report(str(report), str(b), media_root=str(media))
    for t in ("messages", "calls", "locations", "media"):
        assert [r["artifact_id"] for r in rows(a, t)] == [r["artifact_id"] for r in rows(b, t)]


def test_reingest_replaces_db(report, tmp_path):
    db = tmp_path / "t.db"
    ingest.ingest_report(str(report), str(db))
    ingest.ingest_report(str(report), str(db))
    assert len(rows(db, "messages")) == 4


def test_missing_report_dir_is_an_error(tmp_path):
    with pytest.raises(ingest.IngestError):
        ingest.ingest_report(str(tmp_path / "nope"), str(tmp_path / "t.db"))


@pytest.mark.parametrize("raw,utc", [
    ("2024-01-27 14:06:13.591000+00:00", "2024-01-27T14:06:13Z"),
    ("2024-01-27 16:50:18+00:00", "2024-01-27T16:50:18Z"),
    ("2024/02/26 15:59:13", "2024-02-26T15:59:13Z"),
    ("2024-07-27T21:40:04Z", "2024-07-27T21:40:04Z"),
])
def test_parse_aleapp_ts(raw, utc):
    assert ingest.parse_aleapp_ts(raw) == (utc, "utc_string")


def test_parse_aleapp_ts_with_a_real_offset_is_offset_string():
    assert ingest.parse_aleapp_ts("2024-07-27 17:40:04-04:00") == ("2024-07-27T21:40:04Z", "offset_string")


@pytest.mark.parametrize("raw", ["", " ", "N/A", "0", "garbage"])
def test_parse_aleapp_ts_empty(raw):
    assert ingest.parse_aleapp_ts(raw) == (None, "unknown")


def test_duration_parsing():
    assert ingest.parse_duration("00:01:40") == 100 and ingest.parse_duration("121") == 121
    assert ingest.parse_duration("1:05") == 65 and ingest.parse_duration("") is None


def test_jids_and_emails_are_not_names():
    assert not ingest.is_name("12027037640@s.whatsapp.net") and not ingest.is_name("a@b.org")
    assert ingest._party("12027037640@s.whatsapp.net", None) == (None, "+12027037640")
    assert ingest.is_name("Thom De'Fer") and ingest.is_name("USPS")


def test_received_attachment_uses_message_time_not_sender_exif(report, media, tmp_path):
    import shutil
    rel = "data/data/com.google.android.apps.messaging/cache/telephony_cache/conversation_1_message_2_part_3_.bin"
    dst = media / rel
    dst.parent.mkdir(parents=True)
    shutil.copyfile(media / "data/media/0/DCIM/Camera/PXL_20240716_235952437.jpg", dst)  # has EXIF
    write_tsv(report, "GoogleMessages.tsv",
              ["Message Timestamp", "Direction", "Message Sender", "Other Participant/Conversation Name", "Message",
               "Message Type", "Attachment Byte Size", "Attachment Location", "Conversation ID"],
              [["2024-07-26 19:41:16.943000+00:00", "Incoming", "(984) 291-5610", "Thom De'Fer", "", "image/jpeg",
                "1", "/data/user/0/" + rel.removeprefix("data/data/"), "174"]])
    db = tmp_path / "t.db"
    ingest.ingest_report(str(report), str(db), media_root=str(media))
    m = next(r for r in rows(db, "media") if r["source_path"] == rel)
    assert m["ts_utc"] == "2024-07-26T19:41:16Z" and m["app"] == "Google Messages" and m["kind"] == "image"
    msg = next(r for r in rows(db, "messages") if r["app"] == "Google Messages")
    assert msg["attachment_media_id"] == m["artifact_id"]
