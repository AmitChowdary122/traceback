"""Acceptance: `tbx ingest` on the committed ALEAPP sample (CONTRACTS.md sections 2, 3, 4).

Needs src/tbx and data/sample/ (builder, Day 0/1). Until both exist these tests skip.
Point TBX_SAMPLE_REPORT / TBX_SAMPLE_MEDIA at the report and media dirs if they are not in a
default location.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

import pytest
import tbxcase

REPO = tbxcase.REPO
PREFIX = {"messages": "msg", "calls": "call", "locations": "loc", "media": "med", "transcripts": "trn", "image_tags": "tag"}
BASE_TABLES = ("messages", "calls", "locations", "media")
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
TZ_BASIS_RE = re.compile(r"utc_epoch_s|utc_epoch_ms|utc_string|offset_string|unknown|local_converted:(.+)")
ENUMS = {("messages", "direction"): {"incoming", "outgoing", "unknown"},
         ("calls", "direction"): {"incoming", "outgoing", "missed", "rejected", "voicemail", "unknown"},
         ("media", "kind"): {"image", "audio", "video", "other"}}


def sample_dirs():
    report = os.environ.get("TBX_SAMPLE_REPORT")
    if not report:
        for cand in ("data/sample/aleapp", "data/sample/report", "data/sample/ALEAPP_Reports", "data/sample"):
            if (REPO / cand).is_dir() and any((REPO / cand).iterdir()):
                report = str(REPO / cand)
                break
    if not report:
        tbxcase.unavailable("no ALEAPP sample report (data/sample/ or TBX_SAMPLE_REPORT)")
    media = os.environ.get("TBX_SAMPLE_MEDIA") or (str(REPO / "data/sample/media") if (REPO / "data/sample/media").is_dir() else None)
    return report, media


@pytest.fixture(scope="module")
def ingested(tbx_cli, tmp_path_factory):
    import subprocess
    report, media = sample_dirs()
    tmp = tmp_path_factory.mktemp("ingest")
    dbs = []
    for name in ("a.db", "b.db"):
        db = tmp / name
        args = tbx_cli + ["ingest", report, "--db", str(db)] + (["--media-root", media] if media else [])
        proc = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=tmp)
        assert proc.returncode == 0, f"{' '.join(args)}\n{proc.stderr[-3000:]}"
        dbs.append(db)
    return dbs


def rows(db, table):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(f"SELECT * FROM {table}")]
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()


def test_ingest_twice_gives_identical_ids(ingested):
    a, b = ingested
    for table in PREFIX:
        ids_a = sorted(r["artifact_id"] for r in rows(a, table))
        ids_b = sorted(r["artifact_id"] for r in rows(b, table))
        assert ids_a == ids_b, f"{table}: {len(set(ids_a) ^ set(ids_b))} IDs differ between two ingests"


def test_p0_families_present(ingested):
    counts = {t: len(rows(ingested[0], t)) for t in BASE_TABLES}
    assert all(counts.values()), f"empty P0 families: {counts}"


def test_ids_have_table_prefix_and_10_hex(ingested):
    for table, prefix in PREFIX.items():
        bad = [r["artifact_id"] for r in rows(ingested[0], table) if not re.fullmatch(rf"{prefix}_[0-9a-f]{{10}}", r["artifact_id"])]
        assert not bad, f"{table}: {bad[:5]}"


def expected_id(table: str, row: dict) -> str:
    key = row["source_path"] + "|" + row["source_row"]
    return PREFIX[table] + "_" + hashlib.sha1(key.encode()).hexdigest()[:10]


def test_ids_recompute_from_provenance(ingested):
    """CONTRACTS.md v1.1 section 2 (C11): source_row stores the row_key, so IDs recompute from provenance."""
    for table in BASE_TABLES:
        bad = [r["artifact_id"] for r in rows(ingested[0], table) if r["artifact_id"] != expected_id(table, r)]
        assert not bad, f"{table}: {len(bad)} IDs do not recompute from source_path|source_row, e.g. {bad[:3]}"


def test_provenance_columns(ingested):
    for table in PREFIX:
        for r in rows(ingested[0], table):
            sp = r["source_path"]
            assert sp and not sp.startswith("/") and "\\" not in sp and not re.match(r"[A-Za-z]:", sp) and ".." not in sp.split("/"), (table, sp)
            assert r["source_row"] not in (None, ""), (table, r["artifact_id"])
            assert isinstance(json.loads(r["raw_json"]), dict), (table, r["artifact_id"])
            m = TZ_BASIS_RE.fullmatch(r["tz_basis"] or "")
            assert m, (table, r["artifact_id"], r["tz_basis"])
            if m.group(1):
                assert m.group(1) in available_timezones(), r["tz_basis"]
            if r["ts_utc"] is not None:
                assert TS_RE.fullmatch(r["ts_utc"]), (table, r["artifact_id"], r["ts_utc"])
                assert 1990 <= int(r["ts_utc"][:4]) <= datetime.now().year + 1, r["ts_utc"]


def test_enums_and_ranges(ingested):
    for (table, col), allowed in ENUMS.items():
        vals = {r[col] for r in rows(ingested[0], table)}
        assert vals <= allowed, f"{table}.{col}: unexpected {vals - allowed}"
    for table in ("locations", "media"):
        for r in rows(ingested[0], table):
            if r.get("lat") is not None:
                assert -90 <= r["lat"] <= 90 and -180 <= r["lon"] <= 180, (table, r["artifact_id"])
    for r in rows(ingested[0], "media"):
        assert r["sha256"] is None or re.fullmatch(r"[0-9a-f]{64}", r["sha256"]), r["artifact_id"]


def test_case_meta(ingested):
    meta = {r["key"]: r["value"] for r in rows(ingested[0], "case_meta")}
    assert {"device_tz", "aleapp_version", "report_root", "ingested_at"} <= set(meta), sorted(meta)
    ZoneInfo(meta["device_tz"])


def test_store_api_on_ingested_db(tbx, ingested):
    s = tbx.Store(str(ingested[0]))
    assert s.case_tz == {r["key"]: r["value"] for r in rows(ingested[0], "case_meta")}["device_tz"]
    contacts = s.known_contacts()
    assert "" not in contacts and None not in contacts
    some = rows(ingested[0], "messages")[0]
    got = s.get(some["artifact_id"])
    assert got["table"] == "messages" and "raw_json" in got


# ------------------------------------------------------------------------------- v1.2 (from the G1 gate run)
def _call_log_rows(db):
    return [(r, json.loads(r["raw_json"])) for r in rows(db, "calls") if r["source_path"].endswith("Call logs .tsv")]


def test_voicemail_is_not_a_missed_call(ingested):
    """CONTRACTS v1.2 section 4: Android call type 4 is `voicemail` (2,154 rows in the sample)."""
    pairs = _call_log_rows(ingested[0])
    vm = [r for r, raw in pairs if (raw.get("Raw Call Type (as stored)") or "").strip() == "4"]
    assert vm, "no call-type-4 rows found; did the raw_json keys change?"
    wrong = sorted({r["direction"] for r in vm} - {"voicemail"})
    assert not wrong, f"{len(vm)} voicemail rows, directions seen besides 'voicemail': {wrong}"
    missed = [r for r, raw in pairs if (raw.get("Raw Call Type (as stored)") or "").strip() == "3"]
    assert missed and all(r["direction"] == "missed" for r in missed)


def test_voicemail_transcripts_are_citable_and_clean(ingested):
    """v1.2: the carrier transcription becomes calls.transcript; placeholders become NULL.
    The sample has 49 real transcriptions; 2,105 say 'Transcript not available'."""
    calls = rows(ingested[0], "calls")
    with_text = [r for r in calls if r["transcript"] is not None]
    assert not [r for r in with_text if not r["transcript"].strip() or "transcript not available" in r["transcript"].lower()]
    assert len(with_text) == 49, len(with_text)
    assert all(r["direction"] == "voicemail" for r in with_text)
    assert any("u-haul" in r["transcript"].lower() for r in with_text)


def test_exif_times_are_offset_strings(ingested):
    """v1.2 section 3: camera photo times come from EXIF strings with an explicit offset."""
    cam = [r for r in rows(ingested[0], "media") if "/DCIM/Camera/" in "/" + r["file_path"]]
    assert cam, "no camera photos ingested"
    with_offset = [r for r in cam if r["ts_original"] and re.search(r"[+-]\d\d:?\d\d$", r["ts_original"].strip())]
    assert with_offset, "no EXIF time with an offset; check ts_original"
    bad = [(r["file_path"], r["tz_basis"]) for r in with_offset if r["tz_basis"] != "offset_string"]
    assert not bad, bad[:5]
