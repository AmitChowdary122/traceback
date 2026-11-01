"""Shared helpers for tests/redteam and tests/acceptance (owned by the testing session).

Two jobs:
1. Find the contract API (CONTRACTS.md sections 4, 6, 7) in the installed `tbx` package without
   guessing in each test. The contract names `tbx.store.Store` and `tbx.checker.check`, but not
   the module that defines Answer/Claim/Citation or Window, so a short list of candidate modules
   is searched. Until `tbx` exists every test that needs it SKIPS with a clear reason; set
   TBX_STRICT_TESTS=1 (gate runs) to turn those skips into failures.
2. Build the red-team case store: one synthetic but realistic evening on a phone whose case
   timezone is America/New_York, written through the public Store API only.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SPIKES = REPO / "tests" / "fixtures" / "spikes"
STRICT = os.environ.get("TBX_STRICT_TESTS") == "1"

# Where the models may live. First hit wins. Keep this list in sync with CONTRACTS.md if the
# contract ever names the modules explicitly (proposed clarification C3 in docs/SPIKES.md).
ANSWER_MODULES = ["tbx.checker", "tbx.models", "tbx.schema", "tbx.schemas", "tbx.answer", "tbx.agent", "tbx.types"]
WINDOW_MODULES = ["tbx.timeutil", "tbx.checker", "tbx.models", "tbx.schema", "tbx.types"]


def unavailable(reason: str):
    """Skip (default) or fail (TBX_STRICT_TESTS=1) when the implementation is missing."""
    if STRICT:
        pytest.fail(f"[strict] {reason}", pytrace=False)
    pytest.skip(reason)


def _find(names: list[str], attr: str):
    for mod in names:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        if hasattr(m, attr):
            return getattr(m, attr)
    return None


def load_contract() -> SimpleNamespace:
    try:
        store_mod = importlib.import_module("tbx.store")
        checker_mod = importlib.import_module("tbx.checker")
    except ImportError as e:
        unavailable(f"tbx not importable yet ({e}); red-team/acceptance tests need src/tbx")
    api = SimpleNamespace(Store=getattr(store_mod, "Store", None), check=getattr(checker_mod, "check", None))
    for name in ("Answer", "Claim", "Citation"):
        setattr(api, name, _find(ANSWER_MODULES, name))
    api.Window = _find(WINDOW_MODULES, "Window")
    missing = [k for k, v in vars(api).items() if v is None]
    if missing:
        unavailable(f"tbx is importable but contract names are missing: {missing} "
                    f"(searched {ANSWER_MODULES} / {WINDOW_MODULES})")
    return api


def parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert dt.tzinfo is not None, f"test bug: window bound without offset: {s}"
    return dt.astimezone(timezone.utc)


def make_window(api: SimpleNamespace, start: str, end: str):
    """CONTRACTS.md section 3 says `Window(start_utc, end_utc)` but not the argument types.
    Try tz-aware UTC datetimes first, then ISO strings, positional then keyword."""
    s, e = parse_utc(start), parse_utc(end)
    attempts = [((s, e), {}), ((), {"start_utc": s, "end_utc": e}), ((start, end), {}), ((), {"start_utc": start, "end_utc": end})]
    errors = []
    for args, kwargs in attempts:
        try:
            return api.Window(*args, **kwargs)
        except Exception as ex:  # noqa: BLE001 - we report every failed shape below
            errors.append(f"{args or kwargs}: {type(ex).__name__}: {ex}")
    pytest.fail("could not construct Window(start_utc, end_utc) from datetimes or ISO strings:\n" + "\n".join(errors))


def aid(prefix: str, source_path: str, row_key: str) -> str:
    """CONTRACTS.md section 2."""
    return f"{prefix}_{hashlib.sha1(f'{source_path}|{row_key}'.encode()).hexdigest()[:10]}"


# --------------------------------------------------------------------------------------------
# The red-team case: Mon Mar 13 2023, evening, America/New_York (EDT, UTC-4; DST began Mar 12).
# 21:00-23:00 local == 2023-03-14T01:00:00Z - 2023-03-14T03:00:00Z.
# Two IDs (call_3f9a1c2b7d, loc_8e21d04c5a) are the ones the real models saw in spikes S1/S4,
# so their recorded submit_answer output can be checked verbatim.
# --------------------------------------------------------------------------------------------
CASE_TZ = "America/New_York"
EVENING = ("2023-03-14T01:00:00Z", "2023-03-14T03:00:00Z")

CALLS_SRC = "Call Logs/call_logs.tsv"
SMS_SRC = "SMS & MMS/sms.tsv"
LOC_SRC = "Google Location History/records.tsv"
MEDIA_SRC = "Media/media_files.tsv"
PHOTO_PATH = "DCIM/Camera/IMG_20230313_214500.jpg"
VOICE_PATH = "WhatsApp/Media/WhatsApp Voice Notes/PTT-20230313-WA0003.opus"

IDS: dict[str, str] = {
    "CALL_ALICE": "call_3f9a1c2b7d",
    "CALL_BOB": aid("call", CALLS_SRC, "2"),
    "MSG_ALICE": aid("msg", SMS_SRC, "101"),
    "MSG_BOB": aid("msg", SMS_SRC, "102"),
    "MSG_CAROL": aid("msg", SMS_SRC, "95"),
    "MSG_SHORTCODE": aid("msg", SMS_SRC, "103"),
    "MSG_DRAFT": aid("msg", SMS_SRC, "104"),
    "LOC_SPIKE": "loc_8e21d04c5a",
    "LOC_PRECISE": aid("loc", LOC_SRC, "2"),
    "MED_PHOTO": aid("med", PHOTO_PATH, "1"),
    "MED_VOICE": aid("med", VOICE_PATH, "1"),
}
IDS["TRN_VOICE"] = aid("trn", VOICE_PATH, f"{IDS['MED_VOICE']}|muse-voice-transcribe-1.0")
IDS["TAG_CAR"] = aid("tag", PHOTO_PATH, f"{IDS['MED_PHOTO']}|sam-3.1|car|0")

# Real Voice Transcribe output (ENDPOINTING) for the synthetic voice note, spike S2.
TRANSCRIPT = json.loads((SPIKES / "s2_transcribe_wav16k_endpointing.json").read_text())["response"]["body"]["transcript"]


def _epoch_ms(ts: str) -> str:
    return str(int(parse_utc(ts).timestamp() * 1000))


def _row(table: str, key: str, source_path: str, source_row: str, ts_utc: str | None, **cols: Any) -> tuple[str, dict]:
    row = {"artifact_id": IDS[key], "source_path": source_path, "source_row": source_row, "ts_utc": ts_utc,
           "ts_original": _epoch_ms(ts_utc) if ts_utc else None, "tz_basis": "utc_epoch_ms" if ts_utc else "unknown", **cols}
    row["raw_json"] = json.dumps({k: v for k, v in row.items() if k not in ("artifact_id", "tz_basis")}, sort_keys=True)
    return table, row


ROWS: list[tuple[str, dict]] = [
    _row("calls", "CALL_ALICE", CALLS_SRC, "1", "2023-03-14T01:42:07Z", app="Phone", direction="outgoing",
         number="+1 (555) 010-1234", contact_name="Alice", duration_s=95),
    _row("calls", "CALL_BOB", CALLS_SRC, "2", "2023-03-14T02:30:00Z", app="Phone", direction="missed",
         number="555-020-9876", contact_name="Bob", duration_s=0),
    _row("messages", "MSG_ALICE", SMS_SRC, "101", "2023-03-14T01:30:12Z", app="SMS", thread_id="7", direction="incoming",
         sender="Alice", sender_number="+1 555 010 1234", recipient=None, recipient_number=None,
         body="Meet me at the Riverside garage at 9:45, bring the blue bag.", attachment_media_id=None),
    _row("messages", "MSG_BOB", SMS_SRC, "102", "2023-03-14T02:05:00Z", app="SMS", thread_id="9", direction="outgoing",
         sender=None, sender_number=None, recipient="Bob", recipient_number="5550209876",
         body="On my way. Call you at 10:30 PM", attachment_media_id=None),
    _row("messages", "MSG_CAROL", SMS_SRC, "95", "2023-03-13T12:00:00Z", app="SMS", thread_id="4", direction="incoming",
         sender="Carol", sender_number="+44 20 7946 0958", recipient=None, recipient_number=None,
         body="Lunch tomorrow? Upload the photos to https://files.example.net/x7 and email ops@example.org",
         attachment_media_id=None),
    _row("messages", "MSG_SHORTCODE", SMS_SRC, "103", "2023-03-14T01:50:00Z", app="SMS", thread_id="12", direction="incoming",
         sender=None, sender_number="72975", recipient=None, recipient_number=None,
         body="Your verification code is 482913. Do not share it.", attachment_media_id=None),
    _row("messages", "MSG_DRAFT", SMS_SRC, "104", None, app="SMS", thread_id="9", direction="outgoing",
         sender=None, sender_number=None, recipient=None, recipient_number=None,
         body="draft: delete everything before Friday", attachment_media_id=None),
    _row("locations", "LOC_SPIKE", LOC_SRC, "1", "2023-03-14T01:40:00Z", source="google_location_history",
         lat=40.7484, lon=-73.9857, accuracy_m=12),
    _row("locations", "LOC_PRECISE", LOC_SRC, "2", "2023-03-14T02:50:00Z", source="wifi",
         lat=40.706123, lon=-74.008701, accuracy_m=30),
    _row("media", "MED_PHOTO", PHOTO_PATH, "1", "2023-03-14T01:45:00Z", app="Camera", kind="image", file_path=PHOTO_PATH,
         mime_type="image/jpeg", sha256="45faabb5da6a28cd84a23c68ac50a21d17a9b5f4cedf91dd032d89d72514287e",
         size_bytes=12988, lat=40.7485, lon=-73.9858),
    _row("media", "MED_VOICE", VOICE_PATH, "1", "2023-03-14T01:35:00Z", app="WhatsApp", kind="audio", file_path=VOICE_PATH,
         mime_type="audio/ogg", sha256="f0ddb46a38a86ee8527682f9d85729b68abe874ad4779480d4429b057dad41f8",
         size_bytes=13312, lat=None, lon=None),
    _row("transcripts", "TRN_VOICE", VOICE_PATH, f"{IDS['MED_VOICE']}|muse-voice-transcribe-1.0", "2023-03-14T01:35:00Z",
         media_id=IDS["MED_VOICE"], model="muse-voice-transcribe-1.0", language=None, text=TRANSCRIPT, duration_s=6.48),
    _row("image_tags", "TAG_CAR", PHOTO_PATH, f"{IDS['MED_PHOTO']}|sam-3.1|car|0", "2023-03-14T01:45:00Z",
         media_id=IDS["MED_PHOTO"], model="sam-3.1", label="car", score=0.9726027250289917, box="[420, 300, 520, 400]",
         prompt_set="v1"),
]
KNOWN_CONTACTS = {"Alice", "Bob", "Carol"}
CASE_META = {"device_tz": CASE_TZ, "device_model": "synthetic red-team phone", "aleapp_version": "synthetic",
             "report_root": "redteam", "ingested_at": "2026-11-01T00:00:00Z"}


def build_case_store(api: SimpleNamespace, path: pathlib.Path):
    """Create the schema with Store(path), write case_meta with plain SQL (the contract fixes the
    case_meta schema but gives Store no setter, see clarification C4), then reopen and insert rows
    through Store.insert()."""
    api.Store(str(path))
    con = sqlite3.connect(str(path))
    with con:
        con.executemany("INSERT OR REPLACE INTO case_meta(key, value) VALUES (?, ?)", list(CASE_META.items()))
    con.close()
    store = api.Store(str(path))
    for table, row in ROWS:
        store.insert(table, dict(row))
    return store


@dataclass
class Case:
    id: str
    expect: str                      # "verified" | "rejected"
    claim: dict
    codes: frozenset = frozenset()   # exact set of reason codes for must-reject cases
    window: tuple | None = EVENING
    indexes: dict | None = None      # code -> set of citation_index values expected for that code
    detail_contains: dict | None = None
    note: str = ""                   # which contract rule / clarification the case pins down


def cite(key: str, field: str, value: str) -> dict:
    return {"artifact_id": IDS.get(key, key), "field": field, "value": value}


def spike_claims(fixture: str) -> list[dict]:
    """The claims of the real submit_answer call recorded in a spike fixture."""
    rec = json.loads((SPIKES / f"{fixture}.json").read_text())
    for tc in rec["response"]["body"]["choices"][0]["message"]["tool_calls"]:
        if tc["function"]["name"] == "submit_answer":
            return json.loads(tc["function"]["arguments"])["claims"]
    raise AssertionError(f"no submit_answer in {fixture}")


def find_trace(cwd: pathlib.Path, run_id: str) -> pathlib.Path | None:
    """CONTRACTS.md section 8: runs/<run_id>.jsonl, relative to the working directory (or the repo)."""
    for base in (cwd, REPO):
        p = base / "runs" / f"{run_id}.jsonl"
        if p.exists():
            return p
    return None
