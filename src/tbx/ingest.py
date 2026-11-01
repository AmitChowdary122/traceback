"""ALEAPP report (+ extraction media) -> store, with deterministic artifact IDs (CONTRACTS.md 2-4).

- Each supported TSV under `<report>/_TSV Exports/` maps onto one family table.
- source_path is the TSV's path relative to the report root (media: relative to the extraction
  root); source_row is the row_key: a natural key column when the TSV has a unique one, else
  the 1-based data row number. raw_json is the TSV row exactly as ALEAPP wrote it, plus a
  "_tbx" note for anything ingest derived (e.g. a contact name looked up in Contacts.tsv).
- ALEAPP writes timestamps in UTC; a value without an offset is read as UTC (tz_basis utc_string).
  EXIF DateTimeOriginal + OffsetTimeOriginal is converted with that offset (tz_basis offset_string).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pathlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator

from tbx import timeutil as tu
from tbx.checker import digits
from tbx.store import FAMILY, Store, artifact_id

csv.field_size_limit(sys.maxsize)
TSV_DIR = "_TSV Exports"
_TS = re.compile(r"^(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?$")


class IngestError(RuntimeError):
    pass


# ------------------------------------------------------------------------------ parsing helpers
def parse_aleapp_ts(raw: str | None) -> tuple[str | None, str]:
    """ALEAPP timestamp text -> (ts_utc, tz_basis). Values without an offset are UTC (ALEAPP default)."""
    s = (raw or "").strip()
    m = _TS.match(s)
    if not m:
        return None, "unknown"
    y, mo, d, h, mi, se = (int(x) for x in m.groups()[:6])
    try:
        dt = datetime(y, mo, d, h, mi, se, tzinfo=timezone.utc)
    except ValueError:
        return None, "unknown"
    off = m.group(8)
    basis = "utc_string"
    if off and off.upper() != "Z":
        sign = 1 if off[0] == "+" else -1
        hh, mm = int(off[1:3]), int(off[-2:])
        if hh or mm:  # "+00:00" is still a UTC string; any other offset is converted with it
            dt = dt.replace(tzinfo=timezone(sign * timedelta(hours=hh, minutes=mm)))
            basis = "offset_string"
    if dt.year < 1990:
        return None, "unknown"
    return tu.fmt_utc(dt), basis


def parse_duration(raw: str | None) -> float | None:
    s = (raw or "").strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s) if "." in s else int(s)
    if re.fullmatch(r"\d{1,3}(:\d{2}){1,2}", s):
        total = 0
        for part in s.split(":"):
            total = total * 60 + int(part)
        return total
    return None


def _float(raw: str | None) -> float | None:
    try:
        v = float((raw or "").strip())
    except ValueError:
        return None
    return v if v == v else None


def _clean(v: str | None) -> str | None:
    v = (v or "").strip()
    return v if v and v.upper() not in ("N/A", "NONE", "NULL") else None


def is_phone(v: str | None) -> bool:
    """Phone numbers and short codes: digits with phone punctuation only."""
    v = (v or "").strip()
    return bool(v) and bool(re.fullmatch(r"[+()\d\s.\-]+", v)) and len(digits(v)) >= 3


def is_name(v: str | None) -> bool:
    """A person-like display name: has a letter, is not a number or a list of numbers."""
    v = (v or "").strip()
    return (bool(v) and bool(re.search(r"[^\W\d_]", v)) and not is_phone(v) and "@" not in v
            and v.lower() not in ("self", "unknown"))


def jid_number(jid: str | None) -> str | None:
    m = re.match(r"(\d{6,})@s\.whatsapp\.net", (jid or "").strip())
    return f"+{m.group(1)}" if m else None


def _direction(raw: str | None, allowed: tuple[str, ...]) -> str:
    v = (raw or "").strip().lower()
    for a in allowed:
        if v.startswith(a):
            return a
    return "unknown"


MSG_DIR = ("incoming", "outgoing")
CALL_DIR = ("incoming", "outgoing", "missed", "rejected", "voicemail")


# ------------------------------------------------------------------------------ adapters
@dataclass
class Ctx:
    contacts: dict[str, tuple[str, str]]   # last-10 digits -> (display name, provenance)
    media_ids: dict[str, str]              # extraction-relative path -> media artifact_id
    case_tz: str


Mapper = Callable[[dict, Ctx], dict | None]


@dataclass
class Adapter:
    tsv: str
    table: str
    app: str | None
    ts_col: str
    mapper: Mapper
    key_col: str | None = None


def _party(name: str | None, number: str | None) -> tuple[str | None, str | None]:
    """Split one 'participant' value into (name, number)."""
    name, number = _clean(name), _clean(number)
    if name and not is_name(name):
        number = number or (name if is_phone(name) else jid_number(name))
        name = None
    return name, number


def m_google_messages(r: dict, ctx: Ctx) -> dict:
    d = _direction(r.get("Direction"), MSG_DIR)
    name, num = _party(r.get("Other Participant/Conversation Name"), None)
    sender_num = _clean(r.get("Message Sender"))
    att = _clean(r.get("Attachment Location"))
    att_rel = re.sub(r"^/data/user/0/", "data/data/", att).lstrip("/") if att else None
    out = {"thread_id": _clean(r.get("Conversation ID")), "direction": d, "body": r.get("Message") or None,
           "attachment_media_id": ctx.media_ids.get(att_rel) if att_rel else None}
    if d == "incoming":
        out.update(sender=name, sender_number=sender_num if is_phone(sender_num) else num)
    else:
        out.update(recipient=name, recipient_number=num)
    return out


def m_whatsapp_1to1(r: dict, ctx: Ctx) -> dict:
    d = _direction(r.get("Message Direction"), MSG_DIR)
    name, _ = _party(r.get("Other Participant WA User Name"), None)
    num = jid_number(r.get("Other Participant JID"))
    out = {"thread_id": _clean(r.get("Other Participant JID")), "direction": d, "body": r.get("Message") or None}
    if d == "incoming":
        out.update(sender=name, sender_number=num)
    else:
        out.update(recipient=name, recipient_number=num)
    return out


def m_whatsapp_group(r: dict, ctx: Ctx) -> dict:
    d = _direction(r.get("Message Direction"), MSG_DIR)
    name, _ = _party(r.get("Sending Party"), None)
    out = {"thread_id": _clean(r.get("Conversation Name")) or _clean(r.get("Group JID")), "direction": d,
           "body": r.get("Message") or None}
    if d == "incoming":
        out.update(sender=name, sender_number=jid_number(r.get("Sending Party JID")))
    return out


def m_telegram(r: dict, ctx: Ctx) -> dict:
    d = _direction(r.get("Direction"), MSG_DIR)
    out = {"thread_id": _clean(r.get("Dialog ID")), "direction": d, "body": r.get("Message") or None}
    if d == "incoming":
        out["sender"] = _party(r.get("Sender"), None)[0]
    else:
        out["recipient"] = _party(r.get("Chat"), None)[0]
    return out


def m_fb_messenger(r: dict, ctx: Ctx) -> dict:
    return {"thread_id": _clean(r.get("Thread Key")), "direction": _direction(r.get("Direction"), MSG_DIR),
            "sender": _party(r.get("Sender"), None)[0], "body": r.get("Message") or None}


def m_google_voice_msg(r: dict, ctx: Ctx) -> dict:
    return {"thread_id": _clean(r.get("Conversation ID")), "direction": _direction(r.get("Direction"), MSG_DIR),
            "sender_number": _clean(r.get("Sender")), "recipient_number": _clean(r.get("Recipient(s)")),
            "body": r.get("Message") or None}


def _with_contact(row: dict, ctx: Ctx, raw_note: dict) -> dict:
    if not row.get("contact_name") and row.get("number"):
        hit = ctx.contacts.get(digits(row["number"])[-10:])
        if hit:
            row["contact_name"] = hit[0]
            raw_note["contact_name_from"] = hit[1]
    return row


# Android CallLog.Calls.TYPE. 4 is a voicemail (CONTRACTS v1.2), not a missed call; 6 = blocked,
# 7 = answered on another device.
CALL_TYPES = {"1": "incoming", "2": "outgoing", "3": "missed", "4": "voicemail", "5": "rejected", "6": "rejected",
              "7": "incoming"}
TRANSCRIPT_PLACEHOLDERS = {"transcript not available", "transcription not available", "no transcription"}


def voicemail_transcript(raw: str | None) -> str | None:
    """The carrier's voicemail transcription, or None for empty values and placeholders."""
    v = (raw or "").strip()
    if not v or v.casefold().rstrip(".") in TRANSCRIPT_PLACEHOLDERS:
        return None
    return v


def m_call_log(r: dict, ctx: Ctx) -> dict:
    return {"direction": CALL_TYPES.get((r.get("Raw Call Type (as stored)") or "").strip(), "unknown"),
            "number": _clean(r.get("Partner")), "duration_s": parse_duration(r.get("Duration in Secs")),
            "transcript": voicemail_transcript(r.get("Transcription"))}


def m_whatsapp_calls(r: dict, ctx: Ctx) -> dict:
    name, _ = _party(r.get("Caller"), None)
    return {"direction": _direction(r.get("Call Direction"), CALL_DIR), "contact_name": name,
            "number": jid_number(r.get("Caller JID")), "duration_s": parse_duration(r.get("Call Duration"))}


def m_fb_calls(r: dict, ctx: Ctx) -> dict:
    return {"direction": _direction(r.get("Call Direction"), CALL_DIR),
            "contact_name": _party(r.get("Party Name"), None)[0],
            "duration_s": parse_duration(r.get("Call Duration"))}


def m_google_voice_calls(r: dict, ctx: Ctx) -> dict:
    d = _direction(r.get("Direction"), CALL_DIR)
    if d == "incoming" and "not answered" in (r.get("Call Status (existing parser classification)") or "").lower():
        d = "missed"
    other = r.get("Caller") if d in ("incoming", "missed") else r.get("Recipient")
    return {"direction": d, "number": _clean(other), "duration_s": parse_duration(r.get("Duration"))}


def _loc(source: str, lat_col: str, lon_col: str, acc_col: str | None) -> Mapper:
    def m(r: dict, ctx: Ctx) -> dict | None:
        lat, lon = _float(r.get(lat_col)), _float(r.get(lon_col))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            return None
        return {"source": source, "lat": lat, "lon": lon, "accuracy_m": _float(r.get(acc_col)) if acc_col else None}
    return m


ADAPTERS: list[Adapter] = [
    Adapter("GoogleMessages.tsv", "messages", "Google Messages", "Message Timestamp", m_google_messages),
    Adapter("WhatsApp - One To One Messages.tsv", "messages", "WhatsApp", "Message Timestamp", m_whatsapp_1to1),
    Adapter("WhatsApp - Group Messages.tsv", "messages", "WhatsApp", "Message Timestamp", m_whatsapp_group),
    Adapter("Telegram - Messages.tsv", "messages", "Telegram", "Timestamp", m_telegram),
    Adapter("Facebook Messenger - Chats (msys_database).tsv", "messages", "Facebook Messenger", "Message Timestamp",
            m_fb_messenger, key_col="Message ID"),
    Adapter("Google Voice - Messages.tsv", "messages", "Google Voice", "Timestamp", m_google_voice_msg),
    Adapter("Call logs .tsv", "calls", "Phone", "Call Date", m_call_log),
    Adapter("WhatsApp - Call Logs.tsv", "calls", "WhatsApp", "Call Start Timestamp", m_whatsapp_calls),
    Adapter("Facebook Messenger - Calls (msys_database).tsv", "calls", "Facebook Messenger", "Call Timestamp",
            m_fb_calls),
    Adapter("Google Voice - Calls.tsv", "calls", "Google Voice", "Timestamp", m_google_voice_calls),
    Adapter("ODLH Path Points.tsv", "locations", None, "Point Time",
            _loc("google_location_history", "Latitude", "Longitude", None)),
    Adapter("Life360 - Locations.tsv", "locations", None, "Timestamp",
            _loc("life360", "Latitude", "Longitude", "Horizontal Accuracy (as stored)"), key_col="ID"),
    Adapter("Fitbit - Exercise GPS.tsv", "locations", None, "Timestamp",
            _loc("fitbit_exercise_gps", "Latitude", "Longitude", "Accuracy")),
]


# ------------------------------------------------------------------------------ report reading
def read_tsv(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _find_tsv(report: pathlib.Path, name: str) -> pathlib.Path | None:
    p = report / TSV_DIR / name
    return p if p.is_file() else None


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    return path.relative_to(root).as_posix()


def load_contacts(report: pathlib.Path) -> dict[str, tuple[str, str]]:
    p = _find_tsv(report, "Contacts.tsv")
    out: dict[str, tuple[str, str]] = {}
    if not p:
        return out
    for i, r in enumerate(read_tsv(p), 1):
        name, number = _clean(r.get("Display Name")), _clean(r.get("Phone Number")) or _clean(r.get("Data 1"))
        if name and number and is_name(name) and len(digits(number)) >= 7:
            out.setdefault(digits(number)[-10:], (name, f"{_rel(p, report)} row {i}"))
    return out


def _raw(r: dict, note: dict | None = None) -> str:
    d = dict(r)
    if note:
        d["_tbx"] = note
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


def _row_keys(rows: list[dict], key_col: str | None) -> list[str]:
    if key_col:
        keys = [(r.get(key_col) or "").strip() for r in rows]
        if all(keys) and len(set(keys)) == len(keys):
            return keys
    return [str(i) for i in range(1, len(rows) + 1)]


def ingest_tsvs(store: Store, report: pathlib.Path, ctx: Ctx, summary: dict) -> None:
    for ad in ADAPTERS:
        p = _find_tsv(report, ad.tsv)
        if not p:
            continue
        src = _rel(p, report)
        data = read_tsv(p)
        out = []
        for key, r in zip(_row_keys(data, ad.key_col), data):
            mapped = ad.mapper(r, ctx)
            if mapped is None:
                continue
            note: dict = {}
            if ad.table == "calls":
                mapped = _with_contact(mapped, ctx, note)
            if ad.app and "app" in FAMILY[ad.table]:
                mapped.setdefault("app", ad.app)
            ts, basis = parse_aleapp_ts(r.get(ad.ts_col))
            out.append({"artifact_id": artifact_id(ad.table, src, key), "source_path": src, "source_row": key,
                        "ts_utc": ts, "ts_original": r.get(ad.ts_col) or None, "tz_basis": basis,
                        "raw_json": _raw(r, note), **mapped})
        _insert(store, ad.table, out)
        summary["sources"][src] = len(out)
        summary["counts"][ad.table] += len(out)


def _insert(store: Store, table: str, rows: list[dict]) -> None:
    try:
        store.insert_many(table, rows)
    except ValueError as e:
        raise IngestError(str(e)) from None


# ------------------------------------------------------------------------------ media
IMAGE_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
             ".heic": "image/heic", ".webp": "image/webp"}
VIDEO_EXT = {".mp4": "video/mp4", ".3gp": "video/3gpp", ".mov": "video/quicktime", ".webm": "video/webm"}
AUDIO_EXT = {".m4a": "audio/mp4", ".opus": "audio/ogg", ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
             ".wav": "audio/wav", ".amr": "audio/amr", ".aac": "audio/aac"}
SKIP_MEDIA = ("data/property/",)


def sniff(path: pathlib.Path) -> tuple[str, str] | None:
    """-> (kind, mime) from the extension, else from magic bytes (for .bin attachment parts)."""
    ext = path.suffix.lower()
    for table, kind in ((IMAGE_EXT, "image"), (VIDEO_EXT, "video"), (AUDIO_EXT, "audio")):
        if ext in table:
            return kind, table[ext]
    head = path.open("rb").read(16)
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if head.startswith(b"\x89PNG"):
        return "image", "image/png"
    if head.startswith(b"GIF8"):
        return "image", "image/gif"
    if head[4:8] == b"ftyp":
        return ("audio", "audio/mp4") if head[8:11] == b"M4A" else ("video", "video/mp4")
    if head.startswith(b"OggS"):
        return "audio", "audio/ogg"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if head.startswith(b"#!AMR"):
        return "audio", "audio/amr"
    return None


def media_app(rel: str) -> str:
    if "/DCIM/Camera/" in f"/{rel}":
        return "Camera"
    if "com.google.android.apps.messaging" in rel or "com.android.providers.telephony" in rel:
        return "Google Messages"
    if "whatsapp" in rel.lower():
        return "WhatsApp"
    m = re.match(r"data/data/([^/]+)/", rel)
    return m.group(1) if m else "Files"


def _exif(path: pathlib.Path) -> dict:
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769)
            gps = ex.get_ifd(0x8825)
            info: dict = {"width": im.width, "height": im.height}
    except Exception:  # noqa: BLE001 - unreadable images still become media rows
        return {}
    dto = sub.get(36867) or ex.get(306)
    if dto:
        info["DateTimeOriginal"] = str(dto)
    if sub.get(36881):
        info["OffsetTimeOriginal"] = str(sub.get(36881))
    try:
        if gps and 2 in gps and 4 in gps:
            lat = sum(float(v) / 60 ** i for i, v in enumerate(gps[2]))
            lon = sum(float(v) / 60 ** i for i, v in enumerate(gps[4]))
            info["lat"] = -lat if str(gps.get(1, "N")).upper().startswith("S") else lat
            info["lon"] = -lon if str(gps.get(3, "E")).upper().startswith("W") else lon
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return info


def _exif_ts(info: dict, case_tz: str) -> tuple[str | None, str | None, str]:
    dto = info.get("DateTimeOriginal")
    if not dto:
        return None, None, "unknown"
    try:
        naive = datetime.strptime(dto[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, dto, "unknown"
    off = info.get("OffsetTimeOriginal")
    if off and re.fullmatch(r"[+-]\d{2}:\d{2}", off):
        dt = datetime.fromisoformat(naive.isoformat() + off)
        return tu.fmt_utc(dt), f"{dto} {off}", "offset_string"
    return tu.fmt_utc(tu.local_to_utc(naive, case_tz)), dto, f"local_converted:{case_tz}"


def attachment_times(report: pathlib.Path) -> dict[str, tuple[str, str, str]]:
    """Google Messages attachment path -> (raw timestamp, ts_utc, provenance)."""
    p = _find_tsv(report, "GoogleMessages.tsv")
    out: dict[str, tuple[str, str, str]] = {}
    if not p:
        return out
    for i, r in enumerate(read_tsv(p), 1):
        att = _clean(r.get("Attachment Location"))
        ts, _ = parse_aleapp_ts(r.get("Message Timestamp"))
        if att and ts:
            rel = re.sub(r"^/data/user/0/", "data/data/", att).lstrip("/")
            out.setdefault(rel, (r["Message Timestamp"], ts, f"{_rel(p, report)} row {i}"))
    return out


def _iter_media(root: pathlib.Path) -> Iterator[tuple[pathlib.Path, str]]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            path = pathlib.Path(dirpath) / fn
            rel = _rel(path, root)
            if not rel.startswith(SKIP_MEDIA) and path.is_file() and not path.is_symlink():
                yield path, rel


def ingest_media(store: Store, root: pathlib.Path, report: pathlib.Path, ctx: Ctx, summary: dict) -> None:
    manifest_path = root.parent / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()).get("media", {}) if manifest_path.is_file() else {}
    att_times = attachment_times(report)
    media_rows, loc_rows = [], []
    for path, rel in _iter_media(root):
        kind_mime = sniff(path)
        if kind_mime is None:
            continue
        kind, mime = kind_mime
        data = path.read_bytes()
        info = _exif(path) if kind == "image" else {}
        note: dict = {}
        # A received attachment's event on this device is its message time; its EXIF (if any) is the
        # sender's camera clock, so it is kept in raw_json but not used as ts_utc.
        ts, ts_orig, basis = (None, None, "unknown") if rel in att_times else _exif_ts(info, ctx.case_tz)
        if ts:
            note["ts_from"] = "EXIF DateTimeOriginal"
        elif rel in att_times:
            ts_orig, ts, src = att_times[rel]
            basis = "utc_string"
            note["ts_from"] = f"message timestamp, {src}"
        elif rel in manifest and manifest[rel].get("original_mtime"):
            mtime = manifest[rel]["original_mtime"]
            ts, ts_orig, basis = tu.fmt_utc(tu.from_epoch(mtime, "s")), str(mtime), "utc_epoch_s"
            note["ts_from"] = "file mtime in the extraction (MANIFEST.json)"
        if rel in manifest:
            note["manifest"] = manifest[rel]
        mid = artifact_id("media", rel, "1")
        ctx.media_ids[rel] = mid
        raw = {"file": rel, "exif": info} if info else {"file": rel}
        media_rows.append({"artifact_id": mid, "source_path": rel, "source_row": "1", "ts_utc": ts,
                           "ts_original": ts_orig, "tz_basis": basis, "raw_json": _raw(raw, note),
                           "app": media_app(rel), "kind": kind, "file_path": rel, "mime_type": mime,
                           "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
                           "lat": info.get("lat"), "lon": info.get("lon")})
        if info.get("lat") is not None and info.get("lon") is not None:
            loc_rows.append({"artifact_id": artifact_id("locations", rel, "1"), "source_path": rel, "source_row": "1",
                             "ts_utc": ts, "ts_original": ts_orig, "tz_basis": basis,
                             "raw_json": _raw(raw, {**note, "derived_from_media": mid}), "source": "photo_exif",
                             "lat": info["lat"], "lon": info["lon"], "accuracy_m": None})
    _insert(store, "media", media_rows)
    _insert(store, "locations", loc_rows)
    summary["counts"]["media"] += len(media_rows)
    summary["counts"]["locations"] += len(loc_rows)
    summary["sources"]["<media root>"] = len(media_rows)


# ------------------------------------------------------------------------------ case_meta
def read_protobuf_props(data: bytes) -> dict[str, str]:
    """Android persistent_properties: repeated {1: name, 2: value} messages under field 1."""
    def varint(buf: bytes, i: int) -> tuple[int, int]:
        shift = result = 0
        while True:
            b = buf[i]
            i += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result, i
            shift += 7

    def fields(buf: bytes) -> Iterator[tuple[int, bytes]]:
        i = 0
        while i < len(buf):
            key, i = varint(buf, i)
            num, wt = key >> 3, key & 7
            if wt == 2:
                n, i = varint(buf, i)
                yield num, buf[i:i + n]
                i += n
            elif wt == 0:
                _, i = varint(buf, i)
            else:
                return
    out = {}
    try:
        for num, entry in fields(data):
            if num != 1:
                continue
            kv = dict(fields(entry))
            if 1 in kv:
                out[kv[1].decode("utf-8", "replace")] = kv.get(2, b"").decode("utf-8", "replace")
    except IndexError:
        pass
    return out


def device_tz(media_root: pathlib.Path | None) -> tuple[str | None, str | None]:
    if media_root is None:
        return None, None
    p = media_root / "data/property/persistent_properties"
    if p.is_file():
        tz = read_protobuf_props(p.read_bytes()).get("persist.sys.timezone")
        if tz and tu.valid_tz(tz):
            return tz, "data/property/persistent_properties persist.sys.timezone"
    return None, None


def device_model(report: pathlib.Path) -> str:
    p = _find_tsv(report, "Build.tsv")
    if not p:
        return "unknown"
    kv = {(r.get("Label") or "").strip(): (r.get("Value") or "").strip() for r in read_tsv(p)}
    return " ".join(x for x in (kv.get("Manufacturer"), kv.get("Model")) if x) or "unknown"


def aleapp_version(report: pathlib.Path) -> str:
    p = report / "_lava_data.lava"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))["parser_info"]["leapp_version"]
        except (ValueError, KeyError, TypeError):
            pass
    return "unknown"


# ------------------------------------------------------------------------------ entry point
def ingest_report(report_dir: str, db_path: str, media_root: str | None = None) -> dict:
    report = pathlib.Path(report_dir)
    if not report.is_dir():
        raise IngestError(f"report directory not found: {report_dir}")
    if not (report / TSV_DIR).is_dir():
        raise IngestError(f"no '{TSV_DIR}' folder in {report_dir}: is this an ALEAPP report?")
    mroot = pathlib.Path(media_root) if media_root else None
    if mroot is not None and not mroot.is_dir():
        raise IngestError(f"media root not found: {media_root}")
    tmp = pathlib.Path(f"{db_path}.ingest-tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    summary: dict = {"db": str(db_path), "counts": {t: 0 for t in FAMILY}, "sources": {}, "warnings": []}
    store = Store(str(tmp))
    try:
        tz, tz_src = device_tz(mroot)
        if tz is None:
            tz = "UTC"
            summary["warnings"].append("device_tz not found in the extraction (data/property/persistent_properties);"
                                       " using UTC")
        meta = {"device_tz": tz, "device_tz_source": tz_src or "default", "device_model": device_model(report),
                "aleapp_version": aleapp_version(report), "report_root": report.resolve().name,
                "ingested_at": tu.now_utc()}
        for k, v in meta.items():
            store.set_meta(k, v)
        ctx = Ctx(contacts=load_contacts(report), media_ids={}, case_tz=tz)
        if mroot is not None:
            ingest_media(store, mroot, report, ctx, summary)
        ingest_tsvs(store, report, ctx, summary)
        summary["case_meta"] = meta
    except BaseException:
        store.close()
        tmp.unlink(missing_ok=True)
        raise
    store.close()
    os.replace(tmp, db_path)
    return summary
