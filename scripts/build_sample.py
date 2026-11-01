"""Cut data/sample/ from a full ALEAPP report and the extracted file system it was run on.

    python scripts/build_sample.py --report <ALEAPP_Output_dir> --extraction <fs_root> --out data/sample

- TSVs are copied verbatim, so row numbers (and artifact IDs) match the full report.
- Media files keep their extraction-relative paths under <out>/media/. Camera photos are
  downscaled (max 1600 px, JPEG q85, EXIF kept) to keep the repo small; everything else is
  copied byte for byte. MANIFEST.json records both hashes for every file.
Needs Pillow (not a tbx dependency): pip install pillow
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import re
import shutil

from PIL import Image

TSVS = [
    # messages
    "GoogleMessages.tsv", "WhatsApp - One To One Messages.tsv", "WhatsApp - Group Messages.tsv",
    "Telegram - Messages.tsv", "Facebook Messenger - Chats (msys_database).tsv", "Google Voice - Messages.tsv",
    # calls
    "Call logs .tsv", "WhatsApp - Call Logs.tsv", "Facebook Messenger - Calls (msys_database).tsv",
    "Google Voice - Calls.tsv",
    # locations
    "ODLH Path Points.tsv", "Life360 - Locations.tsv", "Fitbit - Exercise GPS.tsv",
    # support: contact names, device model
    "Contacts.tsv", "Build.tsv",
]
VERBATIM = [
    "data/property/persistent_properties",  # persist.sys.timezone -> case_meta.device_tz
]
MESSAGING_CACHE = "data/data/com.google.android.apps.messaging/cache/telephony_cache"
CAMERA = "data/media/0/DCIM/Camera"
PHOTO_FROM = "20240615"   # camera photos taken on or after this UTC date (Pixel names are UTC)
MAX_PX, QUALITY = 1600, 85


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def downscale(raw: bytes) -> bytes:
    im = Image.open(io.BytesIO(raw))
    exif = im.info.get("exif")
    im.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=QUALITY, exif=exif or b"", optimize=True)
    return buf.getvalue()


def attachment_paths(google_messages_tsv: pathlib.Path) -> list[str]:
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)
    out = []
    with open(google_messages_tsv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            loc = (row.get("Attachment Location") or "").strip()
            # Third-party documents (a 7 MB PDF that carries its own copyright) are left out.
            if loc and (row.get("Message Type") or "").startswith(("image/", "video/", "audio/")):
                # /data/user/0 is the per-user alias of /data/data on Android
                out.append(re.sub(r"^/data/user/0/", "data/data/", loc).lstrip("/"))
    return sorted(set(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--extraction", required=True)
    ap.add_argument("--out", default="data/sample")
    a = ap.parse_args()
    report, fs, out = pathlib.Path(a.report), pathlib.Path(a.extraction), pathlib.Path(a.out)
    manifest: dict = {"tsv": {}, "media": {}}

    tsv_out = out / "aleapp" / "_TSV Exports"
    if (out / "aleapp").exists():
        shutil.rmtree(out / "aleapp")
    tsv_out.mkdir(parents=True)
    for name in TSVS:
        src = report / "_TSV Exports" / name
        data = src.read_bytes()
        (tsv_out / name).write_bytes(data)
        manifest["tsv"][f"_TSV Exports/{name}"] = {"sha256": sha256(data), "size": len(data)}
    lava = report / "_lava_data.lava"
    shutil.copyfile(lava, out / "aleapp" / "_lava_data.lava")
    manifest["tsv"]["_lava_data.lava"] = {"sha256": sha256(lava.read_bytes()), "size": lava.stat().st_size}

    media_out = out / "media"
    if media_out.exists():
        shutil.rmtree(media_out)
    wanted = VERBATIM + [p for p in attachment_paths(report / "_TSV Exports" / "GoogleMessages.tsv")
                         if p.startswith(MESSAGING_CACHE) and (fs / p).is_file()]
    photos = sorted(p for p in (fs / CAMERA).glob("PXL_*.jpg") if p.name[4:12] >= PHOTO_FROM)
    for rel in wanted + [str(p.relative_to(fs)) for p in photos]:
        raw = (fs / rel).read_bytes()
        is_photo = rel.startswith(CAMERA)
        data = downscale(raw) if is_photo else raw
        dst = media_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        st = (fs / rel).stat()
        manifest["media"][rel] = {
            "sha256": sha256(data), "size": len(data), "original_sha256": sha256(raw), "original_size": len(raw),
            "original_mtime": int(st.st_mtime),
            "derivation": f"downscaled to max {MAX_PX} px, JPEG q{QUALITY}, EXIF kept" if is_photo else "verbatim"}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(v["size"] for v in manifest["tsv"].values()) + sum(v["size"] for v in manifest["media"].values())
    print(f"{len(manifest['tsv'])} report files, {len(manifest['media'])} media files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
