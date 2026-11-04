"""Acceptance: `tbx enrich --voice --images --offline` (CONTRACTS.md v1.2, sections 2 and 9).

Offline only: everything comes from the committed cache data/sample/derived/enrich_cache.jsonl,
so these tests make no API calls. The cache itself is produced once, live, by the builder.
Facts below come from probes on the real sample (Nov 1):
- both MMS videos have speech: campaign ads ("… Attorney General …"), 15 s and 30 s of audio
- PXL_20240724_163225*.jpg shows two phones and a laptop (SAM: cell phone x2 at 0.98, laptop 0.96)
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess

import pytest
import tbxcase

REPO = tbxcase.REPO
SAMPLE = REPO / "data" / "sample"
CACHE = SAMPLE / "derived" / "enrich_cache.jsonl"
LABELS_V1 = {"person", "cell phone", "laptop", "car", "license plate", "sign"}
pytestmark = pytest.mark.p1


def _rows(db, sql, *args):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


@pytest.fixture(scope="module")
def enriched(tbx_cli, tmp_path_factory):
    if not CACHE.exists():
        tbxcase.unavailable(f"{CACHE.relative_to(REPO)} not committed yet")
    tmp = tmp_path_factory.mktemp("enrich")
    out = []
    for name in ("a.db", "b.db"):
        db = str(tmp / name)
        for args in (["ingest", str(SAMPLE / "aleapp"), "--media-root", str(SAMPLE / "media"), "--db", db],
                     ["enrich", "--voice", "--images", "--offline", "--db", db]):
            proc = subprocess.run(tbx_cli + args, capture_output=True, text=True, timeout=900, cwd=tmp)
            assert proc.returncode == 0, f"tbx {' '.join(args)}\n{proc.stderr[-3000:]}"
        out.append(db)
    return out


def test_enrich_twice_gives_identical_ids(enriched):
    a, b = enriched
    for table in ("transcripts", "image_tags"):
        ids_a = [r["artifact_id"] for r in _rows(a, f"SELECT artifact_id FROM {table} ORDER BY 1")]
        ids_b = [r["artifact_id"] for r in _rows(b, f"SELECT artifact_id FROM {table} ORDER BY 1")]
        assert ids_a and ids_a == ids_b, table


def test_mms_soundtracks_are_transcribed(enriched):
    trn = _rows(enriched[0], "SELECT t.*, m.kind, m.ts_utc AS media_ts, m.source_path AS media_sp "
                             "FROM transcripts t JOIN media m ON m.artifact_id = t.media_id")
    assert len(trn) >= 2, f"expected the 2 MMS video soundtracks, got {len(trn)} transcripts"
    for r in trn:
        assert r["model"] == "muse-voice-transcribe-1.0" and r["kind"] in ("audio", "video"), r["artifact_id"]
        assert r["text"].strip() and r["ts_utc"] == r["media_ts"] and r["duration_s"] > 0
        assert r["language"] is None
        assert r["source_path"] == r["media_sp"] and r["source_row"] == f"{r['media_id']}|{r['model']}"
        key = r["source_path"] + "|" + r["source_row"]
        assert r["artifact_id"] == "trn_" + hashlib.sha1(key.encode()).hexdigest()[:10]
        assert "sessionId" in r["raw_json"], "keep the Voice Transcribe sessionId in raw_json"
    assert any("attorney general" in r["text"].lower() for r in trn)


def test_image_tags_follow_prompt_set_v1(enriched):
    from PIL import Image, ImageOps
    tags = _rows(enriched[0], "SELECT t.*, m.file_path FROM image_tags t JOIN media m ON m.artifact_id = t.media_id")
    assert len(tags) >= 10, len(tags)
    sizes = {}
    for r in tags:
        assert r["model"] == "sam-3.1" and r["prompt_set"] == "v1" and r["label"] in LABELS_V1, r
        assert 0.5 <= r["score"] <= 1.0, r["score"]
        box = json.loads(r["box"])
        assert isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box), r["box"]
        if r["file_path"] not in sizes:
            with Image.open(SAMPLE / "media" / r["file_path"]) as im:
                sizes[r["file_path"]] = ImageOps.exif_transpose(im).size
        w, h = sizes[r["file_path"]]
        assert 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h, (r["file_path"], box, (w, h))


def test_image_tag_ids_use_the_n_ordering(enriched):
    """v1.2 section 2: n orders a media's boxes for one label by score desc, then x0, then y0."""
    tags = _rows(enriched[0], "SELECT * FROM image_tags")
    groups: dict = {}
    for r in tags:
        groups.setdefault((r["media_id"], r["label"]), []).append(r)
    for (media_id, label), rs in groups.items():
        rs.sort(key=lambda r: (-r["score"], json.loads(r["box"])[0], json.loads(r["box"])[1]))
        for n, r in enumerate(rs):
            key = f"{r['source_path']}|{media_id}|sam-3.1|{label}|{n}"
            assert r["artifact_id"] == "tag_" + hashlib.sha1(key.encode()).hexdigest()[:10], (media_id, label, n)


def test_phones_photo_has_two_cell_phones(enriched):
    rows = _rows(enriched[0], "SELECT t.label, t.score FROM image_tags t JOIN media m ON m.artifact_id = t.media_id "
                              "WHERE m.file_path LIKE '%PXL_20240724_163225%'")
    assert sum(1 for r in rows if r["label"] == "cell phone") >= 2, rows
