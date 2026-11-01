"""tbx.enrich offline: request shapes, cache behaviour, box parsing, n ordering (CONTRACTS v1.2 sections 2 and 9).

No network: the HTTP client is an httpx.MockTransport that records every request.
"""
import hashlib
import json
import pathlib
import subprocess

import httpx
import pytest

from tbx import enrich as en
from tbx.store import Store, artifact_id

SPIKES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "spikes"
SAM_TEXT = json.loads((SPIKES / "s3_sam_image_nostream_jpeg_blue_circle.json").read_text())[
    "response"]["body"]["output"][0]["content"][0]["text"]


def test_parse_sam_fixture_text():
    boxes = en.parse_sam_text(SAM_TEXT)
    assert boxes[0] == {"x0": 420, "y0": 300, "x1": 520, "y1": 400, "score": pytest.approx(0.9726027250289917),
                        "w": 640, "h": 480}
    assert len(boxes) == 2 and en.parse_sam_text("") == [] and en.parse_sam_text(None) == []


def test_kept_boxes_floor_order_scale_and_clamp():
    result = {"width": 200, "height": 100, "boxes": [
        {"x0": 10, "y0": 10, "x1": 50, "y1": 50, "score": 0.7, "w": 200, "h": 100},
        {"x0": 5, "y0": 30, "x1": 40, "y1": 60, "score": 0.9, "w": 200, "h": 100},
        {"x0": 1, "y0": 5, "x1": 20, "y1": 20, "score": 0.9, "w": 200, "h": 100},   # same score, smaller x0 first
        {"x0": 0, "y0": 0, "x1": 9, "y1": 9, "score": 0.49, "w": 200, "h": 100},    # under the floor
        {"x0": 150, "y0": 50, "x1": 260, "y1": 120, "score": 0.6, "w": 200, "h": 100},  # clamped to the image
        {"x0": 10, "y0": 10, "x1": 20, "y1": 20, "score": 0.8, "w": 400, "h": 200},  # SAM saw 2x: scaled
        {"x0": 10, "y0": 10, "x1": 50, "y1": 50, "score": None},                     # no confidence: dropped
    ]}
    kept = en.kept_boxes(result)
    assert [b for b, _, _ in kept] == [[1, 5, 20, 20], [5, 30, 40, 60], [5, 5, 10, 10], [10, 10, 50, 50],
                                       [150, 50, 200, 100]]
    assert all(isinstance(v, int) for b, _, _ in kept for v in b)


class Recorder:
    def __init__(self, voice_body=None, sam_text=SAM_TEXT, status=200):
        self.requests: list[httpx.Request] = []
        self.voice_body = voice_body or {"sessionId": "voyager/transcribe/X", "transcript": "the attorney general said",
                                         "audioDurationMs": 15000, "turns": [{"turnId": 0, "startMs": 0, "endMs": 15000,
                                                                              "transcript": "the attorney general said"}]}
        self.sam_text, self.status = sam_text, status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope Bearer KEY-123"}})
        if request.url.path.endswith("/asr/transcribe"):
            return httpx.Response(200, json=self.voice_body)
        return httpx.Response(200, json={"id": "resp_1", "status": "completed",
                                         "output": [{"content": [{"type": "output_text", "text": self.sam_text}]}]})

    def api(self):
        return en.Api("https://api.example.test/v1", "KEY-123", client=httpx.Client(transport=httpx.MockTransport(self)))


def _media_row(path: pathlib.Path, root: pathlib.Path, kind: str, mime: str, ts="2024-02-09T21:24:35Z") -> dict:
    rel = path.relative_to(root).as_posix()
    data = path.read_bytes()
    return {"artifact_id": artifact_id("media", rel, "1"), "source_path": rel, "source_row": "1", "ts_utc": ts,
            "ts_original": ts, "tz_basis": "utc_string", "raw_json": "{}", "app": "Google Messages", "kind": kind,
            "file_path": rel, "mime_type": mime, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "lat": None, "lon": None}


@pytest.fixture
def case(tmp_path):
    from PIL import Image
    root = tmp_path / "media"
    (root / "DCIM").mkdir(parents=True)
    Image.new("RGB", (640, 480), "white").save(root / "DCIM" / "a.jpg")
    im = Image.new("RGB", (60, 40), "white")          # stored 60x40, displayed 40x60 (orientation 6)
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(root / "DCIM" / "rot.jpg", exif=exif)
    Image.new("RGB", (8, 8)).save(root / "DCIM" / "anim.gif")
    exe = en.ffmpeg_exe()
    subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-f", "lavfi", "-i",
                    "color=c=black:s=32x32:d=1", "-shortest", "-c:v", "libx264", "-c:a", "aac", "-y",
                    str(root / "clip.mp4")], check=True)
    subprocess.run([exe, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=1", "-c:v", "libx264", "-y",
                    str(root / "silent.mp4")], check=True)
    db = tmp_path / "t.db"
    s = Store(str(db))
    s.set_meta("device_tz", "America/New_York")
    s.set_meta("media_root", str(root))
    for name, kind, mime in (("DCIM/a.jpg", "image", "image/jpeg"), ("DCIM/rot.jpg", "image", "image/jpeg"),
                             ("DCIM/anim.gif", "image", "image/gif"), ("clip.mp4", "video", "video/mp4"),
                             ("silent.mp4", "video", "video/mp4")):
        s.insert("media", _media_row(root / name, root, kind, mime))
    s.close()
    return db, root, tmp_path / "cache.jsonl"


def _run(db, cache, rec=None, **kw):
    factory = rec.api if rec else (lambda: pytest.fail("no API call expected"))
    return en.enrich(str(db), cache_path=str(cache), api_factory=factory, workers=2, **kw)


def _rows(db, table):
    s = Store(str(db))
    try:
        return s.query(f"SELECT * FROM {table} ORDER BY artifact_id")
    finally:
        s.close()


def test_live_pass_request_shapes_and_rows(case):
    db, root, cache = case
    rec = Recorder()
    out = _run(db, cache, rec)
    voice = [r for r in rec.requests if r.url.path.endswith("/asr/transcribe")]
    sam = [r for r in rec.requests if r.url.path.endswith("/responses")]
    assert len(voice) == 1, "only the video with an audio track is transcribed"
    assert out["voice"]["no_audio"] == 1
    body = voice[0].content
    assert b'name="request"' in body and b'name="audio"' in body and b"RIFF" in body and b"WAVE" in body
    req_json = json.loads(body.split(b'name="request"')[1].split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0])
    assert req_json == {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode": "ENDPOINTING"}
    assert "keywords" not in json.dumps(req_json)
    assert voice[0].headers["authorization"] == "Bearer KEY-123"
    assert len(sam) == 2 * 6, "two JPEGs x six labels; the GIF is skipped"
    labels = set()
    for r in sam:
        j = json.loads(r.content)
        assert j["model"] == "sam-3.1" and j["stream"] is False and j["metadata"] == {"include_confidence": "true"}
        content = j["input"][0]["content"]
        labels.add(content[0]["text"])
        assert content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert labels == set(en.PROMPT_SETS["v1"])
    trn = _rows(db, "transcripts")
    assert len(trn) == 1 and trn[0]["text"] == "the attorney general said" and trn[0]["duration_s"] == 15.0
    assert trn[0]["language"] is None and "voyager/transcribe/X" in trn[0]["raw_json"]
    media_id = trn[0]["media_id"]
    assert trn[0]["source_row"] == f"{media_id}|muse-voice-transcribe-1.0"
    assert trn[0]["artifact_id"] == artifact_id("transcripts", trn[0]["source_path"], trn[0]["source_row"])
    tags = _rows(db, "image_tags")
    assert {t["label"] for t in tags} == set(en.PROMPT_SETS["v1"])
    assert all(json.loads(t["box"]) and t["prompt_set"] == "v1" and t["model"] == "sam-3.1" for t in tags)
    assert "KEY-123" not in cache.read_text()
    assert out["est_cost_usd"] > 0 and out["cache_added"] == 1 + 12


def test_cache_hit_makes_no_call_and_ids_are_stable(case):
    db, root, cache = case
    _run(db, cache, Recorder())
    first = {t: [r["artifact_id"] for r in _rows(db, t)] for t in ("transcripts", "image_tags")}
    lines = cache.read_text()
    out = _run(db, cache)                              # any API call fails the test
    assert out["voice"]["hits"] == 1 and out["images"]["hits"] == 12 and out["cache_added"] == 0
    assert {t: [r["artifact_id"] for r in _rows(db, t)] for t in first} == first
    assert cache.read_text() == lines
    out = _run(db, cache, offline=True)
    assert out["voice"]["misses"] == 0 and out["images"]["misses"] == 0


def test_offline_skips_and_counts_misses(case):
    db, root, cache = case
    out = _run(db, cache, offline=True)
    assert out["voice"]["misses"] == 1 and out["voice"]["no_audio"] == 1
    assert out["images"]["misses"] == 12 and out["images"]["rows"] == 0
    assert _rows(db, "image_tags") == [] and not cache.exists()


def test_exif_orientation_is_applied_before_sending(case):
    db, root, cache = case
    data, mime, w, h = en.oriented_image(root / "DCIM" / "rot.jpg", "image/jpeg")
    assert (w, h) == (40, 60)
    from PIL import Image
    import io
    with Image.open(io.BytesIO(data)) as im:
        assert im.size == (40, 60)
    raw, _, w2, h2 = en.oriented_image(root / "DCIM" / "a.jpg", "image/jpeg")
    assert raw == (root / "DCIM" / "a.jpg").read_bytes() and (w2, h2) == (640, 480)


def test_provider_errors_are_reported_not_cached(case):
    db, root, cache = case
    out = _run(db, cache, Recorder(status=500))
    assert out["errors"] and all("KEY-123" not in e for e in out["errors"])
    assert not cache.exists() or cache.read_text() == ""


def test_empty_transcript_gives_no_row(case):
    db, root, cache = case
    rec = Recorder(voice_body={"sessionId": "s", "transcript": "", "audioDurationMs": 1000, "turns": []})
    _run(db, cache, rec, images=False)
    assert _rows(db, "transcripts") == []


def test_missing_key_is_a_usage_error(case, monkeypatch):
    db, root, cache = case
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(en.EnrichError, match="META_API_KEY"):
        en.enrich(str(db), cache_path=str(cache), workers=1)
