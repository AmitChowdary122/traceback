"""`tbx enrich`: voice transcripts and SAM 3.1 image tags, cached by content (CONTRACTS.md v1.2 sections 2, 4, 9).

- Voice: every media row of kind audio, or video with an audio track, is transcoded to 24 kHz mono
  s16le WAV (ffmpeg from imageio-ffmpeg) and sent to Muse Voice Transcribe (POST /asr/transcribe,
  mode ENDPOINTING). No `keywords` are ever sent: keyword biasing can put case names into evidence.
- Images: every JPEG/PNG media row is sent to SAM 3.1 (POST /responses, not streamed,
  include_confidence "true") once per label of prompt set v1, after EXIF orientation is applied.
  Boxes with score >= 0.5 become image_tags rows.
- Cache: one JSON object per line, {"key": "<sha256>|<model>[|<prompt_set>|<label>]", "result": {...}}.
  A cache hit makes no API call; --offline uses only the cache. The cache holds every box SAM
  returned (not only those over the floor), so the floor can change without new calls.

Keys come from the environment and are never logged, cached or echoed in errors.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pathlib
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from tbx.store import Store, artifact_id

VOICE_MODEL = "muse-voice-transcribe-1.0"
SAM_MODEL = "sam-3.1"
PROMPT_SET = "v1"
PROMPT_SETS = {"v1": ("person", "cell phone", "laptop", "car", "license plate", "sign")}
SCORE_FLOOR = 0.5
SAM_MIMES = ("image/jpeg", "image/png")
VOICE_REQUEST = {"model": VOICE_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"}  # never add "keywords"
FFMPEG_ARGS = ["-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-map_metadata", "-1"]
PRICE_SAM_PER_REQUEST = 0.0025       # $2.50 per 1,000 images, per label (spike S3)
PRICE_VOICE_PER_SECOND = 0.18 / 3600  # $0.18 per hour, billed in whole seconds (spike S2)
DEFAULT_CACHE = pathlib.Path("data/sample/derived/enrich_cache.jsonl")
_BOX = re.compile(r"<\|box;([^|>]*)\|>")


class EnrichError(RuntimeError):
    """Usage or input problem (CLI exit 1)."""


class ProviderError(RuntimeError):
    """Network or HTTP failure talking to the API (CLI exit 3)."""


# ------------------------------------------------------------------------------ keys and parsing
def voice_key(sha256: str) -> str:
    return f"{sha256}|{VOICE_MODEL}"


def sam_key(sha256: str, label: str, prompt_set: str = PROMPT_SET) -> str:
    return f"{sha256}|{SAM_MODEL}|{prompt_set}|{label}"


def parse_sam_text(text: str | None) -> list[dict]:
    """SAM's token grammar -> [{x0, y0, x1, y1, score, w, h}] in the pixels SAM saw.

    `<|box;x1=40;y1=60;x2=140;y2=160;w=640;h=480;c=0.97|><|mask;...|>,1<|box;...|>...`
    """
    out = []
    for m in _BOX.finditer(text or ""):
        kv: dict[str, float] = {}
        for part in m.group(1).split(";"):
            k, sep, v = part.partition("=")
            if sep:
                try:
                    kv[k.strip()] = float(v)
                except ValueError:
                    pass
        if not {"x1", "y1", "x2", "y2"} <= set(kv):
            continue
        box = {"x0": kv["x1"], "y0": kv["y1"], "x1": kv["x2"], "y1": kv["y2"], "score": kv.get("c")}
        if "w" in kv and "h" in kv:
            box["w"], box["h"] = kv["w"], kv["h"]
        out.append({k: (int(v) if isinstance(v, float) and v.is_integer() and k != "score" else v)
                    for k, v in box.items()})
    return out


def sam_output_text(body: dict) -> str:
    try:
        return body["output"][0]["content"][0]["text"] or ""
    except (KeyError, IndexError, TypeError):
        return ""  # zero matches can come back with no output text (spike S3)


def kept_boxes(result: dict, floor: float = SCORE_FLOOR) -> list[tuple[list[int], float, dict]]:
    """Cached SAM result -> [(box ints in stored-file pixels after orientation, score, raw box)], with n order.

    n orders a media's boxes for one label by score desc, then x0, then y0 (CONTRACTS v1.2 section 2).
    """
    w, h = int(result["width"]), int(result["height"])
    out = []
    for b in result.get("boxes", []):
        score = b.get("score")
        if score is None or float(score) < floor:
            continue
        sx = w / float(b["w"]) if b.get("w") else 1.0
        sy = h / float(b["h"]) if b.get("h") else 1.0
        x0 = max(0, min(w, round(float(b["x0"]) * sx)))
        y0 = max(0, min(h, round(float(b["y0"]) * sy)))
        x1 = max(0, min(w, round(float(b["x1"]) * sx)))
        y1 = max(0, min(h, round(float(b["y1"]) * sy)))
        if x0 >= x1 or y0 >= y1:
            continue
        out.append(([x0, y0, x1, y1], float(score), b))
    out.sort(key=lambda t: (-t[1], t[0][0], t[0][1]))
    return out


# ------------------------------------------------------------------------------ cache
class Cache:
    """Append-only JSONL while running; rewritten sorted by key at the end, so the committed file is stable."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.data: dict[str, dict] = {}
        self.added = 0
        self._lock = threading.Lock()
        if self.path.is_file():
            for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["result"]
                except (ValueError, KeyError, TypeError) as e:
                    raise EnrichError(f"{self.path}:{n}: not a cache line ({e})") from None

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, result: dict) -> None:
        with self._lock:
            self.data[key] = result
            self.added += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(_line(key, result))

    def compact(self) -> None:
        if not self.added:
            return
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(_line(k, self.data[k]) for k in sorted(self.data)), encoding="utf-8")
        os.replace(tmp, self.path)


def _line(key: str, result: dict) -> str:
    return json.dumps({"key": key, "result": result}, ensure_ascii=False, sort_keys=True) + "\n"


# ------------------------------------------------------------------------------ media helpers
def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def has_audio_track(path: pathlib.Path) -> bool:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True,
                          timeout=60)
    return bool(re.search(r"Stream #\S+.*: Audio:", proc.stderr))


def to_wav(path: pathlib.Path) -> bytes:
    """24 kHz mono s16le WAV, the only format Voice Transcribe accepts (spike S2)."""
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "audio.wav"
        proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                               *FFMPEG_ARGS, str(out)], capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not out.is_file():
            raise EnrichError(f"ffmpeg could not extract audio from {path.name}: {proc.stderr.strip()[-300:]}")
        return out.read_bytes()


def oriented_image(path: pathlib.Path, mime: str) -> tuple[bytes, str, int, int]:
    """-> (bytes to send, mime, width, height) after EXIF orientation. Unrotated files are sent as-is."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        orientation = im.getexif().get(0x0112, 1)
        if orientation in (None, 1):
            return path.read_bytes(), mime, im.width, im.height
        rot = ImageOps.exif_transpose(im)
        buf = io.BytesIO()
        if mime == "image/png":
            rot.save(buf, format="PNG")
        else:
            rot.convert("RGB").save(buf, format="JPEG", quality=95)
        return buf.getvalue(), mime, rot.width, rot.height


# ------------------------------------------------------------------------------ API calls
@dataclass
class Api:
    """Meta endpoints for Voice Transcribe and SAM. `client` is an httpx.Client (injectable for tests)."""

    base_url: str
    api_key: str
    client: Any = None
    timeout: float = 120.0

    def __post_init__(self):
        if self.client is None:
            import httpx
            self.client = httpx.Client(timeout=self.timeout)
        self.base_url = self.base_url.rstrip("/")

    def _post(self, path: str, **kwargs) -> dict:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        try:
            r = self.client.post(f"{self.base_url}{path}", headers=headers, **kwargs)
        except httpx.HTTPError as e:
            raise ProviderError(f"POST {path} failed: {type(e).__name__}") from None
        if r.status_code != 200:
            raise ProviderError(f"POST {path} returned HTTP {r.status_code}: {self._safe(r.text)}")
        try:
            return r.json()
        except ValueError:
            raise ProviderError(f"POST {path} returned a non-JSON body") from None

    def _safe(self, text: str) -> str:
        return (text or "")[:300].replace(self.api_key, "<redacted>") if self.api_key else (text or "")[:300]

    def transcribe(self, wav: bytes) -> dict:
        files = {"request": (None, json.dumps(VOICE_REQUEST), "application/json"),
                 "audio": ("audio.wav", wav, "audio/wav")}
        body = self._post("/asr/transcribe", files=files)
        return {"transcript": body.get("transcript") or "", "audioDurationMs": body.get("audioDurationMs"),
                "turns": body.get("turns") or [], "sessionId": body.get("sessionId")}

    def segment(self, image: bytes, mime: str, label: str) -> dict:
        url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        body = self._post("/responses", json={
            "model": SAM_MODEL, "stream": False, "metadata": {"include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": label}, {"type": "input_image", "image_url": url}]}]})
        if body.get("status") not in (None, "completed"):
            raise ProviderError(f"SAM response status {body.get('status')!r}")
        return {"boxes": parse_sam_text(sam_output_text(body)), "response_id": body.get("id")}


# ------------------------------------------------------------------------------ the run
@dataclass
class Summary:
    voice: dict = field(default_factory=lambda: {"media": 0, "hits": 0, "calls": 0, "misses": 0, "rows": 0,
                                                 "no_audio": 0, "audio_s": 0.0})
    images: dict = field(default_factory=lambda: {"media": 0, "requests": 0, "hits": 0, "calls": 0, "misses": 0,
                                                  "rows": 0})
    errors: list = field(default_factory=list)
    est_cost_usd: float = 0.0

    def as_dict(self) -> dict:
        return {"voice": self.voice, "images": self.images, "errors": self.errors,
                "est_cost_usd": round(self.est_cost_usd, 4)}


class Enricher:
    def __init__(self, store: Store, cache: Cache, media_root: pathlib.Path | None, offline: bool = False,
                 api_factory: Callable[[], Api] | None = None, workers: int = 8,
                 log: Callable[[str], None] | None = None):
        self.store, self.cache, self.media_root, self.offline = store, cache, media_root, offline
        self._api_factory = api_factory or default_api
        self._api: Api | None = None
        self.workers = workers
        self.summary = Summary()
        self.log = log or (lambda msg: None)
        self._lock = threading.Lock()

    @property
    def api(self) -> Api:
        with self._lock:
            if self._api is None:
                self._api = self._api_factory()
            return self._api

    def _file(self, media: dict) -> pathlib.Path | None:
        if self.media_root is None:
            return None
        p = self.media_root / media["file_path"]
        return p if p.is_file() else None

    # -- voice ----------------------------------------------------------------------------------
    def voice(self) -> None:
        media = self.store.query("SELECT * FROM media WHERE kind IN ('audio', 'video') ORDER BY artifact_id")
        s = self.summary.voice
        rows = []
        for m in media:
            key = voice_key(m["sha256"])
            result = self.cache.get(key)
            path = self._file(m) if result is None else None
            if result is not None:
                s["hits"] += 1
            elif path is not None and m["kind"] == "video" and not has_audio_track(path):
                s["no_audio"] += 1  # a silent video is not a miss, offline or not
                continue
            elif self.offline:
                s["misses"] += 1
                continue
            else:
                if path is None:
                    s["misses"] += 1
                    self.summary.errors.append(f"{m['artifact_id']}: media file not found under the media root")
                    continue
                try:
                    wav = to_wav(path)
                except EnrichError as e:
                    self.summary.errors.append(f"{m['artifact_id']} voice: {e}")
                    continue
                try:
                    result = self.api.transcribe(wav)
                except ProviderError as e:
                    self.summary.errors.append(f"{m['artifact_id']} voice: {e}")
                    continue
                s["calls"] += 1
                self.summary.est_cost_usd += -(-int(result.get("audioDurationMs") or 0) // 1000) * PRICE_VOICE_PER_SECOND
                self.cache.put(key, result)
                self.log(f"voice {m['artifact_id']}: {len(result['transcript'])} chars")
            s["media"] += 1
            row = transcript_row(m, result, key)
            if row is not None:
                rows.append(row)
                s["audio_s"] = round(s["audio_s"] + row["duration_s"], 3)
        self.store.replace_rows("transcripts", "model = ?", [VOICE_MODEL], rows)
        s["rows"] = len(rows)

    # -- images ---------------------------------------------------------------------------------
    def images(self, prompt_set: str = PROMPT_SET) -> None:
        labels = PROMPT_SETS[prompt_set]
        media = self.store.query(f"SELECT * FROM media WHERE kind = 'image' AND mime_type IN "
                                 f"({', '.join('?' * len(SAM_MIMES))}) ORDER BY artifact_id", SAM_MIMES)
        s = self.summary.images
        s["media"] = len(media)
        todo: list[tuple[dict, str]] = []
        for m in media:
            for label in labels:
                s["requests"] += 1
                if self.cache.get(sam_key(m["sha256"], label, prompt_set)) is not None:
                    s["hits"] += 1
                elif self.offline:
                    s["misses"] += 1
                else:
                    todo.append((m, label))
        if todo:
            self._segment_all(todo, prompt_set)
        rows = []
        for m in media:
            for label in labels:
                key = sam_key(m["sha256"], label, prompt_set)
                result = self.cache.get(key)
                if result is not None:
                    rows += tag_rows(m, label, result, key, prompt_set)
        self.store.replace_rows("image_tags", "model = ? AND prompt_set = ?", [SAM_MODEL, prompt_set], rows)
        s["rows"] = len(rows)

    def _segment_all(self, todo: list[tuple[dict, str]], prompt_set: str) -> None:
        prepared: dict[str, tuple[bytes, str, int, int] | None] = {}

        def prep(m: dict):
            if m["artifact_id"] not in prepared:
                path = self._file(m)
                prepared[m["artifact_id"]] = oriented_image(path, m["mime_type"]) if path else None
            return prepared[m["artifact_id"]]

        def one(item: tuple[dict, str]) -> None:
            m, label = item
            with self._lock:
                img = prep(m)
            if img is None:
                with self._lock:
                    self.summary.images["misses"] += 1
                    self.summary.errors.append(f"{m['artifact_id']}: media file not found under the media root")
                return
            data, mime, w, h = img
            try:
                result = self.api.segment(data, mime, label)
            except ProviderError as e:
                with self._lock:
                    self.summary.errors.append(f"{m['artifact_id']} sam {label!r}: {e}")
                return
            result.update(width=w, height=h)
            self.cache.put(sam_key(m["sha256"], label, prompt_set), result)
            with self._lock:
                self.summary.images["calls"] += 1
                self.summary.est_cost_usd += PRICE_SAM_PER_REQUEST
            self.log(f"sam {m['artifact_id']} {label!r}: {len(result['boxes'])} boxes")

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(one, todo))


# ------------------------------------------------------------------------------ derived rows
def _derived(media: dict, table: str, row_key: str, raw: dict) -> dict:
    return {"artifact_id": artifact_id(table, media["source_path"], row_key), "source_path": media["source_path"],
            "source_row": row_key, "ts_utc": media["ts_utc"], "ts_original": media["ts_original"],
            "tz_basis": media["tz_basis"], "raw_json": json.dumps(raw, ensure_ascii=False, sort_keys=True)}


def transcript_row(media: dict, result: dict, key: str) -> dict | None:
    text = (result.get("transcript") or "").strip()
    if not text:
        return None  # media with an empty transcript get no row (CONTRACTS section 9)
    ms = result.get("audioDurationMs")
    raw = {"sessionId": result.get("sessionId"), "turns": result.get("turns") or [], "audioDurationMs": ms,
           "transcript": result.get("transcript"), "media_id": media["artifact_id"], "cache_key": key,
           "request": VOICE_REQUEST, "transcode": "ffmpeg " + " ".join(FFMPEG_ARGS)}
    row = _derived(media, "transcripts", f"{media['artifact_id']}|{VOICE_MODEL}", raw)
    row.update(media_id=media["artifact_id"], model=VOICE_MODEL, language=None, text=text,
               duration_s=(float(ms) / 1000.0) if ms is not None else None)
    return row


def tag_rows(media: dict, label: str, result: dict, key: str, prompt_set: str = PROMPT_SET) -> list[dict]:
    rows = []
    for n, (box, score, raw_box) in enumerate(kept_boxes(result)):
        raw = {"label": label, "n": n, "box": box, "score": score, "sam_box": raw_box,
               "image_size": [int(result["width"]), int(result["height"])], "media_id": media["artifact_id"],
               "cache_key": key, "response_id": result.get("response_id"), "score_floor": SCORE_FLOOR}
        row = _derived(media, "image_tags", f"{media['artifact_id']}|{SAM_MODEL}|{label}|{n}", raw)
        row.update(media_id=media["artifact_id"], model=SAM_MODEL, label=label, score=score, box=json.dumps(box),
                   prompt_set=prompt_set)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------ entry point
def default_api() -> Api:
    base = os.environ.get("META_BASE_URL") or "https://api.meta.ai/v1"
    key = os.environ.get("META_API_KEY")
    if not key:
        raise EnrichError("META_API_KEY is not set (use --offline to build from the committed cache only)")
    return Api(base, key)


def resolve_cache(store: Store, cache: str | None, media_root: pathlib.Path | None) -> pathlib.Path:
    """--cache, then TBX_ENRICH_CACHE, then <media root>/../derived/, then data/sample/derived/ (cwd, then repo)."""
    if cache:
        return pathlib.Path(cache)
    if os.environ.get("TBX_ENRICH_CACHE"):
        return pathlib.Path(os.environ["TBX_ENRICH_CACHE"])
    if media_root is not None:
        return media_root.parent / "derived" / DEFAULT_CACHE.name
    repo = pathlib.Path(__file__).resolve().parents[2]
    for base in (pathlib.Path.cwd(), repo):
        if (base / DEFAULT_CACHE).is_file():
            return base / DEFAULT_CACHE
    return DEFAULT_CACHE


def resolve_media_root(store: Store, media_root: str | None) -> pathlib.Path | None:
    root = media_root or store.meta().get("media_root")
    if not root:
        return None
    p = pathlib.Path(root)
    return p if p.is_dir() else None


def enrich(db_path: str, voice: bool = True, images: bool = True, offline: bool = False,
           media_root: str | None = None, cache_path: str | None = None, workers: int = 8,
           api_factory: Callable[[], Api] | None = None, log: Callable[[str], None] | None = None) -> dict:
    if not pathlib.Path(db_path).is_file():
        raise EnrichError(f"store not found: {db_path} (run `tbx ingest` first, or pass --db)")
    store = Store(db_path)
    try:
        root = resolve_media_root(store, media_root)
        if media_root and root is None:
            raise EnrichError(f"media root not found: {media_root}")
        cache = Cache(resolve_cache(store, cache_path, root))
        e = Enricher(store, cache, root, offline=offline, api_factory=api_factory, workers=workers, log=log)
        if voice:
            e.voice()
        if images:
            e.images()
        cache.compact()
        out = e.summary.as_dict()
        out.update(cache=str(cache.path), cache_added=cache.added, offline=offline,
                   media_root=str(root) if root else None)
        return out
    finally:
        store.close()
