"""`tbx export-static`: a replay-only copy of the web app as plain files (GitHub Pages, any static host).

The export drives the real tbx.web app in-process and stores each response body unchanged, so every JSON file is what
the server would answer (api/case.json alone is edited, to turn live asking off):

    index.html                         the page, with a manifest in <script id="tbx-static"> (static mode)
    api/case.json                      GET /api/case, with live asking turned off
    api/runs.json                      GET /api/runs
    api/runs/<run_id>.json             GET /api/runs/<run_id>, for every recorded run
    api/artifacts/<id>.json            GET /api/artifacts/<id>, for every cited artifact and the media behind it
    api/media/<id>/tags.json           GET /api/media/<id>/tags
    api/media/<id>/transcripts.json    GET /api/media/<id>/transcripts
    media/<id>.<ext>                   the cited media files (extension from the MIME type, so hosts serve the right type)

No server, no keys and no model calls: the page reads these files and disables the ask form. Paths are relative, so
the site also works under a project-pages prefix such as https://<user>.github.io/<repo>/.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
from typing import Any

MARKER = ".tbx-static"          # marks a directory this command wrote, so it may be replaced on the next export
PAGE_HOOK = "<!--tbx-static-->"
EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "video/mp4": ".mp4",
       "video/3gpp": ".3gp", "video/quicktime": ".mov", "video/webm": ".webm", "audio/wav": ".wav",
       "audio/mpeg": ".mp3", "audio/mp4": ".m4a", "audio/ogg": ".ogg", "audio/amr": ".amr"}
SECRET_VARS = ("META_API_KEY", "BASETEN_API_KEY", "MODAL_API_KEY", "MODAL_TOKEN_SECRET")


class ExportError(Exception):
    pass


def _ext(mime: str | None) -> str:
    return EXT.get((mime or "").split(";")[0].strip().lower(), ".bin")


def _media_ids_behind(row: dict) -> list[str]:
    """The media rows the drawer opens for this artifact (the image under a tag, the video under a transcript...)."""
    t = row.get("table")
    if t == "media":
        return [row["artifact_id"]]
    if t in ("image_tags", "transcripts") and row.get("media_id"):
        return [row["media_id"]]
    if t == "messages" and row.get("attachment_media_id"):
        return [row["attachment_media_id"]]
    return []


def _cited_ids(view: dict) -> set[str]:
    claims = list(view.get("verified", [])) + [r.get("claim", {}) for r in view.get("rejected", [])]
    first = view.get("first_round") or {}
    claims += list(first.get("verified", [])) + [r.get("claim", {}) for r in first.get("rejected", [])]
    return {c["artifact_id"] for cl in claims for c in cl.get("citations", []) if c.get("artifact_id")}


def _prepare_out(out: pathlib.Path) -> None:
    if out.exists():
        if not out.is_dir():
            raise ExportError(f"{out} exists and is not a directory")
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise ExportError(f"{out} is not empty and was not written by `tbx export-static`; pick another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def export_static(out: str, db_path: str, runs_dir: str, media_root: str | None = None,
                  featured: str | None = None) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from tbx import web

    if not pathlib.Path(db_path).is_file():
        raise ExportError(f"store not found: {db_path} (run `tbx ingest` and `tbx enrich --offline` first)")
    runs_path = pathlib.Path(runs_dir)
    if not runs_path.is_dir() or not any(runs_path.glob("*.jsonl")):
        raise ExportError(f"no recorded runs in {runs_dir}")

    def no_client(provider, model):  # the export never asks
        raise ExportError("export-static makes no model calls")

    client = TestClient(web.create_app(db_path, runs_dir=str(runs_path), media_root=media_root, client_factory=no_client))

    def get(path: str, ok404: bool = False):
        r = client.get(path)
        if r.status_code == 404 and ok404:
            return None
        if r.status_code != 200:
            raise ExportError(f"GET {path} returned HTTP {r.status_code}: {r.text[:200]}")
        return r

    root = pathlib.Path(out)
    _prepare_out(root)
    written: list[pathlib.Path] = []

    def write(rel: str, data: bytes) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written.append(p)

    def save(rel: str, path: str, ok404: bool = False):
        """GET path and store the response body unchanged at rel; returns the parsed JSON (None on an allowed 404)."""
        resp = get(path, ok404=ok404)
        if resp is None:
            return None
        write(rel, resp.content)
        return resp.json()

    # runs
    runs = save("api/runs.json", "/api/runs")
    cited: set[str] = set()
    for r in runs:
        cited |= _cited_ids(save(f"api/runs/{r['run_id']}.json", f"/api/runs/{r['run_id']}"))

    # artifacts: every cited row that exists (an unknown ID stays a 404, as on the server), plus the media behind it
    rows: dict[str, dict] = {}
    missing: list[str] = []
    for aid in sorted(cited):
        row = save(f"api/artifacts/{aid}.json", f"/api/artifacts/{aid}", ok404=True)
        if row is None:
            missing.append(aid)
        else:
            rows[aid] = row
    media_ids = sorted({m for row in rows.values() for m in _media_ids_behind(row)})
    for mid in media_ids:
        if mid not in rows:
            row = save(f"api/artifacts/{mid}.json", f"/api/artifacts/{mid}", ok404=True)
            if row is not None:
                rows[mid] = row

    # media: tags, transcripts and the file itself
    media_files: dict[str, str] = {}
    for mid in media_ids:
        if mid not in rows:
            continue
        save(f"api/media/{mid}/tags.json", f"/api/media/{mid}/tags")
        save(f"api/media/{mid}/transcripts.json", f"/api/media/{mid}/transcripts")
        resp = get(f"/api/media/{mid}", ok404=True)
        if resp is None:
            continue
        rel = f"media/{mid}{_ext(rows[mid].get('mime_type') or resp.headers.get('content-type'))}"
        write(rel, resp.content)
        media_files[mid] = rel

    # case header: same counts and metadata, live asking off
    run_ids = {r["run_id"] for r in runs}
    case = get("/api/case").json()
    want = featured or case.get("featured_run")
    case.update(providers={p: False for p in case.get("providers", {})}, live_providers=[], ask_limit_per_hour=0,
                asks_remaining=0, static=True, featured_run=want if want in run_ids else None)
    write("api/case.json", json.dumps(case, ensure_ascii=False, separators=(",", ":")).encode())

    # the page, in static mode
    manifest = {"generator": "tbx export-static", "runs": len(runs), "media": media_files}
    page = (pathlib.Path(web.__file__).parent / "index.html").read_text(encoding="utf-8")
    if PAGE_HOOK not in page:
        raise ExportError("index.html has no static-mode hook")
    tag = ('<script id="tbx-static" type="application/json">'
           + json.dumps(manifest, sort_keys=True).replace("</", "<\\/") + "</script>")
    (root / "index.html").write_text(page.replace(PAGE_HOOK, tag, 1), encoding="utf-8")
    written.append(root / "index.html")
    (root / ".nojekyll").write_text("")                     # GitHub Pages: serve files as they are
    (root / MARKER).write_text("written by `tbx export-static`; safe to delete and rebuild\n")

    try:
        _assert_no_secrets(written)
    except ExportError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return {"out": str(root), "runs": len(runs), "artifacts": len(rows), "media_files": len(media_files),
            "unknown_cited_ids": missing, "featured_run": case["featured_run"],
            "bytes": sum(p.stat().st_size for p in written)}


def _assert_no_secrets(files: list[pathlib.Path]) -> None:
    """Belt and braces: no key value from this environment (nor its first or last 16 characters) may be published."""
    names = sorted(k for k in os.environ if k.endswith("_API_KEY") or k in SECRET_VARS)
    needles = {n.encode() for k in names if len(v := os.environ.get(k) or "") >= 12 for n in (v, v[:16], v[-16:])}
    if not needles:
        return
    for p in files:
        data = p.read_bytes()
        if any(n in data for n in needles):
            raise ExportError(f"a key value from the environment appeared in {p.name}; export aborted and removed")
