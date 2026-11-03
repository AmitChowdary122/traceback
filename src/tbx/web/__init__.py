"""Traceback web app (CONTRACTS.md v1.2 section 10). Run it with `uvicorn tbx.web:app`.

- POST /api/ask returns {run_id} at once and runs the same agent code as `tbx ask` in a background thread.
- GET /api/runs and /api/runs/{id} read runs/*.jsonl, so recorded runs replay with no model call. Each recorded
  answer is re-checked on load with the deterministic checker (tbx.runview): the view carries the re-check's verdict,
  a warning when it differs from the recorded check_result, the v1.3 entity spans of every claim it shows, one
  plain-English sentence per rejection reason (tbx.explain) and, for repaired runs, round 1 next to round 2.
- Each run view also carries its coverage (tbx.coverage): what each search looked at and which evidence families were
  never searched, from the trace's tool calls.
- The page (index.html, next to this file) needs no build step.

Environment: TBX_DB (store), TBX_RUNS_DIR (default ./runs), TBX_MEDIA_ROOT (default: the media root
recorded by `tbx ingest`), TBX_ASK_LIMIT_PER_HOUR (default 30; 0 turns asking off), TBX_FEATURED_RUN (the
recorded run the page replays by default), TBX_LIVE_PROVIDERS (comma-separated providers that may be asked live,
default "meta"; any other provider gets HTTP 403 and the page offers its recorded runs as replays; set but empty
means no live asks), plus the provider keys used by `tbx ask`. Keys are never returned, logged or traced.
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import pathlib
import re
import threading
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from tbx import coverage as cov
from tbx import runview
from tbx import timeutil as tu
from tbx.store import Store

HERE = pathlib.Path(__file__).resolve().parent
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
STALE_S = 15 * 60          # a trace with no run_end, untouched this long and not running here, is an error
META_KEYS = ("device_tz", "device_model", "aleapp_version", "report_root", "ingested_at")
OPEN_START, OPEN_END = "1970-01-01T00:00:00Z", "2100-01-01T00:00:00Z"
DEFAULT_LIVE_PROVIDERS = "meta"   # the hosted demo pays for Meta asks only; Baseten runs are recorded replays
PROVIDER_KEYS = (("meta", "META_API_KEY"), ("baseten", "BASETEN_API_KEY"))


def live_providers() -> list[str]:
    """TBX_LIVE_PROVIDERS as a list: unset means "meta"; set but empty means none."""
    raw = os.environ.get("TBX_LIVE_PROVIDERS")
    raw = DEFAULT_LIVE_PROVIDERS if raw is None else raw
    out: list[str] = []
    for p in (x.strip().lower() for x in raw.split(",")):
        if p and p not in out:
            out.append(p)
    return out


class AskBody(BaseModel):
    question: str
    from_: str | None = None
    to: str | None = None
    provider: str | None = None
    model: str | None = None
    repair: int = 0

    model_config = {"populate_by_name": True}

    @classmethod
    def parse(cls, data: dict) -> "AskBody":
        data = dict(data or {})
        if "from" in data:
            data["from_"] = data.pop("from")
        return cls.model_validate(data)


class RateLimit:
    """A global sliding-window limit on asks (hosted demo cost cap)."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.times: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()

    def take(self) -> tuple[bool, int]:
        now = time.monotonic()
        with self.lock:
            while self.times and now - self.times[0] > 3600:
                self.times.popleft()
            if self.per_hour <= 0 or len(self.times) >= self.per_hour:
                retry = int(3600 - (now - self.times[0])) + 1 if self.times else 3600
                return False, retry
            self.times.append(now)
            return True, 0

    def remaining(self) -> int:
        now = time.monotonic()
        with self.lock:
            return max(0, self.per_hour - sum(1 for t in self.times if now - t <= 3600))


def _read_events(path: pathlib.Path) -> list[dict]:
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # a line being written right now
    return out


def summarize(events: list[dict], active: bool, mtime: float | None = None) -> dict:
    """Trace events -> the run view: status, the last check, usage, the first round when repaired."""
    start = next((e["payload"] for e in events if e.get("type") == "run_start"), {})
    end = next((e["payload"] for e in reversed(events) if e.get("type") == "run_end"), None)
    checks = [e["payload"] for e in events if e.get("type") == "check_result"]
    answers = [e["payload"] for e in events if e.get("type") == "final_answer"]
    if end is not None:
        status = end.get("status", "ok")
    elif active or (mtime is not None and time.time() - mtime < STALE_S):
        status = "running"
    else:
        status = "error"
    last = checks[-1] if checks else {}
    view = {"status": status, "question": start.get("question"), "window": start.get("window"),
            "provider": start.get("provider"), "model": start.get("model"), "repair": start.get("repair", 0),
            "case_tz": start.get("case_tz"), "started": events[0]["ts"] if events else None,
            "verified": last.get("verified", []), "rejected": last.get("rejected", []),
            "note": answers[-1].get("note") if answers else None,
            "usage": (end or {}).get("totals", {}), "error": (end or {}).get("error")}
    if len(checks) > 1:
        view["first_round"] = {"verified": checks[0].get("verified", []), "rejected": checks[0].get("rejected", [])}
    if end is None and status == "error":
        view["error"] = "the run stopped without a run_end event"
    return view


def create_app(db_path: str | None = None, runs_dir: str | None = None, media_root: str | None = None,
               client_factory: Callable[[str | None, str | None], Any] | None = None) -> FastAPI:
    """`client_factory(provider, model)` builds the chat client; the default is tbx.llm.LLMClient (tests inject fakes)."""
    db = str(db_path or os.environ.get("TBX_DB") or "data/tbx.db")
    runs = pathlib.Path(runs_dir or os.environ.get("TBX_RUNS_DIR") or "runs").resolve()
    limiter = RateLimit(int(os.environ.get("TBX_ASK_LIMIT_PER_HOUR", "30") or 0))
    state: dict[str, Any] = {"store": None}
    store_lock = threading.Lock()
    active: dict[str, dict] = {}       # run_id -> {"status", "error"} for runs started by this process
    active_lock = threading.Lock()
    rechecked: dict[str, tuple[tuple, dict]] = {}   # run_id -> ((mtime_ns, size), runview.reverify(...))

    app = FastAPI(title="Traceback", docs_url=None, redoc_url=None)

    def store() -> Store:
        if state["store"] is None:
            if not pathlib.Path(db).is_file():
                raise HTTPException(503, f"store not found: {pathlib.Path(db).name} (run `tbx ingest` first)")
            state["store"] = Store(db)
        return state["store"]

    def q(sql: str, params=()) -> list[dict]:
        with store_lock:
            return store().query(sql, params)

    def get_row(artifact_id: str) -> dict | None:
        with store_lock:
            return store().get(artifact_id)

    def case_tz() -> str:
        with store_lock:
            return store().case_tz

    def files_root() -> pathlib.Path | None:
        root = media_root or os.environ.get("TBX_MEDIA_ROOT")
        if not root:
            with store_lock:
                root = store().meta().get("media_root")
        return pathlib.Path(root).resolve() if root else None

    def run_path(run_id: str) -> pathlib.Path:
        if not RUN_ID.fullmatch(run_id or ""):
            raise HTTPException(404, "unknown run")
        return runs / f"{run_id}.jsonl"

    def reverify(p: pathlib.Path, events: list[dict]) -> dict:
        """Re-check a recorded run against the store (cached until its trace file changes)."""
        st = p.stat()
        key = (st.st_mtime_ns, st.st_size, len(events))
        hit = rechecked.get(p.stem)
        if hit and hit[0] == key:
            return hit[1]
        with store_lock:
            out = runview.reverify(events, store())
        rechecked[p.stem] = (key, out)
        return out

    def artifacts_for(view: dict, tz: str) -> dict:
        claims = list(view.get("verified", [])) + [r.get("claim", {}) for r in view.get("rejected", [])]
        if view.get("first_round"):
            claims += view["first_round"].get("verified", [])
            claims += [r.get("claim", {}) for r in view["first_round"].get("rejected", [])]
        ids = sorted({c.get("artifact_id") for cl in claims for c in cl.get("citations", []) if c.get("artifact_id")})
        out = {}
        for i in ids:
            row = get_row(i)
            if row is None:
                continue
            brief = {"table": row["table"], "ts_utc": row.get("ts_utc"), "ts_local": tu.local_iso(row.get("ts_utc"), tz),
                     "source_path": row["source_path"], "source_row": row["source_row"]}
            for k in ("kind", "mime_type", "media_id", "label", "app", "direction"):
                if row.get(k) is not None:
                    brief[k] = row[k]
            out[i] = brief
        return out

    # -- pages ----------------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((HERE / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/case")
    def case() -> dict:
        with store_lock:
            s = store()
            meta = s.meta()
            counts = {t: s.count(t) for t in ("messages", "calls", "locations", "media", "transcripts", "image_tags")}
        live = live_providers()
        # providers[p]: this server can ask p live right now (p is in TBX_LIVE_PROVIDERS and its key is set).
        providers = {p: p in live and bool(os.environ.get(k)) for p, k in PROVIDER_KEYS}
        featured = os.environ.get("TBX_FEATURED_RUN") or None
        if featured and not (RUN_ID.fullmatch(featured) and (runs / f"{featured}.jsonl").is_file()):
            featured = None
        return {"meta": {k: meta.get(k) for k in META_KEYS}, "counts": counts, "providers": providers,
                "live_providers": live, "ask_limit_per_hour": limiter.per_hour,
                "asks_remaining": limiter.remaining(), "featured_run": featured, "static": False}

    # -- asking ---------------------------------------------------------------------------------
    @app.post("/api/ask")
    def ask(body: dict) -> JSONResponse:
        from tbx.agent import new_run_id, run_agent
        from tbx.llm import PROVIDERS, ConfigError, LLMClient

        try:
            a = AskBody.parse(body)
        except ValueError as e:
            raise HTTPException(400, f"bad request: {str(e)[:300]}") from None
        if not a.question.strip():
            raise HTTPException(400, "question is empty")
        if a.repair not in (0, 1):
            raise HTTPException(400, "repair must be 0 or 1")
        provider = (a.provider or os.environ.get("TBX_PROVIDER") or "meta").strip().lower()
        if provider not in PROVIDERS:
            raise HTTPException(400, f"unknown provider {provider!r} (use one of {', '.join(PROVIDERS)})")
        live = live_providers()
        if provider not in live:  # checked before the rate limit, so a refused ask costs nothing
            allowed = f"go to {' and '.join(live)} only" if live else "are turned off"
            raise HTTPException(403, f"Live asks on this server {allowed}. Recorded {provider} runs are listed "
                                     f"under the {provider} toggle: replay one instead (no model call).")
        tz = case_tz()
        window = None
        if a.from_ or a.to:
            try:
                s = tu.parse_user_time(a.from_, tz) if a.from_ else tu.parse_utc(OPEN_START)
                e = tu.parse_user_time(a.to, tz) if a.to else tu.parse_utc(OPEN_END)
                window = tu.Window(s, e)
            except ValueError as ex:
                raise HTTPException(400, f"bad from/to: {ex}") from None
        try:
            client = (client_factory or LLMClient)(provider, a.model)
        except ConfigError as e:
            raise HTTPException(400, str(e)) from None
        ok, retry = limiter.take()
        if not ok:
            return JSONResponse({"detail": f"ask limit reached ({limiter.per_hour} per hour); replay a recorded run",
                                 "retry_after_s": retry}, status_code=429, headers={"Retry-After": str(retry)})
        run_id = new_run_id()
        with active_lock:
            active[run_id] = {"status": "running", "error": None}

        def work() -> None:
            st = Store(db)  # each run gets its own connection
            try:
                run_agent(a.question.strip(), st, window, client, repair=a.repair, trace_dir=runs, run_id=run_id)
                status, err = "ok", None
            except Exception as ex:  # noqa: BLE001 - the trace already has run_end with the error
                status, err = "error", f"{type(ex).__name__}: {ex}"[:500]
            finally:
                st.close()
            with active_lock:
                active[run_id] = {"status": status, "error": err}

        threading.Thread(target=work, name=f"tbx-run-{run_id}", daemon=True).start()
        return JSONResponse({"run_id": run_id})

    # -- runs -----------------------------------------------------------------------------------
    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        out = []
        for p in runs.glob("*.jsonl") if runs.is_dir() else []:
            run_id = p.stem
            events = _read_events(p)
            with active_lock:
                is_active = active.get(run_id, {}).get("status") == "running"
            v = summarize(events, is_active, p.stat().st_mtime)
            try:
                recheck = reverify(p, events)["recheck"]["status"] if v["status"] != "running" else "none"
            except HTTPException:  # no store: the list still works, the run view reports the store error
                recheck = "unavailable"
            out.append({"run_id": run_id, "question": v["question"], "status": v["status"], "started": v["started"],
                        "provider": v["provider"], "model": v["model"], "verified": len(v["verified"]),
                        "rejected": len(v["rejected"]), "recheck": recheck})
        with active_lock:
            pending = [(k, v) for k, v in active.items() if not (runs / f"{k}.jsonl").exists()]
        for run_id, v in pending:
            out.append({"run_id": run_id, "question": None, "status": v["status"], "started": None,
                        "provider": None, "model": None, "verified": 0, "rejected": 0, "recheck": "none"})
        out.sort(key=lambda r: (r["started"] or "9999", r["run_id"]), reverse=True)
        return out

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        p = run_path(run_id)
        with active_lock:
            mine = dict(active[run_id]) if run_id in active else None
        if not p.is_file():
            if mine is None:
                raise HTTPException(404, "unknown run")
            return {"run_id": run_id, "status": mine["status"], "verified": [], "rejected": [], "usage": {},
                    "events": [], "error": mine["error"], "artifacts": {}}
        events = _read_events(p)
        view = summarize(events, bool(mine and mine["status"] == "running"), p.stat().st_mtime)
        if mine and mine["status"] == "error" and view["status"] != "running" and not view.get("error"):
            view["error"] = mine["error"]
        tz = view.get("case_tz") or case_tz()
        view.update(run_id=run_id, events=events, artifacts=artifacts_for(view, tz), case_tz=tz,
                    coverage=cov.coverage(events, tz))
        if view["status"] == "running":
            view.update(recheck={"status": "none", "message": "", "rounds": [], "entities_recorded": False},
                        annotations={"verified": [], "rejected": []}, repair=None)
        else:
            rv = reverify(p, events)
            view.update(recheck=rv["recheck"], annotations=rv["annotations"], repair=rv["repair"])
            if view.get("first_round") and rv["first_round_annotations"]:
                view["first_round"]["annotations"] = rv["first_round_annotations"]
        return view

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        p = run_path(run_id)
        with active_lock:
            known = run_id in active
        if not p.is_file() and not known:
            raise HTTPException(404, "unknown run")

        async def stream():
            sent, idle = 0, 0.0
            while idle < STALE_S:
                events = _read_events(p) if p.is_file() else []
                for e in events[sent:]:
                    yield f"event: {e.get('type', 'message')}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                    if e.get("type") == "run_end":
                        return
                idle = 0.0 if len(events) > sent else idle + 0.5
                sent = len(events)
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # -- artifacts and media --------------------------------------------------------------------
    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict:
        row = get_row(artifact_id)
        if row is None:
            raise HTTPException(404, "unknown artifact")
        row["ts_local"] = tu.local_iso(row.get("ts_utc"), case_tz())
        if row["table"] == "image_tags" and isinstance(row.get("box"), str):
            row["box_array"] = json.loads(row["box"])
        return row

    def media_row(artifact_id: str) -> dict:
        row = get_row(artifact_id)
        if row is None or row["table"] != "media":
            raise HTTPException(404, "unknown media")
        return row

    @app.get("/api/media/{artifact_id}")
    def media(artifact_id: str) -> FileResponse:
        row = media_row(artifact_id)
        root = files_root()
        if root is None:
            raise HTTPException(404, "no media root for this store")
        path = (root / row["file_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, "media file not found")
        return FileResponse(path, media_type=row.get("mime_type") or "application/octet-stream")

    @app.get("/api/media/{artifact_id}/tags")
    def media_tags(artifact_id: str) -> dict:
        media_row(artifact_id)
        rows = q("SELECT * FROM image_tags WHERE media_id = ? ORDER BY label, score DESC, artifact_id", [artifact_id])
        for r in rows:
            r["box"] = json.loads(r["box"]) if isinstance(r.get("box"), str) else r.get("box")
        return {"rows": rows}

    @app.get("/api/media/{artifact_id}/transcripts")
    def media_transcripts(artifact_id: str) -> dict:
        media_row(artifact_id)
        return {"rows": q("SELECT * FROM transcripts WHERE media_id = ? ORDER BY artifact_id", [artifact_id])}

    return app


app = create_app()
