"""`tbx` command line (CONTRACTS.md section 9).

Exit codes: 0 ok, 1 usage or input error, 2 agent output failed validation twice, 3 provider/network error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

from pydantic import ValidationError

EXIT_OK, EXIT_USAGE, EXIT_AGENT, EXIT_PROVIDER = 0, 1, 2, 3
OPEN_START = datetime(1970, 1, 1, tzinfo=timezone.utc)
OPEN_END = datetime(2100, 1, 1, tzinfo=timezone.utc)


class UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # argparse would exit 2, which the contract reserves for the agent
        raise UsageError(f"{self.prog}: {message}")


def _parser() -> _Parser:
    p = _Parser(prog="tbx", description="Traceback: verified answers about a phone extraction.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="load an ALEAPP report into the store")
    ing.add_argument("report_dir")
    ing.add_argument("--media-root", default=None)
    ing.add_argument("--db", default=None)

    enr = sub.add_parser("enrich", help="voice transcripts and image tags (cached); both when neither flag is given")
    enr.add_argument("--voice", action="store_true", help="Muse Voice Transcribe on audio and video soundtracks")
    enr.add_argument("--images", action="store_true", help="SAM 3.1 prompt set v1 on JPEG/PNG photos")
    enr.add_argument("--offline", action="store_true", help="use only the cache; misses are skipped and counted")
    enr.add_argument("--db", default=None)
    enr.add_argument("--media-root", default=None, help="default: the media root recorded by `tbx ingest`")
    enr.add_argument("--cache", default=None, help="default: <media root>/../derived/enrich_cache.jsonl")
    enr.add_argument("--workers", type=int, default=8)
    enr.add_argument("-v", "--verbose", action="store_true")

    ask = sub.add_parser("ask", help="ask a question; claims are verified by the checker")
    ask.add_argument("question")
    ask.add_argument("--from", dest="from_", default=None, metavar="T")
    ask.add_argument("--to", default=None, metavar="T")
    ask.add_argument("--provider", default=None)
    ask.add_argument("--model", default=None)
    ask.add_argument("--repair", type=int, choices=[0, 1], default=0)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--db", default=None)

    chk = sub.add_parser("check", help="run the checker on a saved Answer JSON")
    chk.add_argument("answer_json")
    chk.add_argument("--from", dest="from_", default=None, metavar="T")
    chk.add_argument("--to", default=None, metavar="T")
    chk.add_argument("--db", default=None)

    exp = sub.add_parser("export-static", help="write a replay-only copy of the web app as static files (GitHub Pages)")
    exp.add_argument("--out", default="site", help="output directory (default: site)")
    exp.add_argument("--db", default=None)
    exp.add_argument("--runs-dir", default="deploy/runs", help="recorded runs to include (default: deploy/runs)")
    exp.add_argument("--media-root", default=None, help="default: the media root recorded by `tbx ingest`")
    exp.add_argument("--featured", default=None, help="run the page opens by default (default: TBX_FEATURED_RUN)")
    return p


def _db_path(arg: str | None) -> str:
    return arg or os.environ.get("TBX_DB") or "data/tbx.db"


def _open_store(arg: str | None):
    from tbx.store import Store

    path = _db_path(arg)
    if not pathlib.Path(path).is_file():
        raise UsageError(f"store not found: {path} (run `tbx ingest` first, or pass --db)")
    return Store(path)


def _window(start: str | None, end: str | None, case_tz: str):
    from tbx.timeutil import Window, parse_user_time

    if start is None and end is None:
        return None
    try:
        s = parse_user_time(start, case_tz) if start else OPEN_START
        e = parse_user_time(end, case_tz) if end else OPEN_END
        return Window(s, e)
    except ValueError as ex:
        raise UsageError(f"bad --from/--to: {ex}") from None


def _err(msg: str) -> None:
    print(f"tbx: {msg}", file=sys.stderr)


def cmd_ask(a) -> int:
    from tbx.agent import AgentError, run_agent
    from tbx.llm import ConfigError, LLMClient, ProviderError

    store = _open_store(a.db)
    window = _window(a.from_, a.to, store.case_tz)
    try:
        client = LLMClient(a.provider, a.model)
    except ConfigError as e:
        raise UsageError(str(e)) from None
    try:
        res = run_agent(a.question, store, window, client, repair=a.repair)
    except AgentError as e:
        _err(str(e))
        return EXIT_AGENT
    except ProviderError as e:
        _err(str(e))
        return EXIT_PROVIDER
    out = {"run_id": res.run_id, "verified": [c.model_dump() for c in res.check.verified],
           "rejected": [r.model_dump() for r in res.check.rejected], "usage": res.usage, "note": res.answer.note,
           "provider": res.provider, "model": res.model, "window": window.as_dict() if window else None,
           "case_tz": store.case_tz}
    if res.first_check is not None:
        out["first_round"] = {"verified": [c.model_dump() for c in res.first_check.verified],
                              "rejected": [r.model_dump() for r in res.first_check.rejected]}
    if a.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        _print_human(out, store)
    return EXIT_OK


def _print_human(out: dict, store) -> None:
    from tbx.timeutil import local_iso

    print(f"run {out['run_id']}  ({out['provider']}/{out['model']}, case tz {out['case_tz']})")
    print(f"\nVERIFIED ({len(out['verified'])})")
    for c in out["verified"]:
        print(f"  ✓ {c['text']}")
        for cit in c["citations"]:
            row = store.get(cit["artifact_id"]) or {}
            print(f"      {cit['artifact_id']}.{cit['field']} = {cit['value']!r}   "
                  f"[{row.get('source_path')} row {row.get('source_row')}, {local_iso(row.get('ts_utc'), out['case_tz'])}]")
    print(f"\nREJECTED ({len(out['rejected'])})")
    for r in out["rejected"]:
        print(f"  ✗ {r['claim']['text']}")
        for reason in r["reasons"]:
            idx = "" if reason["citation_index"] is None else f" [citation {reason['citation_index']}]"
            print(f"      {reason['code']}{idx}: {reason['detail']}")
    if out.get("note"):
        print(f"\nnote: {out['note']}")
    u = out["usage"]
    print(f"\n{u['tokens']} tokens, {u['latency_ms'] / 1000:.1f}s model time, ~${u['est_cost_usd']:.4f}")


def cmd_check(a) -> int:
    from tbx.checker import check
    from tbx.models import Answer

    path = pathlib.Path(a.answer_json)
    try:
        answer = Answer.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        raise UsageError(f"answer file not found: {path}") from None
    except (ValueError, ValidationError) as e:
        raise UsageError(f"not a valid Answer JSON: {path}: {str(e)[:500]}") from None
    store = _open_store(a.db)
    result = check(answer, store, _window(a.from_, a.to, store.case_tz))
    print(result.model_dump_json())
    return EXIT_OK


def cmd_ingest(a) -> int:
    from tbx.ingest import IngestError, ingest_report

    try:
        summary = ingest_report(a.report_dir, _db_path(a.db), media_root=a.media_root)
    except IngestError as e:
        raise UsageError(str(e)) from None
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def cmd_enrich(a) -> int:
    from tbx.enrich import EnrichError, enrich

    voice, images = (a.voice, a.images) if (a.voice or a.images) else (True, True)
    try:
        summary = enrich(_db_path(a.db), voice=voice, images=images, offline=a.offline, media_root=a.media_root,
                         cache_path=a.cache, workers=max(1, a.workers), log=_err if a.verbose else None)
    except EnrichError as e:
        raise UsageError(str(e)) from None
    misses = summary["voice"]["misses"] + summary["images"]["misses"]
    if misses:
        _err(f"{misses} cache misses skipped (voice {summary['voice']['misses']}, "
             f"images {summary['images']['misses']} requests)" + (" in --offline mode" if a.offline else ""))
    for e in summary["errors"]:
        _err(e)
    print(json.dumps(summary, indent=2))
    return EXIT_PROVIDER if summary["errors"] and not a.offline else EXIT_OK


def cmd_export_static(a) -> int:
    from tbx.export_static import ExportError, export_static

    try:
        summary = export_static(a.out, _db_path(a.db), a.runs_dir, media_root=a.media_root,
                                featured=a.featured or os.environ.get("TBX_FEATURED_RUN") or None)
    except ExportError as e:
        raise UsageError(str(e)) from None
    print(json.dumps(summary, indent=2))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    try:
        a = _parser().parse_args(argv)
        return {"ask": cmd_ask, "check": cmd_check, "ingest": cmd_ingest, "enrich": cmd_enrich,
                "export-static": cmd_export_static}[a.cmd](a)
    except UsageError as e:
        _err(str(e))
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
