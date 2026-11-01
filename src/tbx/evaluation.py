"""Eval harness (CONTRACTS.md v1.2 section 11): run questions through the `tbx ask` code path and score them.

Used by eval/run.py (runs and writes eval/results/<provider>-<model>[-repair].json) and eval/report.py
(renders the README results table). Scoring is pure: it reads the run results, never calls a model.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from tbx import timeutil as tu
from tbx.checker import _PHONE, fold, phones_match
from tbx.store import Store

PHONE_ENTITY = re.compile(r"\+?\d{7,15}")
COORD_ENTITY = re.compile(r"\s*(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)\s*")
REPORT_START, REPORT_END = "<!-- eval-results:start -->", "<!-- eval-results:end -->"


# ------------------------------------------------------------------------------ questions
def load_questions(path: str | pathlib.Path) -> list[dict]:
    import yaml

    qs = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(qs, list):
        raise ValueError(f"{path}: expected a YAML list of questions")
    seen = set()
    for q in qs:
        for k in ("id", "question"):
            if not q.get(k):
                raise ValueError(f"{path}: a question is missing {k!r}: {q}")
        if q["id"] in seen:
            raise ValueError(f"{path}: duplicate id {q['id']!r}")
        seen.add(q["id"])
    return qs


def window_of(q: dict, case_tz: str) -> tu.Window | None:
    w = q.get("window") or {}
    if not w.get("from") and not w.get("to"):
        return None
    start = tu.parse_user_time(str(w["from"]), case_tz) if w.get("from") else datetime(1970, 1, 1, tzinfo=timezone.utc)
    end = tu.parse_user_time(str(w["to"]), case_tz) if w.get("to") else datetime(2100, 1, 1, tzinfo=timezone.utc)
    return tu.Window(start, end)


def is_negative(q: dict) -> bool:
    return "negative" in (q.get("tags") or [])


# ------------------------------------------------------------------------------ scoring
def _claim_values(claim: dict) -> list[str]:
    return [str(c.get("value", "")) for c in claim.get("citations", [])]


def entity_found(entity: str, verified: list[dict], store: Store | None = None) -> bool:
    """Is an expected entity in the verified claims' text or cited values, normalized as in CONTRACTS section 7?"""
    e = str(entity).strip()
    texts = [c.get("text", "") for c in verified]
    values = [v for c in verified for v in _claim_values(c)]
    if PHONE_ENTITY.fullmatch(e.replace(" ", "")):  # phones are given as digits (section 11)
        cands = list(values)
        for s in texts + values:
            cands += [m.group(0) for m in _PHONE.finditer(s)]
        return any(phones_match(e, c) for c in cands)
    m = COORD_ENTITY.fullmatch(e)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        for c in verified:
            got: dict[str, dict[str, float]] = {}
            for cit in c.get("citations", []):
                if cit.get("field") in ("lat", "lon"):
                    try:
                        got.setdefault(cit["artifact_id"], {})[cit["field"]] = float(cit["value"])
                    except (TypeError, ValueError):
                        pass
            if any(abs(p.get("lat", 1e9) - lat) <= 1e-4 and abs(p.get("lon", 1e9) - lon) <= 1e-4 for p in got.values()):
                return True
        return False
    if store is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}.*", e):
        try:
            want, _ = tu.parse_ts(e, store.case_tz)
        except ValueError:
            want = None
        if want is not None:
            for c in verified:
                for cit in c.get("citations", []):
                    row = store.get(cit.get("artifact_id", ""))
                    if row and row.get("ts_utc") and tu.parse_utc(row["ts_utc"]) == want.replace(microsecond=0):
                        return True
    needle = fold(e)
    return bool(needle) and any(needle in fold(s) for s in texts + values)


def score_question(q: dict, verified: list[dict], rejected: list[dict], store: Store | None = None) -> dict:
    ents = list(q.get("expected_entities") or [])
    ids = list(q.get("expected_artifact_ids") or [])
    found_e = [e for e in ents if entity_found(e, verified, store)]
    cited = {c.get("artifact_id") for cl in verified for c in cl.get("citations", [])}
    found_a = [a for a in ids if a in cited]
    n_claims = len(verified) + len(rejected)
    return {
        "expected_entities": len(ents), "found_entities": len(found_e),
        "missing_entities": [e for e in ents if e not in found_e],
        "expected_artifacts": len(ids), "found_artifacts": len(found_a),
        "missing_artifact_ids": [a for a in ids if a not in found_a],
        "fact_recall": (len(found_e) / len(ents)) if ents else None,
        "artifact_recall": (len(found_a) / len(ids)) if ids else None,
        "citation_precision": (len(verified) / n_claims) if n_claims else None,
        "uncited_in_report": sum(1 for c in verified if not c.get("citations")),
    }


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def totals(per_q: list[dict], questions: list[dict]) -> dict:
    by_id = {q["id"]: q for q in questions}
    exp_e = sum(r["expected_entities"] for r in per_q)
    exp_a = sum(r["expected_artifacts"] for r in per_q)
    n_v = sum(len(r["verified"]) for r in per_q)
    n_r = sum(len(r["rejected"]) for r in per_q)
    reasons = Counter(x["code"] for r in per_q for rj in r["rejected"] for x in rj.get("reasons", []))
    reasons_r1 = Counter(x["code"] for r in per_q for rj in (r.get("first_round") or {}).get("rejected", [])
                         for x in rj.get("reasons", []))
    neg = [r for r in per_q if is_negative(by_id.get(r["id"], {}))]
    lat = [r["latency_s"] for r in per_q if r.get("latency_s") is not None]
    return {
        "questions": len(per_q),
        "errors": sum(1 for r in per_q if r.get("status") != "ok"),
        "fact_recall": round(sum(r["found_entities"] for r in per_q) / exp_e, 4) if exp_e else None,
        "artifact_recall": round(sum(r["found_artifacts"] for r in per_q) / exp_a, 4) if exp_a else None,
        "citation_precision": round(n_v / (n_v + n_r), 4) if (n_v + n_r) else None,
        "verified_claims": n_v, "rejected_claims": n_r,
        "uncited_in_report": sum(r["uncited_in_report"] for r in per_q),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "rejected_by_reason_round1": dict(sorted(reasons_r1.items())),
        "abstention_accuracy": round(sum(1 for r in neg if not r["verified"]) / len(neg), 4) if neg else None,
        "negative_questions": len(neg),
        "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
        "tokens": sum(r.get("tokens") or 0 for r in per_q),
        "est_cost_usd": round(sum(r.get("est_cost_usd") or 0.0 for r in per_q), 6),
    }


# ------------------------------------------------------------------------------ running
def result_name(provider: str, model: str, repair: int) -> str:
    return f"{provider}-{model.replace('/', '_')}{'-repair' if repair else ''}.json"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=pathlib.Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_eval(questions: list[dict], store: Store, client: Any, repair: int = 0, runs_dir: str | pathlib.Path = "runs",
             log: Callable[[str], None] | None = None) -> dict:
    """Run every question through tbx.agent.run_agent (the `tbx ask` code path) and score it."""
    from tbx.agent import AgentError, new_run_id, run_agent
    from tbx.llm import ProviderError

    log = log or (lambda s: None)
    per_q = []
    started = tu.now_utc()
    for q in questions:
        window = window_of(q, store.case_tz)
        run_id = new_run_id()
        t0 = time.monotonic()
        status, error, usage, verified, rejected, first = "ok", None, {}, [], [], None
        try:
            res = run_agent(q["question"], store, window, client, repair=repair, trace_dir=runs_dir, run_id=run_id)
            usage = res.usage
            verified = [c.model_dump() for c in res.check.verified]
            rejected = [r.model_dump() for r in res.check.rejected]
            if res.first_check is not None:
                first = {"verified": [c.model_dump() for c in res.first_check.verified],
                         "rejected": [r.model_dump() for r in res.first_check.rejected]}
        except (AgentError, ProviderError) as e:
            status, error = "error", f"{type(e).__name__}: {e}"[:500]
            usage = _usage_from_trace(pathlib.Path(runs_dir) / f"{run_id}.jsonl")
        latency = round(time.monotonic() - t0, 3)
        row = {"id": q["id"], "run_id": run_id, "status": status, "error": error, "question": q["question"],
               "tags": q.get("tags") or [], "verified": verified, "rejected": rejected,
               **score_question(q, verified, rejected, store), "latency_s": latency,
               "tokens": usage.get("tokens", 0), "est_cost_usd": usage.get("est_cost_usd", 0.0)}
        if first is not None:
            row["first_round"] = first
        per_q.append(row)
        log(f"{q['id']}: {status} verified={len(verified)} rejected={len(rejected)} "
            f"fact={_fmt(row['fact_recall'])} artifact={_fmt(row['artifact_recall'])} {latency:.1f}s "
            f"${row['est_cost_usd']:.4f}")
    return {"provider": client.provider, "model": client.model, "repair": repair, "case_tz": store.case_tz,
            "started": started, "finished": tu.now_utc(), "commit": _git_commit(),
            "totals": totals(per_q, questions), "questions": per_q}


def _usage_from_trace(path: pathlib.Path) -> dict:
    try:
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            e = json.loads(line)
            if e.get("type") == "run_end":
                return e["payload"].get("totals", {})
    except (OSError, ValueError, KeyError):
        pass
    return {}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}"


# ------------------------------------------------------------------------------ report
def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def render_table(results: list[dict]) -> str:
    head = ("| Provider | Model | Repair | Questions | Fact recall | Artifact recall | Citation precision | "
            "Abstention | Uncited | Rejected (top reasons) | p50 / p95 latency | Tokens | Cost |")
    sep = "|" + "---|" * 13
    rows = [head, sep]
    for r in sorted(results, key=lambda r: (r.get("provider", ""), r.get("model", ""), r.get("repair", 0))):
        t = r.get("totals", {})
        top = sorted(t.get("rejected_by_reason", {}).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        rej = f"{t.get('rejected_claims', 0)}" + (f" ({', '.join(f'{k} {v}' for k, v in top)})" if top else "")
        lat = (f"{t['latency_p50_s']:.1f} s / {t['latency_p95_s']:.1f} s"
               if t.get("latency_p50_s") is not None else "—")
        errs = f" ({t['errors']} errors)" if t.get("errors") else ""
        rows.append(f"| {r.get('provider')} | `{r.get('model')}` | {'yes' if r.get('repair') else 'no'} | "
                    f"{t.get('questions', 0)}{errs} | "
                    f"{_pct(t.get('fact_recall'))} | {_pct(t.get('artifact_recall'))} | "
                    f"{_pct(t.get('citation_precision'))} | {_pct(t.get('abstention_accuracy'))} | "
                    f"{t.get('uncited_in_report', '—')} | {rej} | {lat} | {t.get('tokens', 0):,} | "
                    f"${t.get('est_cost_usd', 0):.2f} |")
    return "\n".join(rows)


def splice_readme(readme: str, table: str, note: str = "") -> str:
    block = f"{REPORT_START}\n{table}\n{note}{REPORT_END}" if note else f"{REPORT_START}\n{table}\n{REPORT_END}"
    if REPORT_START in readme and REPORT_END in readme:
        pre, rest = readme.split(REPORT_START, 1)
        _, post = rest.split(REPORT_END, 1)
        return pre + block + post
    return readme.rstrip() + "\n\n## Eval results\n\n" + block + "\n"
