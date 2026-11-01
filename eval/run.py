#!/usr/bin/env python
"""Run the eval questions through the `tbx ask` code path (CONTRACTS.md v1.2 section 11).

    python eval/run.py --provider meta --model muse-spark-1.3 [--repair 0|1] \
        [--questions eval/questions.yaml] [--db data/tbx.db] [--only q01,q02]

Writes eval/results/<provider>-<model>[-repair].json ("/" in the model ID becomes "_") and keeps every
run's trace in runs/<run_id>.jsonl. Live and billed: about $0.05 per question.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--provider", default=None, help="meta | baseten | modal (default TBX_PROVIDER, then meta)")
    p.add_argument("--model", default=None, help="default: the provider's default model")
    p.add_argument("--repair", type=int, choices=[0, 1], default=0)
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.yaml"))
    p.add_argument("--db", default=os.environ.get("TBX_DB") or str(ROOT / "data" / "tbx.db"))
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--only", default=None, help="comma-separated question ids")
    a = p.parse_args(argv)

    from tbx.evaluation import load_questions, result_name, run_eval
    from tbx.llm import ConfigError, LLMClient
    from tbx.store import Store

    if not pathlib.Path(a.db).is_file():
        print(f"eval: store not found: {a.db} (run tbx ingest + tbx enrich --offline first)", file=sys.stderr)
        return 1
    try:
        questions = load_questions(a.questions)
    except (OSError, ValueError) as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1
    if a.only:
        keep = set(a.only.split(","))
        questions = [q for q in questions if q["id"] in keep]
    try:
        client = LLMClient(a.provider, a.model)
    except ConfigError as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1
    store = Store(a.db)
    result = run_eval(questions, store, client, repair=a.repair, runs_dir=a.runs_dir,
                      log=lambda s: print(s, file=sys.stderr, flush=True))
    result["questions_file"] = os.path.relpath(a.questions, ROOT)
    out = pathlib.Path(a.out_dir) / result_name(client.provider, client.model, a.repair)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t = result["totals"]
    print(json.dumps({"out": str(out), **{k: t[k] for k in ("questions", "errors", "fact_recall", "artifact_recall",
                                                            "citation_precision", "uncited_in_report",
                                                            "abstention_accuracy", "tokens", "est_cost_usd")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
