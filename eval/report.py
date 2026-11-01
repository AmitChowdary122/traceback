#!/usr/bin/env python
"""Render the README results table from every eval/results/*.json (CONTRACTS.md v1.2 section 11).

    python eval/report.py            # print the Markdown table
    python eval/report.py --write    # replace the table between the eval-results markers in README.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default=str(ROOT / "eval" / "results"))
    p.add_argument("--readme", default=str(ROOT / "README.md"))
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)

    from tbx.evaluation import render_table, splice_readme

    files = sorted(pathlib.Path(a.results).glob("*.json"))
    if not files:
        print(f"report: no results in {a.results}", file=sys.stderr)
        return 1
    results = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    table = render_table(results)
    qfiles = sorted({r.get("questions_file") or "?" for r in results})
    note = (f"\nFrom {', '.join(f'`eval/results/{f.name}`' for f in files)}; questions: "
            f"{', '.join(f'`{q}`' for q in qfiles)}. Rendered by `eval/report.py`.\n")
    if a.write:
        readme = pathlib.Path(a.readme)
        readme.write_text(splice_readme(readme.read_text(encoding="utf-8"), table, note), encoding="utf-8")
        print(f"report: wrote {len(results)} result files into {readme}", file=sys.stderr)
    else:
        print(table + "\n" + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
