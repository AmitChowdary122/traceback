# Traceback

**An agent with typed forensic tools plus a deterministic citation verifier.**

A forensic examiner asks a question about a phone extraction ("Where was this phone between 9 and
11 pm, and who did it talk to?"). Traceback's agent, running on Meta's Muse Spark, queries the
parsed extraction through six typed tools and must submit its answer as structured claims. Each
claim cites artifact IDs, fields and values. A deterministic checker (pure code, no LLM)
verifies every citation against the store and rejects any claim it can't back up. Every
verified claim opens to the raw source row and its source path.

Built for the Meta Global AI Developer Hackathon, indie track ("Build for Your World").

> Status: under construction (Day 1). See `docs/GAMEPLAN.md` and `docs/CONTRACTS.md`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
cp .env.example .env   # then fill in keys locally; never commit .env
tbx ingest data/sample/aleapp --media-root data/sample/media --db data/tbx.db
tbx ask "Where was this phone between 9 and 11 pm, and who did it talk to?" --from "2024-07-27 21:00" --to "2024-07-27 23:00"
```

## Layout

| Path | What |
|---|---|
| `src/tbx/timeutil.py` | the one module that converts local ⇄ UTC |
| `src/tbx/store.py` | SQLite store with provenance on every row |
| `src/tbx/ingest.py` | ALEAPP report → store, with deterministic artifact IDs |
| `src/tbx/tools.py` | the six typed tools the agent calls |
| `src/tbx/checker.py` | the deterministic citation checker |
| `src/tbx/agent.py` | the tool loop, `submit_answer` validation and the JSONL trace |
| `src/tbx/cli.py` | `tbx ingest`, `tbx ask`, `tbx check` |
| `data/sample/` | cut-down ALEAPP output from a public reference image (see `data/sample/SOURCE.md`) |

## Data

The sample data comes from a public Android 14 reference image created by Joshua Hickman and
distributed by Digital Corpora. See `data/sample/SOURCE.md` for provenance and license terms.

## License

Apache-2.0. See `LICENSE`.
