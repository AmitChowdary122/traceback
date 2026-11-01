# Traceback

**An agent with typed forensic tools plus a deterministic citation verifier.**

A forensic examiner asks a question about a phone extraction ("Where was this phone between 9 and
11 pm, and who did it talk to?"). Traceback's agent, running on Meta's Muse Spark, queries the
parsed extraction through six typed tools and must submit its answer as structured claims. Each
claim cites artifact IDs, fields and values. A deterministic checker (pure code, no LLM)
verifies every citation against the store and rejects any claim it can't back up. Every
verified claim opens to the raw source row and its source path.

Built for the Meta Global AI Developer Hackathon, indie track ("Build for Your World").

> Status: Day 2. The core loop (ingest, tools, checker, agent, CLI) plus `tbx enrich` (voice transcripts and SAM 3.1
> image tags), the web app, the eval harness and the Modal deploy prep. See `docs/GAMEPLAN.md` and `docs/CONTRACTS.md`.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
cp .env.example .env   # fill in keys locally; never commit .env
set -a; . ./.env; set +a   # tbx reads the environment, not the file
tbx ingest data/sample/aleapp --media-root data/sample/media --db data/tbx.db
tbx enrich --offline --db data/tbx.db   # voice transcripts + SAM image tags from the committed cache, no keys
tbx ask "Where was this phone between 9 and 11 pm, and who did it talk to?" --from "2024-06-27 21:00" --to "2024-06-27 23:00"
```

## Web app

```bash
TBX_DB=data/tbx.db uvicorn tbx.web:app --port 8000   # then open http://127.0.0.1:8000
```

- **Ask:** a question with case-local From/To times and a provider toggle (meta | baseten). It runs the same agent code
  as `tbx ask` in a background thread and writes `runs/<run_id>.jsonl`.
- **Replay:** every trace in `runs/*.jsonl` (or `TBX_RUNS_DIR`) is listed and replays with no model call. The newest
  good run (or `TBX_FEATURED_RUN`) opens by default. `TBX_RUNS_DIR=deploy/runs` replays the committed demo runs.
- **Run view:** status, the tool calls from the trace, a VERIFIED timeline in case-local time, and a REJECTED panel
  with each claim's reason codes and details.
- **Evidence drawer:** click any citation for the raw row (`raw_json`), `source_path` and `source_row`. Photos show
  their SAM 3.1 boxes scaled to the displayed size; videos show the Muse Voice Transcribe transcript; MMS rows show
  their attachment.
- **Limits:** `TBX_ASK_LIMIT_PER_HOUR` (default 30, then HTTP 429; `0` turns live asking off).

## Deploy on Modal (prepared, not deployed)

`deploy/modal_app.py` serves `tbx.web` as an ASGI app. Its image copies this repo (never `.env` files, the local store
or local runs) and runs `tbx ingest` + `tbx enrich --offline` at build time, so the hosted store is built from the
same committed inputs as a local one, with the same artifact IDs, and the build needs no API key. Keys come only from a
Modal secret. The page replays a recorded run by default (`deploy/runs/*.jsonl`, copied into the `traceback-runs`
volume on start), and live asks are capped at `TBX_ASK_LIMIT_PER_HOUR` (default 30 per hour, then HTTP 429).

Steps, once a Modal token exists (only the account owner can create one):

1. Create a token at modal.com (Settings → API tokens). On a laptop, `modal token new` stores it. For the Claude Code
   cloud environment, open the environment menu in the session's title bar → Edit, add `MODAL_TOKEN_ID` and
   `MODAL_TOKEN_SECRET` under Network secrets (or as environment variables where that section is not offered), and
   under Network access add `api.modal.com` (plus `modal.run` to open the deployed URL from there) to Allowed domains
   ([steps](https://code.claude.com/docs/en/cloud-environments#network-access)). A new session picks both up.
2. Install the CLI: `pip install -e '.[deploy]'` (or `pip install modal`).
3. Store the provider keys as a Modal secret (they are never baked into the image):
   `modal secret create traceback-keys META_API_KEY=... BASETEN_API_KEY=...`
4. Try it: `modal serve deploy/modal_app.py` prints a temporary URL with hot reload.
5. Deploy: `modal deploy deploy/modal_app.py` prints the stable `https://<workspace>--traceback-web.modal.run` URL.

To change the ask cap or the featured run, edit `TBX_ASK_LIMIT_PER_HOUR` / `FEATURED_RUN` in `deploy/modal_app.py` and
deploy again; `TBX_ASK_LIMIT_PER_HOUR=0` makes the site replay-only. `max_containers=1` keeps the cap global.

## Eval

```bash
python eval/run.py --provider meta --model muse-spark-1.3 [--repair 1]           # live, about $0.05 per question
python eval/run.py --provider baseten --model meta-models/muse-spark-open [--repair 1]
python eval/report.py --write                                                     # refresh the table below
```

`eval/run.py` sends every question in `eval/questions.yaml` through the same agent code as `tbx ask`, keeps each trace
in `runs/`, and writes `eval/results/<provider>-<model>[-repair].json` with per-question scores and the totals of
CONTRACTS §11 (fact recall, artifact recall, citation precision, `uncited_in_report`, rejections by reason,
abstention accuracy, latency p50/p95, tokens, cost).

## Eval results

<!-- eval-results:start -->
No results yet: `eval/questions.yaml` and the eval runs come from the testing session.
<!-- eval-results:end -->

## Layout

| Path | What |
|---|---|
| `src/tbx/timeutil.py` | the one module that converts local ⇄ UTC |
| `src/tbx/store.py` | SQLite store with provenance on every row |
| `src/tbx/ingest.py` | ALEAPP report → store, with deterministic artifact IDs |
| `src/tbx/tools.py` | the six typed tools the agent calls |
| `src/tbx/checker.py` | the deterministic citation checker |
| `src/tbx/enrich.py` | `tbx enrich`: Muse Voice Transcribe and SAM 3.1, cached in `data/sample/derived/enrich_cache.jsonl` |
| `src/tbx/agent.py` | the tool loop, `submit_answer` validation and the JSONL trace |
| `src/tbx/web/` | `tbx.web`: the FastAPI app and its single HTML page (no build step) |
| `src/tbx/cli.py` | `tbx ingest`, `tbx enrich`, `tbx ask`, `tbx check` |
| `deploy/` | `modal_app.py` (hosted web app) and `runs/` (recorded demo runs it replays) |
| `src/tbx/evaluation.py`, `eval/` | the eval harness (`eval/run.py`) and the README table (`eval/report.py`) |
| `data/sample/` | cut-down ALEAPP output from a public reference image (see `data/sample/SOURCE.md`) |

## Data

The sample data comes from a public Android 14 reference image created by Joshua Hickman and
distributed by Digital Corpora. See `data/sample/SOURCE.md` for provenance and license terms.

## License

Apache-2.0. See `LICENSE`.
