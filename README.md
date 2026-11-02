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
> image tags), the web app, the eval harness, the Modal deploy prep and a replay-only static site.
> See `docs/GAMEPLAN.md` and `docs/CONTRACTS.md`.

## Architecture

```mermaid
flowchart LR
  subgraph inputs["data/sample/ (committed)"]
    R["ALEAPP report<br/>TSV exports"]
    M["Media files<br/>photos, MMS videos"]
    C[("enrich_cache.jsonl<br/>recorded once, live")]
  end
  R --> ING["tbx ingest<br/>deterministic artifact IDs"]
  M --> ING
  ING --> S[("SQLite store<br/>provenance on every row")]
  M --> ENR["tbx enrich"]
  C <--> ENR
  ENR -- "WAV soundtrack" --> V["Muse Voice Transcribe"]
  ENR -- "photo + one label" --> SAM["SAM 3.1"]
  ENR -- "transcripts, image_tags" --> S
  Q["Question + case-local window"] --> AG["Agent loop<br/>tbx ask, web app, eval"]
  AG <-- "tool calls / rows" --> T["6 typed, read-only tools"]
  T --> S
  AG <-- "chat + tools" --> LLM["Muse Spark 1.3 (Meta)<br/>Muse Spark Open (Baseten)"]
  AG -- "submit_answer:<br/>claims + citations" --> CK["Checker<br/>pure code, no LLM"]
  S --> CK
  CK --> OUT["VERIFIED / REJECTED<br/>+ JSONL trace in runs/"]
  OUT --> UI["Web UI<br/>live or replay"]
  OUT --> SITE["Static replay site<br/>tbx export-static"]
```

- **Ingest** maps ALEAPP's TSV exports and the media files onto six tables (`messages`, `calls`, `locations`, `media`,
  `transcripts`, `image_tags`). Each row keeps `source_path`, `source_row`, `ts_original`, `tz_basis` and the raw
  source row as `raw_json`. Its ID is `<prefix>_<sha1(source_path|row_key)[:10]>`, so it can be recomputed from its own
  provenance (CONTRACTS §2).
- **Tools** (`search_messages`, `get_calls`, `get_locations`, `find_media`, `get_transcripts`, `get_artifact`) are
  read-only, ordered deterministically and capped at 200 rows. They are the model's only view of the extraction.
- **The agent** loops on tool calls until the model calls `submit_answer`. It validates that call with pydantic,
  retries once on a failure, and caps the loop at 12 steps. `tool_choice` is always `"auto"`. Reasoning text is
  stripped and never sent back or traced.
- **The checker** decides what counts as a finding. Only verified claims reach the VERIFIED timeline; rejected claims
  are shown apart, with their reasons, and the model's `note` is labelled "not a finding".

## How verification works

Every claim is `{text, time_start?, time_end?, citations: [{artifact_id, field, value}]}`. `tbx.checker.check()` is
pure and deterministic (no network, no LLM, no clock; the same input gives byte-identical output) and **strict**: a
claim with any reason is rejected, and every reason is listed, with the index of the citation it applies to where
there is one. The reason codes are a stable API (CONTRACTS §7):

| Rule | Code | Rejects a claim when |
|---|---|---|
| R1 | `UNKNOWN_ARTIFACT` | a cited `artifact_id` is not in the store (for example a tool-call ID copied by mistake) |
| R2 | `UNKNOWN_FIELD` | the cited field is not citable for that table (`raw_json`, `tz_basis` and `ts_original` never are) |
| R2 | `VALUE_MISMATCH` | the cited value doesn't match the stored value after normalization |
| R3 | `OUTSIDE_WINDOW` | a question window is set and a cited artifact's time is outside it, or it has no timestamp |
| R4 | `UNGROUNDED_ENTITY:<type>` | the claim text names a phone number, email, URL, timestamp, date, clock time, coordinate or known contact that none of the claim's own cited values supports |
| R5 | `TIME_RANGE_UNSUPPORTED` | `time_start`/`time_end` is reversed, or falls outside the cited artifacts' times (±60 s; a call's end counts) |
| R6 | `NO_CITATIONS` | the claim cites nothing |

How values are compared:

| Kind | Fields | Rule |
|---|---|---|
| timestamp | `ts_utc`, `ts_local` | parsed and compared in UTC, to the second (to the minute when the cited value has no seconds) |
| phone | `number`, `*_number` | digits only; the last 10 digits when both sides have at least 10 |
| text | `body`, `text`, `transcript` | NFKC, casefold, collapsed whitespace; equal, or a quote of at least 8 characters from the stored text |
| coordinate | `lat`, `lon` | within 1e-4 |
| number / measure | `score` / `duration_s`, `accuracy_m`, `size_bytes` | within 1e-3 / within 1e-6, so `"95"` matches a stored `95` |
| everything else | | NFKC, casefold, collapsed whitespace; exact |

R4 extracts entities from the claim text in a fixed order (URL, email, timestamp, date, clock time, coordinate,
phone), so `2023-03-14T01:42:07Z` is one timestamp and not also a date, a clock time and a phone number. Dates and
clock times (`9:42 PM`, `21:42`) are grounded against the cited timestamps in the case timezone, and contact names
against the cited values. So a claim that restates the question window's times ("between 9:00 PM and 11:00 PM", or the
window's ISO bounds) is rejected unless a cited artifact carries those times: the window is not evidence. The system
prompt also tells the model never to restate the window.

With `--repair 1`, rejected claims and their reasons go back to the model once; the final report is the check of
its second answer, and the trace keeps both rounds. The red-team suite (`tests/redteam/`, 76 cases) pins every rule,
including real model claims recorded in the API spikes.

## Meta tooling used

| Model | Where | What Traceback uses it for |
|---|---|---|
| **Muse Spark** `muse-spark-1.3` | Meta Model API (`/chat/completions`, OpenAI-compatible) | the agent: tool calls over the store, then `submit_answer`. The default provider, and the only one asked live on the hosted site. |
| **Muse Spark Open** `meta-models/muse-spark-open` | Baseten (`inference.baseten.co`) | the same agent on open weights (`--provider baseten`). It sometimes writes its tool call as text; the agent parses those blocks for known tool names only, marks them `tool_calls_from_text` in the trace, and the checker still verifies every claim. The hosted site replays recorded Baseten runs. |
| **Muse Voice Transcribe** `muse-voice-transcribe-1.0` | Meta Model API (`/asr/transcribe`, mode `ENDPOINTING`) | transcripts of the two MMS video soundtracks, the only recorded audio in the image. Sent as 24 kHz mono WAV with **no keyword biasing**, so the ASR can't be steered toward the case's names. |
| **SAM 3.1** `sam-3.1` | Meta Model API (`/responses`) | boxes on the 60 JPEG/PNG images (camera photos and image attachments), one request per label of prompt set `v1` (`person`, `cell phone`, `laptop`, `car`, `license plate`, `sign`). 225 boxes with score ≥ 0.5 become citable `image_tags` rows, drawn on the photo in the UI. |

Voice and SAM results are recorded once into `data/sample/derived/enrich_cache.jsonl`, so everything below runs
offline.

## Run everything offline (no keys, no credits)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
tbx ingest data/sample/aleapp --media-root data/sample/media --db data/tbx.db
tbx enrich --offline --db data/tbx.db       # 2 transcripts + 225 SAM tags from the committed cache
TBX_DB=data/tbx.db TBX_RUNS_DIR=deploy/runs uvicorn tbx.web:app --port 8000
```

Open http://127.0.0.1:8000. With no keys set, the page replays the recorded runs in `deploy/runs/` with no model call:
the trace's tool calls, the verified timeline, the rejected panel, and the evidence drawer with the raw row, the photo
with its SAM boxes, or the video with its transcript. `tbx check answer.json --db data/tbx.db` re-runs the checker on
any saved answer, and `TBX_STRICT_TESTS=1 pytest -q` runs all the suites offline.

## Ask live

```bash
cp .env.example .env   # fill in keys locally; never commit .env
set -a; . ./.env; set +a   # tbx reads the environment, not the file
tbx ask "Where was this phone between 9 and 11 pm, and who did it talk to?" --from "2024-06-27 21:00" --to "2024-06-27 23:00"
tbx ask "What do the photos taken on July 24, 2024 show?" --from "2024-07-24 00:00" --to "2024-07-24 23:59" --provider baseten
```

A run costs about $0.05. In the web app, asks run the same agent code as `tbx ask` in a background thread and write
`runs/<run_id>.jsonl`.

## Web app

```bash
TBX_DB=data/tbx.db uvicorn tbx.web:app --port 8000   # then open http://127.0.0.1:8000
```

- **Ask:** a question with case-local From/To times and a provider toggle (meta | baseten).
- **Live providers:** `TBX_LIVE_PROVIDERS` (comma-separated, default `meta`) lists the providers this server asks
  live. Asking any other provider returns HTTP 403, and its toggle lists that provider's recorded runs to replay
  instead. `TBX_LIVE_PROVIDERS=meta,baseten` turns Baseten asks on; set but empty turns live asking off.
- **Replay:** every trace in `runs/*.jsonl` (or `TBX_RUNS_DIR`) is listed and replays with no model call. The newest
  good run (or `TBX_FEATURED_RUN`) opens by default. `TBX_RUNS_DIR=deploy/runs` replays the committed demo runs.
- **Run view:** status, the tool calls from the trace, a VERIFIED timeline in case-local time, and a REJECTED panel
  with each claim's reason codes and details.
- **Evidence drawer:** click any citation for the raw row (`raw_json`), `source_path` and `source_row`. Photos show
  their SAM 3.1 boxes scaled to the displayed size; videos show the Muse Voice Transcribe transcript; MMS rows show
  their attachment.
- **Limits:** `TBX_ASK_LIMIT_PER_HOUR` (default 30, then HTTP 429; `0` turns live asking off).

## Static replay site (GitHub Pages, no server)

```bash
tbx export-static --out site      # after ingest + enrich --offline; reads deploy/runs/ by default
python -m http.server -d site 8080   # check it locally at http://127.0.0.1:8080
```

`tbx export-static` drives the web app in-process and writes what the page reads as plain files: `index.html`,
`api/case.json`, `api/runs.json`, `api/runs/<id>.json`, `api/artifacts/<id>.json` for every cited row and the media
behind it, `api/media/<id>/tags.json` and `transcripts.json`, and the cited media files under `media/`. The page
detects static mode, reads those files with relative paths (so it works under `https://<user>.github.io/<repo>/`) and
disables the ask form with a short note. No server, keys or credits are involved, and the export refuses to finish if
a key value from the environment appears in any file.

To publish it on GitHub Pages (the repository must be public, or on a plan that allows Pages for private repos):

1. **With the included workflow:** in the repository, Settings → Pages → Build and deployment → Source: **GitHub
   Actions**. Then Actions → **Static replay site** → Run workflow. It rebuilds the store offline, exports `site/` and
   deploys it; the URL appears in the run summary (`https://<user>.github.io/<repo>/`). It only runs when started by
   hand.
2. **Or by hand, from a branch:** build `site/` as above, then put its contents on a `gh-pages` branch and choose
   Settings → Pages → Source: **Deploy from a branch** → `gh-pages` / `(root)`:

   ```bash
   git worktree add --orphan -b gh-pages ../tbx-pages   # git 2.42+
   cp -r site/. ../tbx-pages/
   git -C ../tbx-pages add -A
   git -C ../tbx-pages commit -m "Static replay site"
   git -C ../tbx-pages push -u origin gh-pages
   ```

## Deploy on Modal (prepared, not deployed)

`deploy/modal_app.py` serves `tbx.web` as an ASGI app. Its image copies this repo (never `.env` files, the local store
or local runs) and runs `tbx ingest` + `tbx enrich --offline` at build time, so the hosted store is built from the
same committed inputs as a local one, with the same artifact IDs, and the build needs no API key. Keys come only from a
Modal secret. The page replays a recorded run by default (`deploy/runs/*.jsonl`, copied into the `traceback-runs`
volume on start). Live asks go to Meta only (`TBX_LIVE_PROVIDERS=meta`) and are capped at `TBX_ASK_LIMIT_PER_HOUR`
(default 30 per hour, then HTTP 429); the baseten toggle replays the recorded Muse Spark Open runs.

Steps, once a Modal token exists (only the account owner can create one):

1. Create a token at modal.com (Settings → API tokens). On a laptop, `modal token new` stores it. For the Claude Code
   cloud environment, open the environment menu in the session's title bar → Edit, add `MODAL_TOKEN_ID` and
   `MODAL_TOKEN_SECRET` under Network secrets (or as environment variables where that section is not offered), and
   under Network access add `api.modal.com` (plus `modal.run` to open the deployed URL from there) to Allowed domains
   ([steps](https://code.claude.com/docs/en/cloud-environments#network-access)). A new session picks both up.
2. Install the CLI: `pip install -e '.[deploy]'` (or `pip install modal`).
3. Store the Meta key as a Modal secret (it is never baked into the image; no Baseten key is needed, since live asks
   are Meta-only): `modal secret create traceback-keys META_API_KEY=...`
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
No results yet. The testing session writes `eval/questions.yaml` and runs the eval; `python eval/report.py --write`
then renders the table here from `eval/results/*.json`.
<!-- eval-results:end -->

## Data provenance

The sample is a cut-down excerpt of the ALEAPP report for the public **Android 14 reference image** created by
**Joshua Hickman** and distributed by **Digital Corpora** (a Google Pixel 7a, Android 14, Google Fi; persona
"Liz Dehner", populated Jan 26 – Jul 28, 2024; case timezone `America/New_York`). `data/sample/SOURCE.md` has the full
record: the archive's SHA-256, the extraction path inside it, the ALEAPP version and commit, and the license terms.

- Every TSV row in `data/sample/aleapp/_TSV Exports/` is real ALEAPP output, copied byte for byte, so row numbers and
  the artifact IDs derived from them match the full report. No row was added or edited, and nothing was synthesized.
- `data/sample/media/` mirrors the extraction root: 40 camera photos (downscaled to at most 1,600 px, EXIF kept) and
  26 Google Messages attachments, verbatim, including the two MMS videos. `MANIFEST.json` lists every file's hash and
  the original's hash.
- `data/sample/derived/enrich_cache.jsonl` is machine output (Muse Voice Transcribe and SAM 3.1), not evidence from
  the image.
- Digital Corpora's original content is CC0. Third-party content inside the image (app assets, marketing images and
  videos received by SMS) belongs to its owners and is kept as evidence excerpts for research and demonstration.
  Traceback's own code is Apache-2.0; that license does not cover `data/sample/`.

## Limitations

- **Prose and place names aren't checked.** The checker verifies the cited values and the entities it can extract
  (phones, emails, URLs, dates, clock times, coordinates, known contacts). Free text such as "at a gas station" or
  "near Raleigh" is not verified. A verified claim is backed by real rows, but an examiner still reviews it.
- **SAM and ASR output is machine output.** A `tag_` row means SAM 3.1 drew that box with that score, not that the
  object is there: "shows a person" can rest on an arm at the edge of a photo (person 0.88). Transcripts can mishear
  names; the two MMS soundtracks spell the same candidate "Satana Deberry" and "Satana Dewberry". The UI marks both as
  machine output.
- **An MMS can appear as several rows.** ALEAPP lists an MMS's text and each attachment as separate message rows with
  the same `thread_id` and time. The prompt tells the model to describe them as one message, but a model can still
  count them separately.
- **The image has no voice notes.** The 2,154 Google Fi voicemail files are 0 bytes (the audio stays with the
  carrier). Only the carrier's transcription text survives, for 49 of them, as `calls.transcript`, and it can't be
  checked against audio. The only recorded audio is the soundtrack of the two MMS videos.
- **Phone numbers compare on the last 10 digits**, so two numbers that differ only in their country code match.
- **Photo coordinates are rounded to 7 decimals** at ingest (about 1 cm); the EXIF rationals stay in `raw_json`.

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
| `src/tbx/export_static.py` | `tbx export-static`: the replay-only static site |
| `src/tbx/cli.py` | `tbx ingest`, `tbx enrich`, `tbx ask`, `tbx check`, `tbx export-static` |
| `deploy/` | `modal_app.py` (hosted web app) and `runs/` (recorded demo runs it replays) |
| `src/tbx/evaluation.py`, `eval/` | the eval harness (`eval/run.py`) and the README table (`eval/report.py`) |
| `data/sample/` | cut-down ALEAPP output from a public reference image (see `data/sample/SOURCE.md`) |

## License

Apache-2.0. See `LICENSE`.
