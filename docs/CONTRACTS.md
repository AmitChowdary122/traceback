# Traceback contracts (v1.2)

These are the interfaces that the build session implements and the testing session tests against. Change them only by editing this file in a commit of its own, with a reason, so tests and code stay in sync.

The ALEAPP column names are not fixed here. Ingest maps whatever the real sample output contains onto the store columns below.

---

## 1. Configuration

| Variable | Example | Notes |
|---|---|---|
| `META_API_KEY` | (secret) | Meta's docs call it `MODEL_API_KEY`; we use `META_API_KEY` |
| `META_BASE_URL` | `https://api.meta.ai/v1` | OpenAI-compatible |
| `BASETEN_API_KEY` | (secret) | |
| `BASETEN_BASE_URL` | `https://inference.baseten.co/v1` | |
| `MODAL_BASE_URL`, `MODAL_API_KEY` | | optional |
| `TBX_PROVIDER` | `meta` \| `baseten` \| `modal` | default `meta` |
| `TBX_MODEL` | `muse-spark-1.3` | Baseten: `meta-models/muse-spark-open` (the `-1026` suffix 404s; spike S4) |
| `TBX_DB` | `data/tbx.db` | default store path |

- **Model choice:** never use the `-contributor` model variants, because Meta's contributor mode is a different tier.
- **Secrets:** they are never logged, written to traces or committed. `.env.example` lists the names only.

## 2. Artifact IDs

```
artifact_id = f"{prefix}_{sha1(f'{source_path}|{row_key}'.encode()).hexdigest()[:10]}"
```

| Table | Prefix |
|---|---|
| messages | `msg` |
| calls | `call` |
| locations | `loc` |
| media | `med` |
| transcripts | `trn` |
| image_tags | `tag` |

- `source_path`: POSIX path **relative to the ALEAPP report root** (or to the extraction root, for media files). Never absolute.
- `row_key`:
  - the source's own natural key when ALEAPP exposes it (e.g. the source DB `_id` / rowid)
  - otherwise the 1-based data row number in the TSV/table
  - for derived tables, `transcripts` uses `"{media_id}|{model}"` and `image_tags` uses `"{media_id}|{model}|{label}|{n}"`. `n` is the 0-based position of the box among that media's boxes for that label, ordered by `score` descending, then `x0`, then `y0`. Derived rows use the media's `source_path`.
- **`source_row` stores the `row_key`** used in the formula, as a string. Any ID can then be recomputed from its own row's `source_path` and `source_row`.
- **Determinism:** ingesting the same input twice must produce identical ID sets. Ingest fails loudly on a collision.

## 3. Time

- **`ts_utc`:** ISO 8601 UTC with a `Z` suffix and second precision, e.g. `2023-03-14T01:42:07Z`. It is `NULL` only when the source has no timestamp.
- **`ts_original`:** the source value exactly as found, as a string.
- **`tz_basis`** is one of:
  - `utc_epoch_s` / `utc_epoch_ms`
  - `utc_string`
  - `offset_string`: a string with an explicit UTC offset, converted with that offset (e.g. EXIF `DateTimeOriginal` + `OffsetTimeOriginal`)
  - `local_converted:<IANA tz>`
  - `unknown`
- **Case timezone:** `case_meta.device_tz` (IANA, e.g. `America/New_York`), read from the extraction. All local⇄UTC conversion goes through one module, `tbx/timeutil.py`.
- **Question window:** `tbx.timeutil.Window(start_utc, end_utc)`. Both arguments are timezone-aware `datetime`s in UTC, passed positionally, and both ends are inclusive. CLI/UI inputs without an offset are interpreted in the case timezone; inputs with `Z` or an offset are used as given.

## 4. Store (SQLite)

Every family table has these provenance columns:

```
artifact_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, source_row TEXT NOT NULL,
ts_utc TEXT, ts_original TEXT, tz_basis TEXT NOT NULL, raw_json TEXT NOT NULL
```

Each table adds its own columns. These are also the **citable fields**:

| Table | Columns |
|---|---|
| `messages` | `app`, `thread_id`, `direction` (`incoming`\|`outgoing`\|`unknown`), `sender`, `sender_number`, `recipient`, `recipient_number`, `body`, `attachment_media_id` |
| `calls` | `app`, `direction` (`incoming`\|`outgoing`\|`missed`\|`rejected`\|`voicemail`\|`unknown`), `number`, `contact_name`, `duration_s`, `transcript` (the carrier's voicemail transcription text; `NULL` when there is none, and placeholders such as "Transcript not available" are stored as `NULL`) |
| `locations` | `source` (e.g. `google_location_history`, `photo_exif`, `wifi`), `lat`, `lon`, `accuracy_m` |
| `media` | `app`, `kind` (`image`\|`audio`\|`video`\|`other`), `file_path`, `mime_type`, `sha256`, `size_bytes`, `lat`, `lon` |
| `transcripts` | `media_id`, `model`, `language`, `text`, `duration_s` (`ts_utc` = the media's `ts_utc`) |
| `image_tags` | `media_id`, `model`, `label`, `score`, `box` (JSON `[x0,y0,x1,y1]` in pixels), `prompt_set` (`ts_utc` = the media's `ts_utc`) |
| `case_meta` | `key TEXT PRIMARY KEY, value TEXT`. Keys: `device_tz`, `device_model`, `aleapp_version`, `report_root`, `ingested_at` |

- **Citable fields** are `ts_utc`, `source_path` and every table-specific column above, plus one virtual field, `ts_local`. `ts_local` is `ts_utc` rendered in the case timezone with an offset (e.g. `2023-03-13T21:42:07-04:00`) and is compared after conversion to UTC.
- **Not citable:** `raw_json`, `tz_basis` and `ts_original` (citing them gives `UNKNOWN_FIELD`).

**Python API, used by the tools, the checker and the tests:**

```python
from tbx.store import Store
s = Store("path.db")            # creates the schema if missing
s.insert(table: str, row: dict) # row includes artifact_id + provenance; used by ingest and tests
s.get(artifact_id) -> dict | None   # full row incl. raw_json, plus "table": "<table>"
s.set_meta(key: str, value: str) # insert or replace a case_meta row; used by ingest and tests
s.case_tz -> str                # case_meta.device_tz (default "UTC" if missing); reflects the latest set_meta
s.known_contacts() -> set[str]  # distinct non-empty sender/recipient/contact_name values
```

## 5. Tools (exposed to the model as OpenAI function tools)

All time parameters are ISO strings, either UTC (`Z`) or with an offset. Every tool returns:

```json
{"rows": [ {"artifact_id": "...", "ts_utc": "...", "ts_local": "...", ...} ], "total": 123, "truncated": true}
```

| Tool | Parameters (all optional unless noted) |
|---|---|
| `search_messages` | `query`, `contact`, `number`, `app`, `start`, `end`, `limit=50` |
| `get_calls` | `query` (case-insensitive substring of `transcript`), `number`, `contact`, `direction`, `start`, `end`, `limit=50` |
| `get_locations` | `start`, `end`, `source`, `limit=50` |
| `find_media` | `kind`, `tag`, `start`, `end`, `limit=50` |
| `get_transcripts` | `query`, `media_id`, `start`, `end`, `limit=50` |
| `get_artifact` | `artifact_id` (required); returns the single full row (without `raw_json`) under `rows[0]` |

- **Row order:** rows are ordered by `ts_utc` ascending, with `artifact_id` as the tie-break. The order is deterministic.
- **`limit`:** capped at 200.
- **Long fields:** `body`/`text`/`transcript` longer than 300 characters are cut to 300, and the row gets `"<field>_truncated": true`. `get_artifact` returns the full text.
- **Phone matching:** number filters use the normalization in §7.

## 6. Agent output (the `submit_answer` tool call)

The final answer is the arguments of a `submit_answer` tool call, validated with pydantic. `Citation`, `Claim`, `Answer` and the checker's `CheckResult` live in **`tbx.models`**:

```python
class Citation(BaseModel):
    artifact_id: str
    field: str
    value: str                 # always a string, as copied from the tool row

class Claim(BaseModel):
    text: str
    time_start: str | None = None   # ISO, UTC or with offset
    time_end: str | None = None
    citations: list[Citation] = []  # empty is allowed here; the checker rejects it (NO_CITATIONS)

class Answer(BaseModel):
    claims: list[Claim]
    note: str | None = None         # e.g. "no evidence found in window"; never shown as a finding
```

- An answer with zero claims is valid. That is the correct output for negative questions.
- **Loop limits:** step cap 12, temperature 0.1.
- **Validation failure:** retry once with the pydantic error; on a second failure, raise `AgentError` (exit code 2).
- **Reasoning text:** strip `reasoning_content` and `<think>…</think>` from the model output before parsing.
- **Repair pass (`repair=1`):** after the check, rejected claims and their reasons go back to the model once, and it submits a new answer. The final report is the check of the second answer. The first round's rejections are kept in the trace and the result.

## 7. Citation checker (`tbx.checker`)

```python
from tbx.checker import check
result = check(answer: Answer, store: Store, window: Window | None) -> CheckResult
# CheckResult.verified: list[Claim]
# CheckResult.rejected: list[{"claim": Claim, "reasons": [{"code": str, "detail": str, "citation_index": int | None}]}]
```

- It is pure and deterministic: the same inputs give byte-identical `result.model_dump_json()`. It makes no network calls and no LLM calls.
- **Strict rule:** a claim with *any* reason is rejected, and every reason is listed.

**Normalization:**

| Kind | Applies to | Rule |
|---|---|---|
| timestamp | `ts_utc`, `ts_local` | Parse ISO (`Z`/offset) and convert to UTC. A value with no offset is read in the case timezone. Compare at second precision, or minute precision if the cited value has no seconds. |
| phone | `*_number`, `number` | Digits only. If both sides have ≥10 digits, compare the last 10; otherwise compare all digits exactly. |
| text | `body`, `text`, `transcript` | NFKC, casefold, whitespace collapsed and stripped. Match if equal, **or** the cited value is a substring of the stored value with ≥8 characters. |
| coordinate | `lat`, `lon` | abs diff ≤ 1e-4 |
| number | `score` | abs diff ≤ 1e-3 |
| measure | `duration_s`, `accuracy_m`, `size_bytes` | Parse both sides as numbers; match if abs diff ≤ 1e-6. So `"95"`, `"95.0"` and a stored `95` all match. |
| everything else | all other fields | NFKC, casefold, whitespace collapsed; exact match |

**Rules (codes are a stable API):**

| # | Code | Rule |
|---|---|---|
| R1 | `UNKNOWN_ARTIFACT` | The cited `artifact_id` is not in the store. That citation then gets no R2 or R3 reason. |
| R2 | `UNKNOWN_FIELD` | The field is not citable for that artifact's table. |
| R2 | `VALUE_MISMATCH` | The value does not match the stored value after normalization. |
| R3 | `OUTSIDE_WINDOW` | A window is set and a cited artifact's `ts_utc` is outside it, or is `NULL` (detail `"no timestamp"`). |
| R4 | `UNGROUNDED_ENTITY:<type>` | An entity found in the claim text is not grounded in that claim's cited values (see the entity table). |
| R5 | `TIME_RANGE_UNSUPPORTED` | `time_start` > `time_end`, or either one falls outside the claim's support range: from 60 s before the earliest cited artifact `ts_utc` to 60 s after the latest one, both ends inclusive. For calls, "the latest one" also counts `ts_utc + duration_s`. |
| R6 | `NO_CITATIONS` | The claim has no citations. |

R4 entity types:

| Type | What counts | Grounded when |
|---|---|---|
| `phone` | ≥7 digits with `+ ( ) - . space` | its digits match a cited value under the phone rule above (last 10 digits when both sides have ≥10), not by raw substring |
| `email` | | |
| `url` | | |
| `timestamp` | ISO date-time | |
| `date` | `YYYY-MM-DD`, or "Mar 14" / "March 14" | it matches a cited timestamp's date in the case timezone |
| `clock_time` | `9:42 PM`, `21:42` | it matches a cited timestamp's hour:minute in the case timezone |
| `coordinate` | a decimal pair with ≥3 decimals | it is within 1e-4 of a cited lat/lon |
| `contact` | any `store.known_contacts()` name of length ≥3 appearing as a whole word in the text (case-insensitive) | it appears as a whole word in a cited value, after text normalization |

Types with no "Grounded when" entry follow the general R4 rule. After text normalization (NFKC, casefold, whitespace collapsed), the entity must be **contained in** at least one of the claim's cited values; it does not have to equal one. For example, "Alice" is grounded by a cited transcript quote "it's Alice, meet me…".

**Extraction order:** entities are extracted in this order: `url`, `email`, `timestamp`, `date`, `clock_time`, `coordinate`, `phone`. A span matched by an earlier type is removed before the later types scan the text. So `2023-03-14T01:42:07Z` is one `timestamp` entity, not also a `date`, a `clock_time` and a `phone`; a `YYYY-MM-DD` date is never a phone; and a coordinate pair is never a phone. `contact` names are matched against the original text.

**Known limitation:** free-text place names and other prose are not checked. Verified claims still need an examiner's review.

## 8. Trace (JSONL, `runs/<run_id>.jsonl`)

Each line has the fields `{"ts": ISO, "run_id": str, "step": int, "type": str, "payload": {...}}`.

| Type | Payload |
|---|---|
| `run_start` | question, window, provider, model, repair |
| `model_request` | messages count, tools |
| `model_response` | content, tool_calls, usage {prompt_tokens, completion_tokens}, latency_ms |
| `tool_call` | name, args |
| `tool_result` | name, total, returned, truncated, artifact_ids |
| `validation_error` | error |
| `final_answer` | Answer |
| `check_result` | CheckResult |
| `repair` | rejected claims sent back |
| `run_end` | status (`ok`\|`error`), totals {tokens, latency_ms, est_cost_usd} |

The web UI streams these same events over SSE.

## 9. CLI (`tbx`)

```
tbx ingest <aleapp_report_dir> [--media-root DIR] [--db PATH]
tbx enrich [--voice] [--images] [--offline] [--db PATH]   # cached by sha256 + model (+ prompt_set + label)
tbx ask "<question>" [--from T] [--to T] [--provider P] [--model M] [--repair 0|1] [--json] [--db PATH]
tbx check <answer.json> [--from T] [--to T] [--db PATH]   # run the checker on a saved Answer (used by red-team tests)
```

- **Times:** `T` is `"YYYY-MM-DD HH:MM"` in the case timezone, or ISO with `Z`/an offset.
- **`tbx enrich`:**
  - **`--voice`:** every `media` row of kind `audio`, or `video` with an audio track, is transcoded to 24 kHz mono s16le WAV (ffmpeg from `imageio-ffmpeg`) and sent to Muse Voice Transcribe (`POST /asr/transcribe`, mode `ENDPOINTING`, **no `keywords`**). That gives one `transcripts` row per media: `model` = `muse-voice-transcribe-1.0`, `language` = `NULL`, `text` = `transcript`, `duration_s` = `audioDurationMs / 1000`, and `raw_json` holding `sessionId` and `turns`. Media with an empty transcript get no row.
  - **`--images`:** every `media` row of kind `image` that is JPEG or PNG is sent to SAM 3.1 (`POST /responses`, not streamed, `include_confidence: "true"`), one request per label of prompt set `v1` = `person`, `cell phone`, `laptop`, `car`, `license plate`, `sign`. Each box with `score` ≥ 0.5 becomes an `image_tags` row: `model` = `sam-3.1`, `prompt_set` = `v1`, and `box` = `[x0,y0,x1,y1]` in the pixels of the stored file after EXIF orientation is applied.
  - **Cache:** `data/sample/derived/enrich_cache.jsonl` is committed, so anyone can rebuild the store offline. It has one JSON object per line: `{"key": "<sha256>|<model>[|<prompt_set>|<label>]", "result": {...}}`. Voice results hold `transcript`, `audioDurationMs`, `turns` and `sessionId`; SAM results hold `boxes: [{x0,y0,x1,y1,score}]` and the image size. A cache hit makes no API call. `--offline` uses only the cache: misses are skipped and counted on stderr, and the exit code is 0.
  - Enriching the same store twice gives identical `trn_`/`tag_` ID sets.
- **Exit codes:**

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | usage or input error |
| 2 | agent output failed validation twice |
| 3 | provider/network error |

- **`--json` output:** `{"run_id", "verified": [...], "rejected": [...], "usage": {...}}`.
- **`tbx check` output:** the `CheckResult` as one JSON document on stdout, `{"verified": [...], "rejected": [...]}`, with the same shapes as `ask --json`. It exits 0 whenever the check ran, even if claims were rejected, and 1 for a missing or invalid answer file or bad arguments.

## 10. Web API (FastAPI)

The app lives in **`tbx.web`**. It exposes `create_app(db_path: str | None = None)`, which defaults to `TBX_DB`, and a module-level `app = create_app()`. Run it with `uvicorn tbx.web:app`.

| Method + path | Body / result |
|---|---|
| `GET /` | the single HTML page |
| `POST /api/ask` | Body `{question, from?, to?, provider?, model?, repair?}`; returns `{run_id}` at once and runs the agent in the background |
| `GET /api/runs` | the recorded runs, newest first: `[{run_id, question, status, started}]`. Read from `runs/*.jsonl`, so a recorded run can be replayed with no model call. |
| `GET /api/runs/{run_id}` | `{status, verified, rejected, usage, events}`, where `status` is `running`\|`ok`\|`error` and `events` holds the trace lines so far |
| `GET /api/runs/{run_id}/events` | optional (P2): an SSE stream of the same trace events |
| `GET /api/artifacts/{artifact_id}` | the full row, including `raw_json`, `source_path`, `source_row` |
| `GET /api/media/{artifact_id}` | the file bytes (images and audio) |
| `GET /api/media/{artifact_id}/tags` | `{"rows": [...]}`: that media's `image_tags` rows, with `box` as a JSON array |

## 11. Eval

`eval/questions.yaml`:

```yaml
- id: q01
  question: "Where was this phone between 9 and 11 pm, and who did it talk to?"
  window: {from: "2023-03-13 21:00", to: "2023-03-13 23:00"}   # case-local
  expected_entities: ["+15550101234", "Alice"]                   # phones as digits
  expected_artifact_ids: ["call_0a1b2c3d4e", "loc_..."]
  tags: [calls, locations]          # negative questions get tag "negative" and empty expectations
```

`eval/run.py --provider X --model Y [--repair 0|1] [--questions eval/questions.yaml] [--db PATH]` runs each question through the same agent code path as `tbx ask`, keeps every run's trace, and writes `eval/results/<provider>-<model>[-repair].json` (`/` in the model ID becomes `_`) with:

- **Per question:** `id`, `run_id`, `verified`, `rejected`, `fact_recall`, `artifact_recall`, `citation_precision`, `latency_s`, `tokens`, `est_cost_usd`.
- **Totals:**
  - **`fact_recall`:** the share of expected entities found (normalized as in §7) in the verified claims' text or cited values.
  - **`artifact_recall`:** the share of expected artifact IDs that appear among the verified citations.
  - **`citation_precision`:** verified claims / all claims, micro-averaged.
  - **`uncited_in_report`:** must be 0.
  - **`rejected_by_reason`:** counts per reason code.
  - **`abstention_accuracy`:** negative questions with 0 verified claims, as a share of all negative questions.
  - **`latency_p50_s`**, **`latency_p95_s`**, **`tokens`**, **`est_cost_usd`.**

`eval/report.py` renders the README results table from every file in `eval/results/`.

## Changelog

- **v1.2 (Nov 1, 2026):** additions for Day 2, from the G0/G1 gate run (`docs/GATES.md`).
  - §2: the order of `n` in `image_tags` row keys.
  - §3: new `tz_basis` value `offset_string` (EXIF times carry an explicit offset).
  - §4, §5, §7: `calls.direction` gains `voicemail`. 2,154 voicemail records had been stored as `missed`.
  - §4, §5, §7: new citable `calls.transcript` (49 real carrier voicemail transcriptions) under the text rule, and `get_calls(query=…)`.
  - §9: `tbx enrich` behavior, prompt set `v1`, the committed cache and `--offline`.
  - §10: the app lives in `tbx.web`; runs are replayable from `runs/*.jsonl`; `GET /api/runs`; SSE becomes optional; the shape of the tags response.
  - §11: `eval/run.py` flags and result fields.
- **v1.1 (Nov 1, 2026):** clarifications from the Day-0 spikes and the red-team suite. The IDs C1–C13 refer to `docs/SPIKES.md`.
  - §1: the Baseten model ID is `meta-models/muse-spark-open` (C10).
  - §2: `source_row` stores the `row_key` (C11).
  - §3, §6: `Window` lives in `tbx.timeutil` and takes UTC datetimes; the models live in `tbx.models` (C3).
  - §4: new `Store.set_meta()` (C4).
  - §7: entity extraction order (C1); an unknown artifact gets only `UNKNOWN_ARTIFACT` (C2); the R5 slack is inclusive (C5); new `measure` rule for `duration_s`/`accuracy_m`/`size_bytes` (C6); phone entities ground through the phone rule (C7); the general R4 rule means containment (C8).
  - §9: `tbx check` output and exit code (C13).
  - Still open: C9 (clock times quoted inside message text) and C12 (country codes under the last-10-digit phone rule).
