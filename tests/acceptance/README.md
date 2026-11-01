# Acceptance tests

These run offline and need no keys. To run everything except the P1 features:

```bash
pytest tests/acceptance -m "not p1"
```

To run all of it at a gate, with skips counted as failures:

```bash
TBX_STRICT_TESTS=1 pytest tests/acceptance
```

| File | What it covers | Needs |
|---|---|---|
| `test_acceptance_hygiene.py` | No key values in the repo, spike fixtures redacted, no `-contributor` models, `.env` not tracked | nothing (runs today) |
| `test_acceptance_fakeprovider.py` | Self-test of the local OpenAI-compatible fake provider | nothing (runs today) |
| `test_acceptance_cli_agent.py` | `tbx ask`/`tbx check` end to end. Covers exit codes 0, 1, 2 and 3, `tool_choice`, `max_tokens`, the retry nudge, the step cap, Baseten routing, tool-result shape, the trace JSONL, key canaries, and the repair pass (P1) | `tbx` CLI + `tbx.store` |
| `test_acceptance_ingest.py` | Two ingests give identical IDs; ID format and formula; provenance; `ts_utc`/`tz_basis`; enums; `case_meta` | `tbx` CLI + `data/sample/` (or `TBX_SAMPLE_REPORT`) |
| `test_acceptance_enrich.py` (P1) | `tbx enrich --voice --images --offline` from the committed cache: both MMS soundtracks transcribed, SAM prompt set `v1`, boxes inside the image, `n` ordering, identical IDs when enriched twice | `tbx` CLI + `data/sample/` + `data/sample/derived/enrich_cache.jsonl` |
| `test_acceptance_web_eval.py` | Web API in `tbx.web` (P1), including replay of `runs/*.jsonl`; `eval/questions.yaml` schema; eval results with `uncited_in_report == 0` | FastAPI app / eval files |

How the CLI tests run:

- The fake provider (`tests/support/fakeprovider.py`) replays the response shapes recorded in spikes S1 and S4.
- Like the real Meta API, it returns HTTP 400 for any `tool_choice` other than `"auto"`.
- The `tbx` subprocess gets **canary** keys, never the real ones. Both base URLs point at a closed localhost port unless a test sets them, so no test can reach a real API.
- The CLI runs with `cwd` set to a temp dir, and its trace is read from `runs/<run_id>.jsonl` there.
