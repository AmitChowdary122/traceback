# Red-team suite for the citation checker

These are claim cases checked against CONTRACTS.md v1.4 §7. Each one is a single claim, run through `tbx.checker.check` against a small case store. The store comes from `tests/support/tbxcase.py`: one evening, case timezone `America/New_York`. Mar 13, 2023 is the day after the DST change.

```bash
pytest tests/redteam -v
```

At a gate, run it in strict mode, where a skip counts as a failure:

```bash
TBX_STRICT_TESTS=1 pytest tests/redteam
```

- `redteam_cases.py` has the cases.
  - `ACCEPT` (A01–A26, including the five phone-format variants A04a–e) must verify.
  - `REJECT` (R01–R41) must reject with **exactly** the listed reason codes. Where a case lists `citation_index` values, those must match too.
- `REAL` cases are verbatim `submit_answer` output from muse-spark-1.3 and muse-spark-open, recorded in spikes S1 and S4 (`tests/fixtures/spikes/`).
- Notes tagged `[Cn]` pin a rule added in CONTRACTS.md v1.1 (see its changelog and `docs/SPIKES.md` → "Contract clarifications").
- `tbxcase.load_contract()` imports the modules the contract names: `tbx.store`, `tbx.checker`, `tbx.models` and `tbx.timeutil`.
- The case store is written only through `Store(path)`, `Store.set_meta()` and `Store.insert()`.
- `test_redteam_entities.py` (v1.3): `CheckResult.entities`. Spans must line up with the claim text, `grounded=false` must match the `UNGROUNDED_ENTITY` reasons exactly, and specific spans are pinned on real model outputs (R27's two red window timestamps; A01 with no double extraction).
