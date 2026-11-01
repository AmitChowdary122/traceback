# Red-team suite for the citation checker

These are claim cases checked against CONTRACTS.md §7. Each one is a single claim, run through `tbx.checker.check` against a small case store. The store comes from `tests/support/tbxcase.py`: one evening, case timezone `America/New_York`. Mar 13, 2023 is the day after the DST change.

```bash
pytest tests/redteam -v
```

At a gate, run it in strict mode, where a skip counts as a failure:

```bash
TBX_STRICT_TESTS=1 pytest tests/redteam
```

- `redteam_cases.py` has the cases.
  - `ACCEPT` (A01–A21, including the five phone-format variants A04a–e) must verify.
  - `REJECT` (R01–R36) must reject with **exactly** the listed reason codes. Where a case lists `citation_index` values, those must match too.
- `REAL` cases are verbatim `submit_answer` output from muse-spark-1.3 and muse-spark-open, recorded in spikes S1 and S4 (`tests/fixtures/spikes/`).
- Notes tagged `[Cn]` rely on a proposed contract clarification. See `docs/SPIKES.md` → "Proposed contract clarifications".
- The contract API is found by `tbxcase.load_contract()`, which searches the candidate modules listed there. If your module names differ, add them there or tell the testing session; don't edit the cases.
- The case store is written only through `Store(path)` and `Store.insert()`, except `case_meta`, which is written with plain SQL (C4).
