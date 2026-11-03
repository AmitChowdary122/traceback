"""Acceptance: labeled fault injection on REAL model answers (offline, no API).

Every recorded answer in deploy/runs/ is real, unedited model output. Its recorded verdicts must hold: re-checked
against the store built from the committed sample, the claims the recording verified still verify, and the claims
it rejected (a real rejection, e.g. Eval 2's q10, where the model named senders without citing them) are still
rejected with the same reason codes. Each claim that verified is then broken in one controlled way at a time and
re-checked:

- a cited value is altered: a timestamp +1 minute, a phone digit, lat/lon +0.001, a quote cut to 5 chars,
  a label/name/direction typo, a measure +2;
- the first citation points to a near-miss artifact ID, or cites raw_json;
- the claim text gets a wrong clock time, a wrong day of the month, or an invented phone number;
- the claim gets a time range on a day none of its rows covers.

Every faulty copy must be rejected. This is how the README's fault-injection catch rate is measured. The faults
are labeled as injected; they are never shown as model errors, and a claim the recording already rejected is not
broken further (it is counted as a real rejection instead).
"""
from __future__ import annotations

import copy
import json
import re
import subprocess
from datetime import datetime

import pytest
import tbxcase

REPO = tbxcase.REPO
SAMPLE = REPO / "data" / "sample"
RUNS = sorted((REPO / "deploy" / "runs").glob("*.jsonl"))
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _perturb_value(field: str, v: str) -> str:
    if field in ("ts_utc", "ts_local") and re.search(r"T\d\d:\d\d", v):
        m = re.search(r"T(\d\d):(\d\d)", v)
        return v[:m.start(2)] + f"{(int(m.group(2)) + 1) % 60:02d}" + v[m.end(2):]
    if field == "number" or field.endswith("_number"):
        return v[:-1] + ("1" if v[-1] != "1" else "2") if re.search(r"\d", v) else v + "9"
    if field in ("lat", "lon"):
        return str(round(float(v) + 0.001, 7))
    if field in ("body", "text", "transcript"):
        return v[:5]
    if field in ("duration_s", "accuracy_m", "size_bytes", "score"):
        return str(float(v) + 2)
    return v + "x"


def faults(claim: dict) -> list[tuple[str, dict]]:
    out = []
    for k, cit in enumerate(claim["citations"]):
        a = copy.deepcopy(claim)
        a["citations"][k]["value"] = _perturb_value(cit["field"], cit["value"])
        out.append((f"value[{k}]:{cit['field']}", a))
    a = copy.deepcopy(claim)
    aid = a["citations"][0]["artifact_id"]
    a["citations"][0]["artifact_id"] = aid[:-1] + ("0" if aid[-1] != "0" else "1")
    out.append(("near-miss artifact id", a))
    a = copy.deepcopy(claim)
    a["citations"][0]["field"] = "raw_json"
    out.append(("uncitable field", a))
    t = claim["text"]
    m = re.search(r"\b(\d{1,2}):(\d\d) ?([AP]M)", t)
    if m:
        a = copy.deepcopy(claim)
        a["text"] = t[:m.start(2)] + f"{(int(m.group(2)) + 7) % 60:02d}" + t[m.end(2):]
        out.append(("wrong clock time in text", a))
    a = copy.deepcopy(claim)
    new = re.sub(rf"({MONTHS}) (\d{{1,2}})", lambda mm: f"{mm.group(1)} {int(mm.group(2)) % 28 + 1}", t, count=1)
    if new != t:
        a["text"] = new
        out.append(("wrong day in text", a))
    a = copy.deepcopy(claim)
    a["text"] = t.rstrip(".") + ", then called (919) 555-0142."
    out.append(("invented phone in text", a))
    a = copy.deepcopy(claim)
    a["time_start"], a["time_end"] = "2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"
    out.append(("unsupported time range", a))
    return out


@pytest.fixture(scope="module")
def real_store(tbx, tbx_cli, tmp_path_factory):
    if not RUNS or not (SAMPLE / "aleapp").is_dir():
        tbxcase.unavailable("needs deploy/runs/*.jsonl and data/sample/ (build branch)")
    db = tmp_path_factory.mktemp("fault") / "real.db"
    for args in (["ingest", str(SAMPLE / "aleapp"), "--media-root", str(SAMPLE / "media"), "--db", str(db)],
                 ["enrich", "--voice", "--images", "--offline", "--db", str(db)]):
        proc = subprocess.run(tbx_cli + args, capture_output=True, text=True, timeout=900, cwd=db.parent)
        assert proc.returncode == 0, proc.stderr[-2000:]
    return tbx.Store(str(db))


def _recorded(path):
    """-> (window, last final answer, last recorded check result)."""
    ev = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    start = next(e["payload"] for e in ev if e["type"] == "run_start")
    answers = [e["payload"] for e in ev if e["type"] == "final_answer"]
    checks = [e["payload"] for e in ev if e["type"] == "check_result"]
    return start.get("window"), answers[-1], (checks[-1] if checks else None)


def _key(tbx, claim: dict) -> str:
    """A claim's canonical JSON (as the Claim model dumps it), to match recorded and re-checked claims."""
    dumped = tbx.Answer.model_validate({"claims": [claim]}).model_dump(mode="json")["claims"][0]
    return json.dumps(dumped, sort_keys=True)


def _rejected_codes(tbx, rejected: list[dict]) -> dict[str, list[str]]:
    """{claim key: sorted reason codes (duplicates kept)} for a CheckResult's rejected list."""
    return {_key(tbx, r["claim"]): sorted(x["code"] for x in r["reasons"]) for r in rejected}


def _window(tbx, w):
    if not w:
        return None
    dt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    return tbx.Window(dt(w["start_utc"]), dt(w["end_utc"]))


@pytest.mark.p1
def test_recorded_verdicts_hold_and_every_injected_fault_is_rejected(tbx, real_store):
    caught, total, missed, real_rejections = 0, 0, [], 0
    for path in RUNS:
        w, answer, recorded = _recorded(path)
        assert recorded is not None, f"{path.name}: no recorded check_result"
        window = _window(tbx, w)
        claims = [{k: v for k, v in c.items()} for c in answer["claims"]]
        now = tbx.check(tbx.Answer.model_validate({"claims": claims}), real_store, window).model_dump(mode="json")
        was = _rejected_codes(tbx, recorded["rejected"])
        assert _rejected_codes(tbx, now["rejected"]) == was, \
            f"{path.name}: the re-check differs from the recorded verdicts"
        for ci, claim in enumerate(claims):
            if _key(tbx, claim) in was:
                real_rejections += 1        # rejected in the recording too: a real rejection, not a fault target
                continue
            for label, bad in faults(claim):
                total += 1
                res = tbx.check(tbx.Answer.model_validate({"claims": [bad]}), real_store, window)
                if res.rejected:
                    caught += 1
                else:
                    missed.append(f"{path.name} claim {ci}: {label}: {bad['text'][:90]}")
    print(f"\nfault injection on real answers: {caught}/{total} rejected")
    print(f"real rejections in the recorded runs (reproduced, not injected): {real_rejections} claims")
    assert total >= 100, f"too few injected faults ({total}); are the recorded runs present?"
    assert not missed, "faults that passed the checker:\n" + "\n".join(missed[:20])
