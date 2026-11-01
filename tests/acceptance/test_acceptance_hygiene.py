"""Acceptance: repo and fixture hygiene. Runs today, with or without src/.

- No API key value (from this environment) appears in any file in the repo.
- Every recorded spike fixture has its Authorization header redacted.
- No fixture request ever used a `-contributor` model (CONTRACTS.md section 1).
- .env files are not tracked.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest
import tbxcase

REPO = tbxcase.REPO
SPIKES = tbxcase.SPIKES


def repo_files() -> list:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO, capture_output=True, text=True, check=True)
    return [REPO / p for p in out.stdout.splitlines() if (REPO / p).is_file()]


def test_no_key_values_in_repo():
    keys = {k: v for k, v in os.environ.items() if k.endswith("_API_KEY") and len(v) >= 12}
    if not keys:
        pytest.skip("no *_API_KEY values in this environment to scan for")
    leaks = []
    for f in repo_files():
        data = f.read_bytes()
        for name, value in keys.items():
            for needle in {value, value[:16], value[-16:]}:
                if needle.encode() in data:
                    leaks.append(f"{f.relative_to(REPO)} contains part of {name}")
    assert not leaks, leaks  # names only, never the value


def fixtures() -> list:
    return sorted(SPIKES.glob("*.json"))


def test_spike_fixtures_exist_for_s1_to_s5():
    names = {f.stem.split("_")[0] for f in fixtures()}
    assert {"s1", "s2", "s3", "s4", "s5"} <= names, names


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_spike_fixture_is_sanitized(path):
    rec = json.loads(path.read_text())
    headers = rec["request"].get("headers") or {}
    auth = headers.get("Authorization")
    assert auth in (None, "Bearer <REDACTED>"), f"{path.name}: Authorization not redacted"
    blob = json.dumps(rec)
    assert "Bearer " not in blob.replace("Bearer <REDACTED>", ""), f"{path.name}: unredacted bearer token"
    for h in (rec.get("response") or {}).get("headers") or {}:
        assert h.lower() not in ("set-cookie", "authorization"), f"{path.name}: sensitive response header {h}"


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_no_contributor_models_in_fixtures(path):
    body = json.loads(path.read_text())["request"].get("json") or {}
    assert "contributor" not in str(body.get("model", "")), path.name


def test_env_files_not_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    bad = [p for p in tracked if os.path.basename(p) == ".env" or (os.path.basename(p).startswith(".env.") and not p.endswith(".env.example"))]
    assert not bad, bad


def test_env_example_lists_names_only():
    ex = REPO / ".env.example"
    if not ex.exists():
        pytest.skip(".env.example not created yet (builder scaffold)")
    for line in ex.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip().endswith("_API_KEY"):
            assert value.strip() in ("", '""', "''"), f".env.example has a value for {name.strip()}"
