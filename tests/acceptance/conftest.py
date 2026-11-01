import importlib.util
import os
import pathlib
import secrets
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "support"))

import pytest  # noqa: E402
import tbxcase  # noqa: E402
from fakeprovider import closed_port  # noqa: E402

REPO = tbxcase.REPO


def pytest_configure(config):
    config.addinivalue_line("markers", "p1: P1 feature (repair pass, web UI, eval); deselect with -m 'not p1' before Day 2")


@pytest.fixture(scope="session")
def tbx():
    return tbxcase.load_contract()


@pytest.fixture(scope="session")
def tbx_cli() -> list[str]:
    """argv prefix for the `tbx` CLI (CONTRACTS.md section 9)."""
    exe = shutil.which("tbx")
    if exe:
        return [exe]
    for mod in ("tbx.__main__", "tbx.cli"):
        try:
            if importlib.util.find_spec(mod) is not None:
                return [sys.executable, "-m", mod.removesuffix(".__main__")]
        except ImportError:
            continue
    tbxcase.unavailable("no `tbx` executable on PATH and no tbx.__main__/tbx.cli module")


@pytest.fixture
def case_db(tbx, tmp_path) -> pathlib.Path:
    path = tmp_path / "case.db"
    tbxcase.build_case_store(tbx, path)
    return path


@pytest.fixture
def canaries() -> dict:
    """Fake keys. The real ones are never put in a test subprocess's environment."""
    return {"META_API_KEY": f"tbx-canary-meta-{secrets.token_hex(8)}",
            "BASETEN_API_KEY": f"tbx-canary-baseten-{secrets.token_hex(8)}"}


@pytest.fixture
def cli_env(canaries, case_db) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    no_proxy = ",".join(filter(None, [env.get("NO_PROXY", ""), "127.0.0.1,localhost"]))
    dead = f"http://127.0.0.1:{closed_port()}/v1"  # tests override this; nothing may reach a real API
    env.update(canaries, TBX_PROVIDER="meta", TBX_MODEL="muse-spark-1.3", TBX_DB=str(case_db),
               META_BASE_URL=dead, BASETEN_BASE_URL=dead,
               NO_PROXY=no_proxy, no_proxy=no_proxy, PYTHONUNBUFFERED="1")
    for k in ("HTTP_PROXY", "http_proxy"):  # the fake provider is plain http on localhost
        env.pop(k, None)
    return env


@pytest.fixture
def run_cli(tbx_cli, tmp_path):
    """Run `tbx ...` with cwd=tmp_path (so runs/<run_id>.jsonl lands there)."""
    def run(args: list[str], env: dict, timeout: float = 180) -> subprocess.CompletedProcess:
        return subprocess.run(tbx_cli + args, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=timeout)
    return run

