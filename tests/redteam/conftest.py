import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "support"))

import pytest  # noqa: E402
import tbxcase  # noqa: E402


@pytest.fixture(scope="session")
def tbx():
    return tbxcase.load_contract()


@pytest.fixture(scope="session")
def case_store(tbx, tmp_path_factory):
    return tbxcase.build_case_store(tbx, tmp_path_factory.mktemp("redteam") / "case.db")
