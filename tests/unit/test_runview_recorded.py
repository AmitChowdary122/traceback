"""tbx.runview on the real recorded runs in deploy/runs, against the store built from the committed sample (offline:
`tbx ingest` + `tbx enrich --offline`, no model, no network)."""
import copy
import json
import pathlib

import pytest

from tbx import cli, runview
from tbx.store import Store

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "deploy" / "runs"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db = tmp_path_factory.mktemp("sample") / "tbx.db"
    assert cli.main(["ingest", str(REPO / "data/sample/aleapp"), "--media-root", str(REPO / "data/sample/media"),
                     "--db", str(db)]) == 0
    assert cli.main(["enrich", "--offline", "--db", str(db)]) == 0
    s = Store(str(db))
    yield s
    s.close()


def events(suffix):
    (path,) = RUNS.glob(f"*-{suffix}.jsonl")
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def verified(store, suffix):
    """[(claim text, its annotation spans)] for the run's verified claims, as the page gets them."""
    ev = events(suffix)
    out = runview.reverify(ev, store)
    assert out["recheck"]["status"] == "match", out["recheck"]
    claims = [e["payload"] for e in ev if e["type"] == "check_result"][-1]["verified"]
    return [(c["text"], a["entities"]) for c, a in zip(claims, out["annotations"]["verified"])]


def green(spans):
    return {i for e in spans if e["grounded"] for i in range(e["start"], e["end"])}


def assert_highlighted(claims, sub):
    hits = [(text, spans) for text, spans in claims if sub in text]
    assert hits, f"no claim contains {sub!r}"
    for text, spans in hits:
        at = text.index(sub)
        missing = set(range(at, at + len(sub))) - green(spans)
        assert not missing, f"{sub!r} is not fully green in {text!r}: {spans}"


def assert_well_formed(claims):
    for text, spans in claims:
        r4 = [e for e in spans if e["type"] != "cited_value"]
        for e in spans:
            assert text[e["start"]:e["end"]] == e["text"] and e["text"] == e["text"].strip()
        for c in (e for e in spans if e["type"] == "cited_value"):
            assert c["grounded"] is True and isinstance(c["citation_index"], int) and c["field"]
            assert not any(c["start"] < e["end"] and e["start"] < c["end"] for e in r4), (text, c)   # R4 wins


def test_quoted_message_text_is_highlighted_in_ae5039(store):
    claims = verified(store, "ae5039")
    assert_highlighted(claims, "Team DLCC: President Biden and Trump just left the stage.")
    assert_well_formed(claims)
    (text, spans) = next(c for c in claims if "Team DLCC" in c[0])
    assert [(e["type"], e["field"]) for e in spans if e["type"] == "cited_value"] == [("cited_value", "body")]
    assert "The message said" not in "".join(e["text"] for e in spans)        # the model's own words stay plain


def test_a_cited_label_is_highlighted_in_797bbf(store):
    claims = verified(store, "797bbf")
    assert_highlighted(claims, "cell phone")
    for label in ("person", "laptop", "car", "license plate", "sign"):
        assert_highlighted(claims, label)
    assert_well_formed(claims)
    assert all("shows" not in e["text"] for _, spans in claims for e in spans)


def test_cited_coordinates_written_apart_are_highlighted_in_3f05db(store):
    claims = verified(store, "3f05db")
    assert_highlighted(claims, "35.6596081")
    assert_highlighted(claims, "-78.8730535")
    assert_well_formed(claims)
    text, spans = claims[0]
    assert "latitude" not in "".join(e["text"] for e in spans)
    assert {e["field"] for e in spans if e["type"] == "cited_value"} == {"lat", "lon"}


def test_every_recorded_run_still_rechecks_the_same_and_its_spans_are_well_formed(store):
    for path in sorted(RUNS.glob("*.jsonl")):
        claims = verified(store, path.stem.rsplit("-", 1)[1])
        assert_well_formed(claims)
