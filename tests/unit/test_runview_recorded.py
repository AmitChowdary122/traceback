"""tbx.runview on the real recorded runs in deploy/runs, against the store built from the committed sample (offline:
`tbx ingest` + `tbx enrich --offline`, no model, no network)."""
import copy
import json
import pathlib

import pytest

from tbx import cli, explain, runview
from tbx.checker import check
from tbx.models import Answer
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


# ------------------------------------------------------------------------------ F2 on real data (in memory only)
def with_claim(ev, k, **change):
    """A copy of a run's events whose answer has claim k changed (the recorded check is left as it was)."""
    ev = copy.deepcopy(ev)
    for e in ev:
        if e["type"] == "final_answer":
            claim = e["payload"]["claims"][k]
            for key, value in change.items():
                if key == "cite":
                    i, v = value
                    claim["citations"][i]["value"] = v
                else:
                    claim[key] = value
    return ev


def rechecked_rejections(store, ev):
    """[(claim text, its sentences)] for the claims the checker rejects in the (changed) answer, under its window."""
    answer = Answer.model_validate({"claims": next(e for e in ev if e["type"] == "final_answer")["payload"]["claims"]})
    res = check(answer, store, runview.recorded_window(ev)).model_dump(mode="json")
    spans = dict(zip((json.dumps(c, sort_keys=True) for c in answer.model_dump(mode="json")["claims"]), res["entities"]))
    return [(r["claim"]["text"], explain.sentences(r, spans[json.dumps(r["claim"], sort_keys=True)], store, store.case_tz))
            for r in res["rejected"]]


def test_a_wrong_time_in_the_text_names_the_records_true_time(store):
    ev = events("ae5039")
    text = next(e for e in ev if e["type"] == "final_answer")["payload"]["claims"][0]["text"]
    assert text.startswith("At 9:06 PM")
    got = rechecked_rejections(store, with_claim(ev, 0, text=text.replace("9:06 PM", "9:36 PM")))
    assert got == [(text.replace("9:06 PM", "9:36 PM"), ["'9:36 PM' doesn't match the cited record's time (9:06 PM)."])]


def test_a_wrong_cited_time_is_shown_case_local_and_the_text_is_not_backed(store):
    ev = events("ae5039")
    text = next(e for e in ev if e["type"] == "final_answer")["payload"]["claims"][0]["text"]
    got = rechecked_rejections(store, with_claim(ev, 0, text=text.replace("9:06 PM", "9:36 PM"),
                                                 cite=(0, "2024-06-27T21:36:18-04:00")))
    assert got[0][1] == [
        "The cited time (ts_local) is Thu Jun 27, 2024 9:36:18 PM (UTC-04:00); the record's time is "
        "Thu Jun 27, 2024 9:06:18 PM (UTC-04:00).",
        "'June 27' isn't backed by a correctly cited time (the cited record's date is Thu Jun 27, 2024).",
        "'9:36 PM' isn't backed by a correctly cited time (the cited record's time is 9:06 PM)."]


# ------------------------------------------------------------------------------ B3: a re-check that differs (real trace)
# Eval 1, q07 on Meta (docs/gates/eval1/runs_meta/20261103T113123Z-eb2773.jsonl, unedited): the recorded check rejected
# a word-for-word voicemail quote for the times inside it ("at 8:00", "9:30"); under contract v1.4 it verifies.
# It is a test fixture only, never a demo run in deploy/runs.
Q07 = pathlib.Path(__file__).parent / "fixtures" / "eval1_q07_20261103T113123Z-eb2773.jsonl"


def q07_events():
    return [json.loads(x) for x in Q07.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_q07_fixture_is_the_unedited_eval_trace():
    assert Q07.read_bytes() == (REPO / "docs/gates/eval1/runs_meta/20261103T113123Z-eb2773.jsonl").read_bytes()
    assert not (RUNS / "20261103T113123Z-eb2773.jsonl").exists()


def test_q07_recheck_differs_and_shows_the_rechecked_verdicts(store):
    ev = q07_events()
    rec = [e["payload"] for e in ev if e["type"] == "check_result"][-1]
    assert (len(rec["verified"]), len(rec["rejected"])) == (3, 1)
    out = runview.reverify(ev, store)
    rc = out["recheck"]
    assert rc["status"] == "differs" and rc["shown"] == "rechecked"
    (rnd,) = rc["rounds"]
    assert (rnd["recorded"], rnd["rechecked"]) == ("3 verified, 1 rejected", "4 verified, 0 rejected")
    quote = rec["rejected"][0]["claim"]
    k = next(i for i, c in enumerate(next(e for e in ev if e["type"] == "final_answer")["payload"]["claims"])
             if c["text"] == quote["text"])
    assert any(f"Claim {k + 1} was recorded as rejected; the re-check verifies it." == c for c in rnd["changes"])
    recorded = {v["claim"]: (v["verdict"], v["codes"]) for v in rnd["recorded_verdicts"]}
    assert recorded[k + 1] == ("rejected", ["UNGROUNDED_ENTITY:clock_time", "UNGROUNDED_ENTITY:clock_time"])
    assert all(v == ("verified", []) for n, v in recorded.items() if n != k + 1)
    assert f"claim {k + 1} rejected (UNGROUNDED_ENTITY:clock_time, UNGROUNDED_ENTITY:clock_time)" in rc["message"]
    # what the page shows: the re-check's four verified claims, in answer order, with spans aligned to them
    assert out["verdicts"]["rejected"] == [] and len(out["verdicts"]["verified"]) == 4
    assert [a["index"] for a in out["annotations"]["verified"]] == [0, 1, 2, 3]
    spans = out["annotations"]["verified"][k]["entities"]
    text = quote["text"]
    assert {(e["text"], e["grounded"]) for e in spans if e["type"] == "clock_time"} >= {("8:00", True), ("9:30", True)}
    assert_well_formed([(c["text"], a["entities"]) for c, a in zip(out["verdicts"]["verified"],
                                                                   out["annotations"]["verified"])])
    assert text.index("8:00") > text.index('"')     # the exempt times are inside the quote


def test_q07_run_view_and_list_use_the_rechecked_verdicts(store, tmp_path):
    from fastapi.testclient import TestClient

    from tbx.web import create_app

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20261103T113123Z-eb2773.jsonl").write_bytes(Q07.read_bytes())
    client = TestClient(create_app(db_path=store.path, runs_dir=str(runs)))
    view = client.get("/api/runs/20261103T113123Z-eb2773").json()
    assert view["recheck"]["status"] == "differs"
    assert (len(view["verified"]), len(view["rejected"])) == (4, 0)
    assert len(view["annotations"]["verified"]) == 4
    (row,) = client.get("/api/runs").json()
    assert (row["recheck"], row["verified"], row["rejected"]) == ("differs", 4, 0)
