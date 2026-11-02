"""tbx export-static: a replay-only static copy of the web app (no server, no keys, no model calls)."""
import json

import pytest
from fastapi.testclient import TestClient

from tbx import cli, web
from tbx.export_static import ExportError, export_static
from tbx.store import Store, artifact_id

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
MED = artifact_id("media", "DCIM/a.png", "1")
VID = artifact_id("media", "cache/v.bin", "1")
TAG = artifact_id("image_tags", "DCIM/a.png", f"{MED}|sam-3.1|car|0")
TAG2 = artifact_id("image_tags", "DCIM/a.png", f"{MED}|sam-3.1|sign|0")
TRN = artifact_id("transcripts", "cache/v.bin", f"{VID}|muse-voice-transcribe-1.0")
UNKNOWN = "call_0000000000"
RUN = "20261101T000000Z-abc123"


def _claim(text, *cites):
    return {"text": text, "time_start": None, "time_end": None,
            "citations": [{"artifact_id": a, "field": f, "value": v} for a, f, v in cites]}


@pytest.fixture
def case(tmp_path, monkeypatch):
    from PIL import Image
    for k in ("META_API_KEY", "BASETEN_API_KEY", "TBX_FEATURED_RUN", "TBX_LIVE_PROVIDERS"):
        monkeypatch.delenv(k, raising=False)
    media = tmp_path / "media"
    (media / "DCIM").mkdir(parents=True)
    (media / "cache").mkdir()
    Image.new("RGB", (100, 50), "white").save(media / "DCIM" / "a.png")
    (media / "cache" / "v.bin").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    (media / "DCIM" / "uncited.png").write_bytes(b"not exported")
    db = tmp_path / "c.db"
    s = Store(str(db))
    s.set_meta("device_tz", NY)
    s.set_meta("media_root", str(media))
    prov = {"ts_original": "x", "tz_basis": "utc_string", "raw_json": json.dumps({"Partner": "9195794674"})}
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-07-28T01:40:00Z", **prov, "app": "Phone", "direction": "outgoing",
                       "number": "9195794674", "contact_name": "Liz Dehner", "duration_s": 121})
    s.insert("media", {"artifact_id": MED, "source_path": "DCIM/a.png", "source_row": "1",
                       "ts_utc": "2024-07-24T16:05:00Z", **prov, "app": "Camera", "kind": "image",
                       "file_path": "DCIM/a.png", "mime_type": "image/png", "sha256": "0" * 64, "size_bytes": 1})
    s.insert("media", {"artifact_id": VID, "source_path": "cache/v.bin", "source_row": "1",
                       "ts_utc": "2024-02-09T21:24:00Z", **prov, "app": "Google Messages", "kind": "video",
                       "file_path": "cache/v.bin", "mime_type": "video/mp4", "sha256": "1" * 64, "size_bytes": 40})
    for aid, label, n, box in ((TAG, "car", 0, "[1, 2, 30, 40]"), (TAG2, "sign", 0, "[50, 5, 90, 20]")):
        s.insert("image_tags", {"artifact_id": aid, "source_path": "DCIM/a.png", "source_row": f"{MED}|sam-3.1|{label}|{n}",
                                "ts_utc": "2024-07-24T16:05:00Z", **prov, "media_id": MED, "model": "sam-3.1",
                                "label": label, "score": 0.9, "box": box, "prompt_set": "v1"})
    s.insert("transcripts", {"artifact_id": TRN, "source_path": "cache/v.bin",
                             "source_row": f"{VID}|muse-voice-transcribe-1.0", "ts_utc": "2024-02-09T21:24:00Z", **prov,
                             "media_id": VID, "model": "muse-voice-transcribe-1.0", "language": None,
                             "text": "running for Attorney General", "duration_s": 30.0})
    s.close()
    runs = tmp_path / "runs"
    runs.mkdir()
    ok = [_claim("A photo shows a car.", (TAG, "label", "car")),
          _claim("The video said \"running for Attorney General\".", (TRN, "text", "running for Attorney General"))]
    bad = {"claim": _claim("Bob called.", (UNKNOWN, "contact_name", "Bob")),
           "reasons": [{"code": "UNKNOWN_ARTIFACT", "detail": "not in store", "citation_index": 0}]}
    events = [("run_start", {"question": "What do the photos show?", "window": None, "provider": "baseten",
                             "model": "meta-models/muse-spark-open", "repair": 0, "case_tz": NY}),
              ("final_answer", {"claims": ok + [bad["claim"]], "note": None}),
              ("check_result", {"verified": ok, "rejected": [bad]}),
              ("run_end", {"status": "ok", "totals": {"tokens": 10, "latency_ms": 5, "est_cost_usd": 0.01}})]
    (runs / f"{RUN}.jsonl").write_text("".join(
        json.dumps({"ts": "2026-11-01T00:00:00Z", "run_id": RUN, "step": i, "type": t, "payload": p}) + "\n"
        for i, (t, p) in enumerate(events)))
    return tmp_path, db, runs, media


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_every_file_the_page_reads(case):
    tmp, db, runs, media = case
    out = tmp / "site"
    summary = export_static(str(out), str(db), str(runs))
    assert summary["runs"] == 1 and summary["unknown_cited_ids"] == [UNKNOWN] and summary["media_files"] == 2

    server = TestClient(web.create_app(str(db), runs_dir=str(runs)))
    assert (out / "api/runs.json").read_bytes() == server.get("/api/runs").content
    assert (out / f"api/runs/{RUN}.json").read_bytes() == server.get(f"/api/runs/{RUN}").content
    for aid in (TAG, TRN, MED, VID):                 # cited rows, and the media the drawer opens behind them
        assert (out / f"api/artifacts/{aid}.json").read_bytes() == server.get(f"/api/artifacts/{aid}").content
    assert not (out / f"api/artifacts/{UNKNOWN}.json").exists()     # stays a 404, as on the server
    assert not (out / f"api/artifacts/{CALL}.json").exists()        # not cited
    tags = _load(out / f"api/media/{MED}/tags.json")["rows"]
    assert {t["artifact_id"]: t["box"] for t in tags} == {TAG: [1, 2, 30, 40], TAG2: [50, 5, 90, 20]}
    assert _load(out / f"api/media/{VID}/transcripts.json")["rows"][0]["text"] == "running for Attorney General"
    assert (out / f"media/{MED}.png").read_bytes() == (media / "DCIM/a.png").read_bytes()
    assert (out / f"media/{VID}.mp4").read_bytes() == (media / "cache/v.bin").read_bytes()  # .bin -> .mp4 by MIME
    assert not any("uncited" in p.name for p in out.rglob("*"))
    assert (out / ".nojekyll").is_file()


def test_exported_page_is_in_static_mode_with_asking_off(case):
    tmp, db, runs, _ = case
    out = tmp / "site"
    export_static(str(out), str(db), str(runs), featured=RUN)
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "<!--tbx-static-->" not in page
    start = page.index('<script id="tbx-static" type="application/json">') + len('<script id="tbx-static" type="application/json">')
    manifest = json.loads(page[start:page.index("</script>", start)])
    assert manifest["media"] == {MED: f"media/{MED}.png", VID: f"media/{VID}.mp4"}
    case_json = _load(out / "api/case.json")
    assert case_json["static"] is True and case_json["ask_limit_per_hour"] == 0 and case_json["live_providers"] == []
    assert set(case_json["providers"].values()) == {False} and case_json["featured_run"] == RUN
    assert case_json["counts"]["image_tags"] == 2


def test_featured_run_must_be_exported(case):
    tmp, db, runs, _ = case
    export_static(str(tmp / "site"), str(db), str(runs), featured="not-a-run")
    assert _load(tmp / "site/api/case.json")["featured_run"] is None


def test_export_replaces_its_own_output_but_never_a_foreign_directory(case):
    tmp, db, runs, _ = case
    out = tmp / "site"
    export_static(str(out), str(db), str(runs))
    (out / "api/runs/stale.json").write_text("{}")
    export_static(str(out), str(db), str(runs))
    assert not (out / "api/runs/stale.json").exists()
    foreign = tmp / "mine"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("keep me")
    with pytest.raises(ExportError, match="not empty"):
        export_static(str(foreign), str(db), str(runs))
    assert (foreign / "notes.txt").read_text() == "keep me"


def test_export_refuses_missing_inputs(case, tmp_path):
    tmp, db, runs, _ = case
    with pytest.raises(ExportError, match="store not found"):
        export_static(str(tmp / "s1"), str(tmp / "nope.db"), str(runs))
    (tmp / "empty").mkdir()
    with pytest.raises(ExportError, match="no recorded runs"):
        export_static(str(tmp / "s2"), str(db), str(tmp / "empty"))


def test_export_aborts_if_a_key_value_would_be_published(case, monkeypatch):
    tmp, db, runs, _ = case
    monkeypatch.setenv("META_API_KEY", "running for Attorney General")   # pretend the key leaked into a row
    with pytest.raises(ExportError, match="key value"):
        export_static(str(tmp / "site"), str(db), str(runs))
    assert not (tmp / "site").exists()


def test_cli_export_static(case, capsys):
    tmp, db, runs, _ = case
    rc = cli.main(["export-static", "--out", str(tmp / "site"), "--db", str(db), "--runs-dir", str(runs)])
    assert rc == 0 and json.loads(capsys.readouterr().out)["runs"] == 1
    assert cli.main(["export-static", "--out", str(tmp / "s2"), "--db", str(tmp / "nope.db"), "--runs-dir", str(runs)]) == 1
