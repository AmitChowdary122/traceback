"""tbx.coverage: what the agent searched, from the trace's tool_call / tool_result events (U3)."""
from tbx import coverage

NY = "America/New_York"
WIN = {"start_utc": "2024-06-28T01:00:00Z", "end_utc": "2024-06-28T03:00:00Z"}   # Jun 27, 9-11 PM EDT
W = {"start": "2024-06-28T01:00:00Z", "end": "2024-06-28T03:00:00Z"}


def events(window, *calls):
    """calls: (tool, args, total, returned, truncated[, error]) in order, as tbx.agent traces them."""
    ev = [{"step": 0, "type": "run_start", "payload": {"question": "q", "window": window, "case_tz": NY}}]
    for n, (tool, args, total, returned, truncated, *err) in enumerate(calls, 1):
        ev.append({"step": n, "type": "tool_call", "payload": {"name": tool, "args": args}})
        res = {"name": tool, "total": total, "returned": returned, "truncated": truncated,
               "artifact_ids": [f"x{i}" for i in range(returned)]}
        if err:
            res["error"] = err[0]
        ev.append({"step": n, "type": "tool_result", "payload": res})
    return ev


def by_family(cov):
    return {f["family"]: f for f in cov["families"]}


def test_flagship_run_in_plain_english():
    """The calls of recorded run 20261101T162631Z-ae5039."""
    cov = coverage.coverage(events(WIN,
        ("get_locations", {**W, "limit": 50}, 73, 50, True),
        ("get_calls", {**W, "limit": 50}, 0, 0, False),
        ("search_messages", {**W, "limit": 50}, 2, 2, False),
        ("get_artifact", {"artifact_id": "msg_c9a0e4d555"}, 1, 1, False),
        ("get_artifact", {"artifact_id": "loc_3f4397b779"}, 1, 1, False),
        ("get_locations", {**W, "limit": 200}, 73, 73, False)), NY)
    f = by_family(cov)
    assert f["locations"]["sentence"] == "Locations: 73 in the window, 50 shown; then all 73 with a higher limit."
    assert f["calls"]["sentence"] == "Calls and voicemail: none in the window."
    assert f["messages"]["sentence"] == "Messages: 2 in the window, all shown."
    assert f["media"]["sentence"] == "Photos and media: never searched."
    assert f["transcripts"]["sentence"] == "Voice transcripts: never searched."
    assert cov["summary"] == ("Searched 3 of 5 evidence families. Never searched: photos and media, voice transcripts. "
                              "Opened 2 records directly by ID.")
    loc = f["locations"]["calls"]
    assert [(c["window"], c["total"], c["returned"], c["truncated"], c["limit"]) for c in loc] == [
        ("Jun 27, 2024 9:00 PM – 11:00 PM", 73, 50, True, 50), ("Jun 27, 2024 9:00 PM – 11:00 PM", 73, 73, False, 200)]
    assert [f["family"] for f in cov["families"]] == ["messages", "calls", "locations", "media", "transcripts"]


def test_narrowed_searches_name_their_range():
    """Like candidate run 20261102T154901Z-61b070: a truncated day-wide search, then narrow windows."""
    day = {"start_utc": "2024-07-24T04:00:00Z", "end_utc": "2024-07-25T03:59:00Z"}
    cov = coverage.coverage(events(day,
        ("find_media", {"kind": "image", "start": "2024-07-24T04:00:00Z", "end": "2024-07-25T03:59:00Z"}, 3, 3, False),
        ("get_locations", {"start": "2024-07-24T04:00:00Z", "end": "2024-07-25T03:59:00Z", "limit": 200}, 780, 200, True),
        ("get_locations", {"start": "2024-07-24T15:55:00Z", "end": "2024-07-24T16:10:00Z"}, 0, 0, False),
        ("get_locations", {"start": "2024-07-24T19:50:00Z", "end": "2024-07-24T20:05:00Z"}, 40, 40, False)), NY)
    f = by_family(cov)
    assert f["media"]["sentence"] == "Photos and media: 3 of kind image in the window, all shown."
    assert f["locations"]["sentence"] == ("Locations: 780 in the window, 200 shown; then narrowed to "
                                          "Jul 24, 2024 11:55 AM – 12:10 PM: none; then narrowed to "
                                          "Jul 24, 2024 3:50 PM – 4:05 PM: all 40.")


def test_filters_whole_extraction_and_single_rows():
    cov = coverage.coverage(events(None,
        ("search_messages", {"query": "gate"}, 1, 1, False),
        ("get_calls", {"contact": "Liz Dehner", "direction": "outgoing"}, 12, 12, False),
        ("get_transcripts", {"start": "2024-02-09T05:00:00Z"}, 2, 2, False)), NY)
    f = by_family(cov)
    assert f["messages"]["sentence"] == "Messages: 1 matching 'gate' in the whole extraction, shown."
    assert f["calls"]["sentence"] == ("Calls and voicemail: 12 for contact 'Liz Dehner', direction outgoing in the "
                                      "whole extraction, all shown.")
    assert f["transcripts"]["sentence"] == "Voice transcripts: 2 from Feb 9, 2024 12:00 AM on, all shown."
    assert cov["summary"] == "Searched 3 of 5 evidence families. Never searched: locations, photos and media."


def test_repeats_errors_lookups_and_pending_calls():
    ev = events(WIN,
        ("search_messages", {**W}, 2, 2, False),
        ("search_messages", {**W}, 2, 2, False),
        ("get_locations", {"start": "yesterday"}, 0, 0, False, "bad start: 'yesterday'"),
        ("get_artifact", {"artifact_id": "msg_c9a0e4d555"}, 1, 1, False),
        ("get_artifact", {"artifact_id": "call_0d96ca43"}, 0, 0, False))
    ev.append({"step": 9, "type": "tool_call", "payload": {"name": "find_media", "args": {}}})   # still running
    cov = coverage.coverage(ev, NY)
    f = by_family(cov)
    assert f["messages"]["sentence"] == "Messages: 2 in the window, all shown; then the same search again."
    assert f["locations"]["sentence"] == "Locations: the search failed (bad start: 'yesterday')."
    assert f["locations"]["searched"] is False
    assert f["media"]["sentence"] == "Photos and media: never searched."
    assert cov["summary"].endswith("Opened 2 records directly by ID (1 not found).")


def test_no_tool_calls():
    cov = coverage.coverage(events(WIN), NY)
    assert cov["summary"] == "No evidence family was searched."
    assert all(not f["searched"] for f in cov["families"])
