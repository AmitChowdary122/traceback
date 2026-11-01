"""Acceptance: `tbx ask` / `tbx check` end to end, offline (CONTRACTS.md sections 5, 6, 8, 9).

The model is a local fake provider that replays the real responses recorded in spikes S1/S4,
so these tests also pin down the behaviours the spikes found:
- Meta 400s on any tool_choice except "auto" (fixture s1_tool_choice_named_rejected)
- reasoning tokens count against max_tokens; 800 was not enough (s1_..._max800_reasoning_exhausted)
- the open model can answer in plain text instead of calling submit_answer (s4_baseten_chat_negative_*)
- tool_call ids look like call_<uuid> on Meta (s1_chat_tools_turn1)
"""
from __future__ import annotations

import json

import pytest
import tbxcase
from fakeprovider import FakeProvider, closed_port, has_tool_results, text_reply, tool_messages, tool_reply
from tbxcase import IDS, cite, find_trace, spike_claims

META_CLAIMS = spike_claims("s1_chat_tools_turn2_submit_4k")
OPEN_CLAIMS = spike_claims("s4_baseten_chat_tools_turn2_submit")
WINDOW_ARGS = ["--from", "2023-03-13 21:00", "--to", "2023-03-13 23:00"]  # case-local (America/New_York)
TOOL_ARGS = {"start": "2023-03-14T01:00:00Z", "end": "2023-03-14T03:00:00Z", "limit": 50}
CONTRACT_TOOLS = {"search_messages", "get_calls", "get_locations", "find_media", "get_transcripts", "get_artifact"}
TRACE_TYPES = {"run_start", "model_request", "model_response", "tool_call", "tool_result", "validation_error",
               "final_answer", "check_result", "repair", "run_end"}
QUESTION = "Where was this phone between 9 and 11 pm, and who did it talk to?"

# One out-of-window claim, to prove --from/--to are read in the case timezone.
CAROL_MORNING = {"text": "A message was received.", "citations": [cite("MSG_CAROL", "sender", "Carol")]}
SUBMIT_ARGS = {"claims": META_CLAIMS + [OPEN_CLAIMS[0], CAROL_MORNING]}  # 2 verified, 2 rejected


def scripted(submit_args: dict, model: str = "muse-spark-1.3", **submit_extra):
    """Turn 1: parallel get_calls + get_locations (as muse-spark-1.3 really did). Then submit_answer."""
    def responder(req, n):
        if not has_tool_results(req):
            return tool_reply(("get_calls", TOOL_ARGS), ("get_locations", TOOL_ARGS), model=model,
                              content="I'll pull the call and location records for that window.")
        return tool_reply(("submit_answer", submit_args), model=model, **submit_extra)
    return responder


def ask(run_cli, env, *extra, question=QUESTION, timeout=180):
    return run_cli(["ask", question, *WINDOW_ARGS, "--json", "--db", env["TBX_DB"], *extra], env, timeout=timeout)


def parse_json_out(proc) -> dict:
    assert proc.returncode == 0, f"exit {proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    try:
        return json.loads(proc.stdout)
    except ValueError:
        pytest.fail(f"--json stdout is not one JSON document:\n{proc.stdout[-3000:]}")


def codes(rejected_entry) -> set[str]:
    return {r["code"] for r in rejected_entry["reasons"]}


# ------------------------------------------------------------------------------- happy path
@pytest.fixture
def happy(run_cli, cli_env):
    with FakeProvider(scripted(SUBMIT_ARGS)) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env)
    return proc, fp, cli_env


def test_ask_json_output_and_check(happy):
    proc, fp, env = happy
    out = parse_json_out(proc)
    assert {"run_id", "verified", "rejected", "usage"} <= set(out), sorted(out)
    assert sorted(c["text"] for c in out["verified"]) == sorted(c["text"] for c in META_CLAIMS)
    by_text = {r["claim"]["text"]: codes(r) for r in out["rejected"]}
    assert by_text == {OPEN_CLAIMS[0]["text"]: {"UNGROUNDED_ENTITY:timestamp"},
                       CAROL_MORNING["text"]: {"OUTSIDE_WINDOW"}}, by_text


def test_model_requests_follow_contract_and_spike_findings(happy):
    proc, fp, env = happy
    assert proc.returncode == 0, proc.stderr[-2000:]
    reqs = fp.chat_requests
    assert len(reqs) >= 2
    for r in reqs:
        assert r.get("model") == "muse-spark-1.3" and "contributor" not in r.get("model", "")
        assert r.get("temperature") == pytest.approx(0.1)
        assert r.get("tool_choice") in (None, "auto"), "Meta returns 400 for any tool_choice except 'auto' (spike S1)"
        budget = r.get("max_tokens", r.get("max_completion_tokens"))
        assert budget is None or budget >= 4000, f"max_tokens={budget}: reasoning tokens count against it (spike S1)"
        names = {t["function"]["name"] for t in r.get("tools", [])}
        assert CONTRACT_TOOLS | {"submit_answer"} <= names, f"missing tools: {CONTRACT_TOOLS | {'submit_answer'} - names}"
    for rec in fp.requests:
        assert rec["headers"].get("authorization") == f"Bearer {env['META_API_KEY']}"


def test_tool_results_follow_contract_shape(happy):
    proc, fp, env = happy
    assert proc.returncode == 0, proc.stderr[-2000:]
    follow_up = next(r for r in fp.chat_requests if has_tool_results(r))
    results = [json.loads(m["content"]) for m in tool_messages(follow_up)]
    assert len(results) == 2, "both parallel tool calls must get a tool message back"
    rows_by_id = {}
    for res in results:
        assert set(res) >= {"rows", "total", "truncated"} and isinstance(res["truncated"], bool), res
        assert res["total"] >= len(res["rows"])
        keys = [(r["ts_utc"], r["artifact_id"]) for r in res["rows"]]
        assert keys == sorted(keys), "rows ordered by ts_utc, then artifact_id"
        for r in res["rows"]:
            assert {"artifact_id", "ts_utc", "ts_local"} <= set(r) and "raw_json" not in r, r
            rows_by_id[r["artifact_id"]] = r
    call = rows_by_id[IDS["CALL_ALICE"]]
    assert call["ts_local"] == "2023-03-13T21:42:07-04:00"
    assert {IDS["CALL_BOB"], IDS["LOC_SPIKE"], IDS["LOC_PRECISE"]} <= set(rows_by_id)
    vm = rows_by_id[IDS["CALL_VM"]]  # v1.2
    assert vm["direction"] == "voicemail" and "ready for pickup" in vm["transcript"]


def test_trace_jsonl(happy, tmp_path):
    proc, fp, env = happy
    out = parse_json_out(proc)
    trace = find_trace(tmp_path, out["run_id"])
    assert trace, f"runs/{out['run_id']}.jsonl not found in cwd or repo"
    events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    for e in events:
        assert {"ts", "run_id", "step", "type", "payload"} <= set(e), e
        assert e["run_id"] == out["run_id"] and isinstance(e["step"], int) and e["type"] in TRACE_TYPES, e
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert {"model_request", "model_response", "tool_call", "tool_result", "final_answer", "check_result"} <= set(types)
    assert events[-1]["payload"]["status"] == "ok"
    tr = next(e for e in events if e["type"] == "tool_result")["payload"]
    assert {"name", "total", "returned", "truncated", "artifact_ids"} <= set(tr), tr


def test_keys_never_reach_output_or_trace(happy, tmp_path):
    proc, fp, env = happy
    blobs = [proc.stdout, proc.stderr] + [p.read_text() for p in tmp_path.rglob("*.jsonl")]
    for key in ("META_API_KEY", "BASETEN_API_KEY"):
        assert all(env[key] not in b for b in blobs), f"{key} canary leaked into output or a trace"


# ------------------------------------------------------------------------------- failure paths
def test_text_only_reply_is_retried_once(run_cli, cli_env, tmp_path):
    """Spike S4: muse-spark-open answered a negative question in plain text; one nudge fixed it."""
    def responder(req, n):
        if not has_tool_results(req):
            return tool_reply(("get_calls", TOOL_ARGS))
        if n == 2:
            return text_reply("No data found in the requested window.")
        return tool_reply(("submit_answer", {"claims": [], "note": "no evidence found in window"}))
    with FakeProvider(responder) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        out = parse_json_out(ask(run_cli, cli_env))
    assert out["verified"] == [] and out["rejected"] == []
    trace = find_trace(tmp_path, out["run_id"])
    assert trace and '"validation_error"' in trace.read_text()


def test_two_text_only_replies_exit_2(run_cli, cli_env):
    def responder(req, n):
        return text_reply("No data found.") if has_tool_results(req) else tool_reply(("get_calls", TOOL_ARGS))
    with FakeProvider(responder) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])
    assert len(fp.chat_requests) <= 4


def test_invalid_submit_arguments_twice_exit_2(run_cli, cli_env):
    bad = {"claims": [{"citations": [{"artifact_id": IDS["CALL_ALICE"], "field": "number"}]}]}  # no text, no value
    with FakeProvider(scripted(bad)) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-2000:])


def test_reasoning_text_is_stripped(run_cli, cli_env):
    secret = "PRIVATE-CHAIN-OF-THOUGHT-7731"
    with FakeProvider(scripted({"claims": META_CLAIMS[:1]}, content=f"<think>{secret}</think>",
                               reasoning_content=f"{secret} reasoning")) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env)
    out = parse_json_out(proc)
    assert len(out["verified"]) == 1
    assert secret not in proc.stdout


def test_provider_unreachable_exit_3(run_cli, cli_env):
    cli_env["META_BASE_URL"] = f"http://127.0.0.1:{closed_port()}/v1"
    proc = ask(run_cli, cli_env)
    assert proc.returncode == 3, (proc.returncode, proc.stderr[-2000:])
    assert cli_env["META_API_KEY"] not in proc.stdout + proc.stderr


def test_provider_http_500_exit_3(run_cli, cli_env):
    with FakeProvider(lambda req, n: (500, {"error": {"message": "upstream exploded", "type": "server_error"}})) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env)
    assert proc.returncode == 3, (proc.returncode, proc.stderr[-2000:])


def test_step_cap_is_enforced(run_cli, cli_env):
    """A model that never stops calling tools must not loop forever (step cap 12)."""
    with FakeProvider(lambda req, n: tool_reply(("get_calls", TOOL_ARGS))) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        proc = ask(run_cli, cli_env, timeout=300)
    assert proc.returncode != 0, "no answer was ever submitted, so the run cannot succeed"
    assert len(fp.chat_requests) <= 13, f"{len(fp.chat_requests)} model requests; step cap is 12 (+1 retry)"


def test_baseten_provider_routing(run_cli, cli_env):
    """--provider baseten must use BASETEN_BASE_URL/BASETEN_API_KEY and the id Baseten really lists (spike S4)."""
    model = "meta-models/muse-spark-open"
    with FakeProvider(scripted(SUBMIT_ARGS, model=model)) as bt, FakeProvider(lambda r, n: (500, {})) as meta:
        cli_env.update(BASETEN_BASE_URL=bt.base_url, META_BASE_URL=meta.base_url)
        out = parse_json_out(ask(run_cli, cli_env, "--provider", "baseten", "--model", model))
    assert meta.chat_requests == [], "meta endpoint must not be called with --provider baseten"
    assert all(r["model"] == model for r in bt.chat_requests)
    assert all(rec["headers"].get("authorization") == f"Bearer {cli_env['BASETEN_API_KEY']}" for rec in bt.requests)
    assert len(out["verified"]) == 2


# ------------------------------------------------------------------------------- CLI surface
def test_usage_errors_exit_1(run_cli, cli_env, tmp_path):
    """Contract: 1 = usage or input error, 2 is reserved for agent validation failure.
    (argparse exits 2 on usage errors by default, so this needs explicit handling.)"""
    for args in (["ask"], ["check", str(tmp_path / "missing.json"), "--db", cli_env["TBX_DB"]],
                 ["ask", QUESTION, "--repair", "5", "--db", cli_env["TBX_DB"]],
                 ["ask", QUESTION, "--from", "not-a-time", "--db", cli_env["TBX_DB"]]):
        proc = run_cli(args, cli_env, timeout=60)
        assert proc.returncode == 1, (args, proc.returncode, proc.stderr[-1500:])


def test_tbx_check_cli_prints_check_result_json(run_cli, cli_env, tmp_path):
    """CONTRACTS.md v1.1 section 9 (C13): one CheckResult JSON document on stdout, exit 0 even
    when claims are rejected."""
    tool_call_id_claim = {"text": "An outgoing call was placed.",
                          "citations": [cite("call_0d96ca43-8907-4d91-aac2-346156bd69d6", "direction", "outgoing")]}
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"claims": [META_CLAIMS[0], tool_call_id_claim]}))
    proc = run_cli(["check", str(path), *WINDOW_ARGS, "--db", cli_env["TBX_DB"]], cli_env, timeout=60)
    out = parse_json_out(proc)
    assert {"verified", "rejected"} <= set(out), sorted(out)
    assert [c["text"] for c in out["verified"]] == [META_CLAIMS[0]["text"]]
    assert len(out["rejected"]) == 1 and out["rejected"][0]["claim"]["text"] == tool_call_id_claim["text"]
    reasons = out["rejected"][0]["reasons"]
    assert [(r["code"], r["citation_index"]) for r in reasons] == [("UNKNOWN_ARTIFACT", 0)], reasons


def test_tbx_check_invalid_answer_file_exit_1(run_cli, cli_env, tmp_path):
    for name, content in (("not_json.json", "{claims: oops"), ("not_an_answer.json", json.dumps({"claims": [{"citations": []}]}))):
        path = tmp_path / name
        path.write_text(content)
        proc = run_cli(["check", str(path), "--db", cli_env["TBX_DB"]], cli_env, timeout=60)
        assert proc.returncode == 1, (name, proc.returncode, proc.stderr[-1500:])


# ------------------------------------------------------------------------------- P1: repair pass
@pytest.mark.p1
def test_repair_pass_sends_rejections_back_once(run_cli, cli_env, tmp_path):
    fixed = dict(OPEN_CLAIMS[0], text="An outgoing phone call was made to Alice at +1 (555) 010-1234 "
                                      "at 2023-03-14T01:42:07Z lasting 95 seconds.")
    seen_feedback = []

    def responder(req, n):
        if not has_tool_results(req):
            return tool_reply(("get_calls", TOOL_ARGS), ("get_locations", TOOL_ARGS))
        blob = json.dumps(req["messages"])
        if "UNGROUNDED_ENTITY" in blob:
            seen_feedback.append(n)
            return tool_reply(("submit_answer", {"claims": META_CLAIMS + [fixed]}))
        return tool_reply(("submit_answer", {"claims": META_CLAIMS + [OPEN_CLAIMS[0]]}))
    with FakeProvider(responder) as fp:
        cli_env["META_BASE_URL"] = fp.base_url
        out = parse_json_out(ask(run_cli, cli_env, "--repair", "1"))
    assert len(seen_feedback) == 1, "rejections go back to the model exactly once"
    assert len(out["verified"]) == 3 and out["rejected"] == []
    trace = find_trace(tmp_path, out["run_id"])
    assert trace and '"repair"' in trace.read_text(), "first-round rejections are kept in the trace"


def test_case_window_is_case_local(happy):
    """--from/--to without an offset are case-local: 21:00-23:00 EDT. The 08:00 EDT message is outside."""
    proc, fp, env = happy
    out = parse_json_out(proc)
    carol = [r for r in out["rejected"] if r["claim"]["text"] == CAROL_MORNING["text"]]
    assert carol and codes(carol[0]) == {"OUTSIDE_WINDOW"}
    assert tbxcase.CASE_TZ == "America/New_York"
