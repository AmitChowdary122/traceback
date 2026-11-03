"""tbx.agent: tool loop, submit_answer validation, repair pass, JSONL trace (CONTRACTS.md 6, 8)."""
import json

import pytest

from tbx import agent
from tbx.llm import ChatResponse, ToolCall
from tbx.store import Store, artifact_id
from tbx.timeutil import Window, parse_utc

NY = "America/New_York"
CALL = artifact_id("calls", "calls.tsv", "1")
GOOD = {"claims": [{"text": "Liz Dehner placed an outgoing call.",
                    "citations": [{"artifact_id": CALL, "field": "contact_name", "value": "Liz Dehner"},
                                  {"artifact_id": CALL, "field": "direction", "value": "outgoing"}]}]}
BAD_CLAIM = {"text": "Liz Dehner called Bob Smith.", "citations": [
    {"artifact_id": CALL, "field": "contact_name", "value": "Bob Smith"}]}


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "c.db"))
    s.set_meta("device_tz", NY)
    s.insert("calls", {"artifact_id": CALL, "source_path": "calls.tsv", "source_row": "1",
                       "ts_utc": "2024-07-28T01:40:00Z", "ts_original": "x", "tz_basis": "utc_string",
                       "raw_json": "{}", "app": "Phone", "direction": "outgoing", "number": "9195794674",
                       "contact_name": "Liz Dehner", "duration_s": 121})
    return s


class FakeClient:
    provider, model = "fake", "fake-model"

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def chat(self, messages, tools):
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        step = self.script.pop(0)
        return step(self.requests[-1]) if callable(step) else step


def tools_turn(*calls, content=None, reasoning=None):
    return ChatResponse(content=content, reasoning=reasoning, finish_reason="tool_calls", latency_ms=5,
                        usage={"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0},
                        tool_calls=[ToolCall(id=f"call_{i}", name=n, arguments=json.dumps(a) if isinstance(a, dict) else a)
                                    for i, (n, a) in enumerate(calls)])


def text_turn(text, finish="stop"):
    return ChatResponse(content=text, reasoning=None, finish_reason=finish, latency_ms=5,
                        usage={"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 0}, tool_calls=[])


WINDOW = Window(parse_utc("2024-07-28T01:00:00Z"), parse_utc("2024-07-28T03:00:00Z"))


def run(store, client, tmp_path, repair=0):
    return agent.run_agent("Who did the phone call?", store, WINDOW, client, repair=repair, trace_dir=tmp_path / "runs")


def events(tmp_path, run_id):
    return [json.loads(x) for x in (tmp_path / "runs" / f"{run_id}.jsonl").read_text().splitlines()]


def test_happy_path_tools_then_submit(store, tmp_path):
    c = FakeClient([tools_turn(("get_calls", {"start": "2024-07-28T01:00:00Z"})), tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path)
    assert [x.text for x in res.check.verified] == [GOOD["claims"][0]["text"]] and res.status == "ok"
    tool_msg = [m for m in c.requests[1]["messages"] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["total"] == 1 and tool_msg["tool_call_id"] == "call_0"
    types = [e["type"] for e in events(tmp_path, res.run_id)]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert {"model_request", "model_response", "tool_call", "tool_result", "final_answer", "check_result"} <= set(types)
    assert res.usage["prompt_tokens"] == 200 and res.usage["est_cost_usd"] > 0


def test_system_prompt_and_window_in_messages(store, tmp_path):
    c = FakeClient([tools_turn(("submit_answer", GOOD))])
    run(store, c, tmp_path)
    msgs = c.requests[0]["messages"]
    assert msgs[0]["role"] == "system" and "submit_answer" in msgs[0]["content"]
    assert "121 seconds" in msgs[0]["content"]  # durations as m:ss would read as clock times (R4)
    sp = msgs[0]["content"]
    assert "voicemail" in sp and "not a missed call" in sp                    # v1.2
    assert "same thread_id and the same ts_utc" in sp and "ONE MMS" in sp    # one MMS, several rows
    assert "Never restate the question window in claim text" in sp
    assert "2024-07-27T21:00:00-04:00" in msgs[1]["content"] and "America/New_York" in msgs[1]["content"]


def test_reasoning_and_think_blocks_never_sent_back_or_traced(store, tmp_path):
    c = FakeClient([tools_turn(("get_calls", {}), content="<think>SECRET-1</think>ok", reasoning="SECRET-2"),
                    tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path)
    blob = json.dumps(c.requests[1]["messages"]) + (tmp_path / "runs" / f"{res.run_id}.jsonl").read_text()
    assert "SECRET-1" not in blob and "SECRET-2" not in blob and "reasoning_content" not in blob


def test_text_reply_gets_one_nudge(store, tmp_path):
    c = FakeClient([text_turn("No data found."), tools_turn(("submit_answer", {"claims": [], "note": "none"}))])
    res = run(store, c, tmp_path)
    assert res.answer.claims == [] and res.answer.note == "none"
    assert c.requests[1]["messages"][-1]["role"] == "user" and "submit_answer" in c.requests[1]["messages"][-1]["content"]
    assert "validation_error" in [e["type"] for e in events(tmp_path, res.run_id)]


def test_second_validation_failure_raises(store, tmp_path):
    c = FakeClient([text_turn("No."), text_turn("Still no.")])
    with pytest.raises(agent.AgentError):
        run(store, c, tmp_path)
    run_end = [e for e in (json.loads(x) for x in next((tmp_path / "runs").glob("*.jsonl")).read_text().splitlines())
               if e["type"] == "run_end"]
    assert run_end and run_end[0]["payload"]["status"] == "error"


def test_invalid_arguments_retry_with_pydantic_error(store, tmp_path):
    c = FakeClient([tools_turn(("submit_answer", {"claims": [{"citations": []}]})), tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path)
    fb = [m for m in c.requests[1]["messages"] if m["role"] == "tool"][-1]["content"]
    assert "text" in fb and "Field required" in fb
    assert res.check.verified


def test_unparseable_json_arguments_count_as_validation_failure(store, tmp_path):
    c = FakeClient([tools_turn(("submit_answer", "{not json")), tools_turn(("submit_answer", "{still not"))])
    with pytest.raises(agent.AgentError):
        run(store, c, tmp_path)


def test_length_cutoff_without_tool_call_is_retryable(store, tmp_path):
    c = FakeClient([text_turn(None, finish="length"), tools_turn(("submit_answer", GOOD))])
    assert run(store, c, tmp_path).check.verified


class MetaFakeClient(FakeClient):
    """Records the per-request options a Meta LLMClient would get."""
    provider, model = "meta", "muse-spark-1.3"

    def chat(self, messages, tools, **opts):
        self.opts = getattr(self, "opts", []) + [opts]
        return super().chat(messages, tools)


def _empty_assistant_turns(messages):
    return [m for m in messages if m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls")]


def test_length_cutoff_is_not_echoed_and_gets_the_cutoff_nudge(store, tmp_path):
    """Eval 1 q03/q10: the cut-off reply (no content, no tool calls) was sent back and Meta returned HTTP 400."""
    c = FakeClient([tools_turn(("get_calls", {})), text_turn(None, finish="length"), tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path)
    assert res.check.verified and len(c.requests) == 3
    retry = c.requests[2]["messages"]
    assert not _empty_assistant_turns(retry)
    assert retry[-1] == {"role": "user", "content": agent.LENGTH_NUDGE}
    assert agent.LENGTH_NUDGE == ("Your previous reply was cut off before you called submit_answer. Call "
                                  "submit_answer now. Keep claims short and quote at most 200 characters.")
    assert [m["role"] for m in retry[-3:]] == ["assistant", "tool", "user"]  # the get_calls turn, then the nudge
    errors = [e for e in events(tmp_path, res.run_id) if e["type"] == "validation_error"]
    assert len(errors) == 1 and "finish_reason=length" in errors[0]["payload"]["error"]


def test_empty_stop_reply_is_not_echoed_and_gets_the_plain_nudge(store, tmp_path):
    c = FakeClient([text_turn(None, finish="stop"), tools_turn(("submit_answer", GOOD))])
    assert run(store, c, tmp_path).check.verified
    retry = c.requests[1]["messages"]
    assert not _empty_assistant_turns(retry) and retry[-1] == {"role": "user", "content": agent.NUDGE}


def test_text_reply_is_still_echoed(store, tmp_path):
    c = FakeClient([text_turn("No data found."), tools_turn(("submit_answer", GOOD))])
    run(store, c, tmp_path)
    assert c.requests[1]["messages"][-2] == {"role": "assistant", "content": "No data found."}


def test_meta_cutoff_retry_uses_low_reasoning_effort_once(store, tmp_path):
    c = MetaFakeClient([tools_turn(("get_calls", {})), text_turn(None, finish="length"),
                        tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path)
    assert res.check.verified
    assert c.opts == [{}, {}, {"reasoning_effort": "low"}]
    reqs = [e["payload"] for e in events(tmp_path, res.run_id) if e["type"] == "model_request"]
    assert [r.get("reasoning_effort") for r in reqs] == [None, None, "low"]


def test_meta_plain_nudge_keeps_default_reasoning_effort(store, tmp_path):
    c = MetaFakeClient([text_turn("No data found."), tools_turn(("submit_answer", GOOD))])
    run(store, c, tmp_path)
    assert c.opts == [{}, {}]


def test_cutoff_twice_is_a_validation_failure(store, tmp_path):
    c = MetaFakeClient([text_turn(None, finish="length"), text_turn(None, finish="length")])
    with pytest.raises(agent.AgentError, match="twice"):
        run(store, c, tmp_path)
    assert len(c.requests) == 2 and not _empty_assistant_turns(c.requests[1]["messages"])


class _Recorder:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        from types import SimpleNamespace as NS
        self.calls.append(kwargs)
        msg = NS(content="hi", tool_calls=None, model_extra={})
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=1, completion_tokens=1,
                                                                         prompt_tokens_details=None))


@pytest.mark.parametrize("provider,env_effort,arg,expected", [
    ("meta", None, None, None), ("meta", None, "low", "low"), ("meta", "high", None, "high"),
    ("meta", "high", "low", "low"), ("baseten", None, "low", None)])
def test_llm_client_reasoning_effort(monkeypatch, provider, env_effort, arg, expected):
    from types import SimpleNamespace as NS
    from tbx.llm import LLMClient
    monkeypatch.setenv("META_API_KEY", "k"), monkeypatch.setenv("BASETEN_API_KEY", "k")
    if env_effort:
        monkeypatch.setenv("TBX_REASONING_EFFORT", env_effort)
    else:
        monkeypatch.delenv("TBX_REASONING_EFFORT", raising=False)
    client = LLMClient(provider)
    rec = _Recorder()
    client._client = NS(chat=NS(completions=rec))
    client.chat([{"role": "user", "content": "x"}], [], **({"reasoning_effort": arg} if arg else {}))
    assert rec.calls[0].get("reasoning_effort") == expected


def test_step_cap(store, tmp_path):
    c = FakeClient([tools_turn(("get_calls", {}))] * 30)
    with pytest.raises(agent.AgentError, match="step"):
        run(store, c, tmp_path)
    assert len(c.requests) <= agent.STEP_CAP


def test_unknown_tool_and_bad_tool_json_reported_to_model(store, tmp_path):
    c = FakeClient([tools_turn(("drop_tables", {}), ("get_calls", "{oops")), tools_turn(("submit_answer", GOOD))])
    run(store, c, tmp_path)
    outs = [json.loads(m["content"]) for m in c.requests[1]["messages"] if m["role"] == "tool"]
    assert all("error" in o for o in outs)


def test_repair_pass_sends_rejections_once_and_keeps_first_round(store, tmp_path):
    first = {"claims": GOOD["claims"] + [BAD_CLAIM]}
    c = FakeClient([tools_turn(("submit_answer", first)), tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path, repair=1)
    assert len(c.requests) == 2 and "VALUE_MISMATCH" in json.dumps(c.requests[1]["messages"])
    assert len(res.check.verified) == 1 and res.check.rejected == []
    assert len(res.first_check.rejected) == 1
    assert "repair" in [e["type"] for e in events(tmp_path, res.run_id)]


def test_no_repair_when_nothing_rejected(store, tmp_path):
    c = FakeClient([tools_turn(("submit_answer", GOOD))])
    res = run(store, c, tmp_path, repair=1)
    assert len(c.requests) == 1 and res.first_check is None


def test_estimate_cost():
    assert agent.estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "cached_tokens": 0}) \
        == pytest.approx(1.25 + 4.25)
    assert agent.estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 0, "cached_tokens": 1_000_000}) \
        == pytest.approx(0.15)


TEXT_CALL = (__import__("pathlib").Path(__file__).parent / "fixtures" / "baseten_text_tool_call.txt").read_text()


def test_parse_text_tool_calls_from_a_real_baseten_reply():
    from tbx.llm import parse_text_tool_calls
    calls = parse_text_tool_calls(TEXT_CALL, {"submit_answer"})
    assert [c.name for c in calls] == ["submit_answer"]
    args = json.loads(calls[0].arguments)
    assert set(args) == {"claims"} and len(args["claims"]) >= 2           # the empty note is dropped
    assert args["claims"][0]["citations"][2] == {"artifact_id": "tag_735fb27cd7", "field": "label", "value": "cell phone"}
    assert parse_text_tool_calls(TEXT_CALL, {"get_calls"}) == []           # unknown names are never invented
    assert parse_text_tool_calls("plain words", {"submit_answer"}) == []


def test_text_tool_call_is_recovered_not_counted_as_a_failure(store, tmp_path):
    reply = ('<x:function_calls>\n<x:invoke name="submit_answer">\n<x:parameter name="claims">'
             + json.dumps(GOOD["claims"]) + '</x:parameter>\n</x:invoke>\n</x:function_calls>')
    c = FakeClient([tools_turn(("get_calls", {})), text_turn(reply)])
    res = run(store, c, tmp_path)
    assert len(res.check.verified) == 1 and len(c.requests) == 2
    evs = events(tmp_path, res.run_id)
    assert not [e for e in evs if e["type"] == "validation_error"]
    resp = [e for e in evs if e["type"] == "model_response"][-1]["payload"]
    assert resp["tool_calls_from_text"] is True and resp["tool_calls"][0]["name"] == "submit_answer"
