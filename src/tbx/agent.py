"""The agent loop: tools until submit_answer, pydantic validation, check, optional repair pass.

Trace events go to runs/<run_id>.jsonl (CONTRACTS.md section 8). Reasoning text is stripped
before parsing and is never sent back to the model or written to the trace.
"""
from __future__ import annotations

import json
import pathlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import ValidationError

from tbx import timeutil as tu
from tbx.checker import check
from tbx.llm import ChatResponse, parse_text_tool_calls, strip_reasoning
from tbx.models import Answer, CheckResult
from tbx.store import Store
from tbx.tools import TOOL_SPECS, run_tool

STEP_CAP = 12          # model requests in the first round (CONTRACTS section 6)
REPAIR_STEP_CAP = 6    # model requests allowed for the single repair round
PRICE_PER_M = {"input": 1.25, "cached": 0.15, "output": 4.25}  # Meta and Baseten list the same prices

NUDGE = ("Your reply was not a submit_answer tool call. Call submit_answer now. If there is no evidence, "
         "use an empty claims list and put the explanation in note.")
LAST_STEP = "You have reached the step limit. Call submit_answer now with the evidence you already have."

SYSTEM_PROMPT = """You are Traceback, an analysis agent working for a digital forensic examiner. You answer \
questions about ONE parsed phone extraction, using only the tools provided. You never guess.

How to work:
1. Call tools to find evidence. Narrow with start/end, contact, number or query instead of pulling everything. \
Use the question window for start/end unless the question says otherwise.
2. When you are done, call submit_answer exactly once. Never answer in plain text.

How to read the evidence:
- A call row with direction "voicemail" is a voicemail, not a missed call. Never describe a voicemail as a missed \
call. Its transcript field, when present, is the carrier's machine transcription; quote it, do not paraphrase it \
as fact.
- Message rows with the same thread_id and the same ts_utc are parts of ONE MMS (e.g. its text and its \
attachment). Describe them as one message, citing each part; never count them as separate messages.

Rules for submit_answer (a deterministic checker verifies every citation and rejects any claim it cannot back up):
- Each claim is one short factual statement, supported by its own citations.
- Copy artifact_id, field and value EXACTLY from a tool row. value is always a string; copy numbers as strings \
(e.g. "95").
- artifact_id values look like msg_, call_, loc_, med_, trn_ or tag_ followed by 10 hex characters. Never cite a \
tool call id.
- Citable fields: ts_local, ts_utc, source_path and the row's data columns. Never cite raw_json, tz_basis, \
ts_original or source_row.
- Every name, phone number, email, URL, date, clock time and coordinate written in a claim's text must appear in \
that claim's own cited values. When the text mentions a date or time, cite that artifact's ts_local.
- Write times as case-local clock times, e.g. "9:42 PM on July 27". Do not write UTC ISO timestamps in the text. \
Write durations in seconds or minutes ("121 seconds"), never as m:ss.
- Never restate the question window in claim text (no "between 9 and 11 PM", no window start or end times): the \
window is not evidence, and only times of the cited artifacts may appear in a claim.
- When citing body or text, quote it exactly (at least 8 characters).
- Leave time_start and time_end empty unless the claim describes a span; if set, they must lie within the cited \
artifacts' times.
- If you find no evidence, submit an empty claims list and explain in note."""

_CITATION = {"type": "object", "properties": {
    "artifact_id": {"type": "string"}, "field": {"type": "string"},
    "value": {"type": "string", "description": "copied exactly from the tool row, as a string"}},
    "required": ["artifact_id", "field", "value"]}
SUBMIT_SPEC = {"type": "function", "function": {
    "name": "submit_answer",
    "description": "Submit the final answer: factual claims, each with citations to tool rows. Call exactly once.",
    "parameters": {"type": "object", "properties": {
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "time_start": {"type": "string", "description": "optional ISO time with Z or offset"},
            "time_end": {"type": "string", "description": "optional ISO time with Z or offset"},
            "citations": {"type": "array", "items": _CITATION}}, "required": ["text", "citations"]}},
        "note": {"type": "string", "description": "e.g. why there is no evidence; never shown as a finding"}},
        "required": ["claims"]}}}
ALL_TOOLS = TOOL_SPECS + [SUBMIT_SPEC]


class AgentError(RuntimeError):
    """The model's output failed validation twice, or the step cap was reached (CLI exit 2)."""


class ChatClient(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResponse: ...


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def estimate_cost(usage: dict) -> float:
    cached = usage.get("cached_tokens", 0)
    fresh = max(0, usage.get("prompt_tokens", 0) - cached)
    return round((fresh * PRICE_PER_M["input"] + cached * PRICE_PER_M["cached"]
                  + usage.get("completion_tokens", 0) * PRICE_PER_M["output"]) / 1e6, 6)


class Trace:
    def __init__(self, run_id: str, trace_dir: str | pathlib.Path):
        self.run_id = run_id
        self.dir = pathlib.Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"

    def emit(self, type_: str, step: int, payload: dict) -> None:
        line = {"ts": tu.now_utc(), "run_id": self.run_id, "step": step, "type": type_, "payload": payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


@dataclass
class RunResult:
    run_id: str
    answer: Answer
    check: CheckResult
    usage: dict
    status: str = "ok"
    first_answer: Answer | None = None
    first_check: CheckResult | None = None
    trace_path: str = ""
    provider: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)


def user_message(question: str, window: tu.Window | None, case_tz: str) -> str:
    lines = [f"Question: {question}", "", f"Case timezone: {case_tz}."]
    if window:
        s, e = window.start_utc, window.end_utc
        lines.append(f"Question window (inclusive): {tu.local_iso(s, case_tz)} to {tu.local_iso(e, case_tz)} "
                     f"case-local, which is {tu.fmt_utc(s)} to {tu.fmt_utc(e)} UTC. Use it for start/end in tool calls.")
    else:
        lines.append("No time window was given.")
    return "\n".join(lines)


def _assistant_message(resp: ChatResponse) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        msg["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                             for tc in resp.tool_calls]
    return msg


def _feedback(result: CheckResult) -> str:
    return json.dumps({
        "status": "checked", "verified_claims": len(result.verified),
        "rejected": [{"text": r.claim.text, "reasons": [x.model_dump() for x in r.reasons]} for r in result.rejected],
        "instruction": "Some claims failed the citation checker. Fix or drop each rejected claim, keep the verified "
                       "ones, and call submit_answer again with the complete final answer. This is the only repair pass."})


class _Run:
    def __init__(self, question: str, store: Store, window: tu.Window | None, client: ChatClient,
                 trace: Trace):
        self.store, self.window, self.client, self.trace = store, window, client, trace
        self.case_tz = store.case_tz
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": user_message(question, window, self.case_tz)}]
        self.step = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "latency_ms": 0}
        self.last_submit_id: str | None = None

    def _ask_model(self) -> ChatResponse:
        self.step += 1
        self.trace.emit("model_request", self.step, {"messages": len(self.messages),
                                                     "tools": [t["function"]["name"] for t in ALL_TOOLS]})
        resp = self.client.chat(self.messages, ALL_TOOLS)
        resp.content, resp.reasoning = strip_reasoning(resp.content), None  # never parsed, sent back or traced
        from_text = False
        if not resp.tool_calls and resp.content:
            recovered = parse_text_tool_calls(resp.content, {t["function"]["name"] for t in ALL_TOOLS})
            if recovered:  # the provider did not parse a tool call the model wrote as text
                resp.tool_calls, resp.content, from_text = recovered, None, True
        for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            self.usage[k] += int(resp.usage.get(k, 0) or 0)
        self.usage["latency_ms"] += resp.latency_ms
        self.trace.emit("model_response", self.step, {
            "content": resp.content, "finish_reason": resp.finish_reason, "latency_ms": resp.latency_ms,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls],
            **({"tool_calls_from_text": True} if from_text else {}),
            "usage": {"prompt_tokens": resp.usage.get("prompt_tokens", 0),
                      "completion_tokens": resp.usage.get("completion_tokens", 0)}})
        self.messages.append(_assistant_message(resp))
        return resp

    def _tool_reply(self, call_id: str, content: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def answer_round(self, cap: int) -> Answer:
        """Run the loop until a valid submit_answer, with one validation retry."""
        retried = False
        start = self.step

        def fail(error: str) -> None:
            nonlocal retried
            self.trace.emit("validation_error", self.step, {"error": error})
            if retried:
                raise AgentError(f"agent output failed validation twice: {error}")
            retried = True

        while True:
            if self.step - start >= cap:
                raise AgentError(f"step cap reached ({cap} model requests) without a valid submit_answer")
            if self.step - start == cap - 1:
                self.messages.append({"role": "user", "content": LAST_STEP})
            resp = self._ask_model()
            submits = [tc for tc in resp.tool_calls if tc.name == "submit_answer"]
            if not resp.tool_calls:
                why = "output cut off (finish_reason=length) before a tool call" if resp.finish_reason == "length" \
                    else "reply was not a submit_answer tool call"
                fail(why)
                self.messages.append({"role": "user", "content": NUDGE})
                continue
            if submits:
                tc = submits[0]
                try:
                    answer = Answer.model_validate(json.loads(tc.arguments or "{}"))
                except (ValueError, ValidationError) as e:
                    err = str(e) if isinstance(e, ValidationError) else f"arguments are not valid JSON: {e}"
                    fail(err)
                    for other in resp.tool_calls:
                        self._tool_reply(other.id, json.dumps(
                            {"error": f"submit_answer failed validation: {err}. Call submit_answer again with "
                                      f"corrected arguments."} if other is tc else
                            {"error": "not run: submit_answer was called in the same turn"}))
                    continue
                for other in resp.tool_calls:
                    if other is not tc:
                        self._tool_reply(other.id, json.dumps({"error": "not run: submit_answer was called"}))
                self.last_submit_id = tc.id
                return answer
            for tc in resp.tool_calls:
                self._run_tool(tc)

    def _run_tool(self, tc) -> None:
        try:
            args = json.loads(tc.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            out = {"rows": [], "total": 0, "truncated": False, "error": f"arguments are not valid JSON: {e}"}
            args = {"_raw": tc.arguments}
        else:
            out = run_tool(self.store, tc.name, args)
        self.trace.emit("tool_call", self.step, {"name": tc.name, "args": args})
        payload = {"name": tc.name, "total": out.get("total", 0), "returned": len(out.get("rows", [])),
                   "truncated": bool(out.get("truncated")), "artifact_ids": [r["artifact_id"] for r in out.get("rows", [])]}
        if "error" in out:
            payload["error"] = out["error"]
        self.trace.emit("tool_result", self.step, payload)
        self._tool_reply(tc.id, json.dumps(out, ensure_ascii=False, default=str))


def run_agent(question: str, store: Store, window: tu.Window | None, client: ChatClient, repair: int = 0,
              trace_dir: str | pathlib.Path = "runs", run_id: str | None = None) -> RunResult:
    trace = Trace(run_id or new_run_id(), trace_dir)
    r = _Run(question, store, window, client, trace)
    trace.emit("run_start", 0, {"question": question, "window": window.as_dict() if window else None,
                                "provider": client.provider, "model": client.model, "repair": repair,
                                "case_tz": r.case_tz})
    first_answer = first_check = None
    try:
        answer = r.answer_round(STEP_CAP)
        trace.emit("final_answer", r.step, {**answer.model_dump(), "round": 1})
        result = check(answer, store, window)
        trace.emit("check_result", r.step, {**result.model_dump(), "round": 1})
        if repair and result.rejected:
            first_answer, first_check = answer, result
            trace.emit("repair", r.step, {"rejected": [x.model_dump() for x in result.rejected]})
            r._tool_reply(r.last_submit_id or "submit_answer", _feedback(result))
            try:
                answer = r.answer_round(REPAIR_STEP_CAP)
            except AgentError as e:  # keep the first, checked answer rather than lose the run
                trace.emit("validation_error", r.step, {"error": f"repair round failed: {e}; keeping round 1"})
                answer = first_answer
            else:
                trace.emit("final_answer", r.step, {**answer.model_dump(), "round": 2})
                result = check(answer, store, window)
                trace.emit("check_result", r.step, {**result.model_dump(), "round": 2})
    except Exception as e:
        trace.emit("run_end", r.step, {"status": "error", "error": f"{type(e).__name__}: {e}",
                                       "totals": _totals(r.usage)})
        raise
    totals = _totals(r.usage)
    trace.emit("run_end", r.step, {"status": "ok", "totals": totals})
    return RunResult(run_id=trace.run_id, answer=answer, check=result, usage=totals, first_answer=first_answer,
                     first_check=first_check, trace_path=str(trace.path), provider=client.provider,
                     model=client.model)


def _totals(usage: dict) -> dict:
    return {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
            "cached_tokens": usage["cached_tokens"], "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "latency_ms": usage["latency_ms"], "est_cost_usd": estimate_cost(usage)}
