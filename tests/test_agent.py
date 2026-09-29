"""Agent loop tests with a scripted fake model (no API calls, no network)."""

import json
from types import SimpleNamespace as NS

import pytest

from app import agent as agent_mod
from app import tools as tools_mod
from app.chunking import PageSpan
from app.store import Hit
from app.tools import Toolbox, ToolError, calculator

DOC = "Franchisees pay an ongoing royalty of 6 percent of gross sales."
HITS = [Hit(id="h::0", text=DOC, source="orbit.pdf", page_start=1, page_end=1, spans=[PageSpan(1, 0, len(DOC))], distance=0.2)]


# --- calculator ----------------------------------------------------------------

@pytest.mark.parametrize("expr, expected", [
    ("0.06 * 300000", "18000"), ("2^10", "1024"), ("sqrt(16) + 1", "5"), ("(45,000 + 5) / 5", "9001"),
    ("round(pi, 3)", "3.142"), ("-3 ** 2", "-9"), ("10 // 3 + 10 % 3", "4"),
])
def test_calculator(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')", "().__class__", "open('x')", "9 ** 9 ** 9", "1/0", "", "x + 1", "True + 1",
])
def test_calculator_rejects(expr):
    with pytest.raises(ToolError):
        calculator(expr)


# --- toolbox -------------------------------------------------------------------

def test_toolbox_labels_are_stable_and_errors_are_returned(monkeypatch):
    monkeypatch.setattr(tools_mod.retrieval, "search", lambda cfg, q, k, *a: HITS)
    box = Toolbox()
    out1, err1, _ = box.execute("search_docs", {"query": "royalty"})
    out2, _, _ = box.execute("search_docs", {"query": "royalty again"})
    assert not err1 and "[S1] (orbit.pdf, page 1)" in out1 and "[S1]" in out2 and len(box.sources) == 1

    assert box.execute("calculator", {"expression": "1/0"})[:2] == ("Error: division by zero", True)
    assert box.execute("nope", {})[1] is True
    assert box.execute("search_docs", {"wrong_arg": "x"})[1] is True


def test_web_tool_hidden_when_disabled():
    assert "web_search" not in Toolbox(allow_web=False).names
    assert Toolbox(allow_web=False).execute("web_search", {"query": "x"})[1] is True


# --- OpenAI-compatible loop ----------------------------------------------------

def _call(i, name, args):
    return NS(id=f"call_{i}", function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))


def _resp(content=None, calls=None):
    return NS(model="fake-model", usage=NS(prompt_tokens=100, completion_tokens=10),
              choices=[NS(message=NS(content=content, tool_calls=calls or None),
                          finish_reason="tool_calls" if calls else "stop")])


class ScriptedModel:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        nxt = self.script.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.setattr(tools_mod.retrieval, "search", lambda cfg, q, k, *a: HITS)
    monkeypatch.setattr(agent_mod, "LOG_DIR", tmp_path)

    def _run(script, provider="groq", **kw):
        model = ScriptedModel(script)
        monkeypatch.setattr(agent_mod, "LLM_PROVIDER", provider)
        monkeypatch.setattr(agent_mod, "_openai_client", lambda: NS(chat=NS(completions=model)))
        monkeypatch.setattr(agent_mod, "_anthropic_client", lambda: NS(beta=NS(messages=model)))
        return agent_mod.run_agent("What royalty on $300k gross?", **kw), model, tmp_path

    return _run


def test_multi_step_run_with_log_and_citations(run):
    result, model, logdir = run([
        _resp("Let me check the docs.", [_call(1, "search_docs", {"query": "royalty rate"})]),
        _resp(None, [_call(2, "calculator", {"expression": "0.06 * 300000"})]),
        _resp("The royalty is 6% of gross sales [S1], so $18,000 per year."),
    ], max_steps=5)

    assert result.stopped_reason == "answered"
    assert result.answer == "The royalty is 6% of gross sales[1], so $18,000 per year."
    assert [(c.kind, c.file, c.pages) for c in result.citations] == [("doc", "orbit.pdf", [1])]
    assert [s.action for s in result.steps] == ["tool_calls", "tool_calls", "final_answer"]
    assert result.steps[0].message == "Let me check the docs."
    assert result.steps[1].tool_calls[0].tool == "calculator" and result.steps[1].tool_calls[0].output_preview == "18000"
    assert (result.input_tokens, result.output_tokens) == (300, 30)

    # Tool results were fed back with the matching tool_call_id.
    third = model.requests[2]["messages"]
    assert {"role": "tool", "tool_call_id": "call_2", "content": "18000"} in third
    assert all(r["tool_choice"] == "auto" for r in model.requests)

    record = json.loads((logdir / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["run_id"] == result.run_id and len(record["steps"]) == 3


def test_step_limit_forces_final_answer(run):
    always_search = _resp(None, [_call(1, "search_docs", {"query": "royalty"})])
    result, model, _ = run([always_search, _resp("Best effort: 6% [S1].")], max_steps=2)

    assert result.stopped_reason == "step_limit"
    assert result.steps[-1].forced and len(result.steps) == 2
    last = model.requests[-1]
    assert last["tool_choice"] == "none"
    assert last["messages"][-1] == {"role": "user", "content": agent_mod.STEP_LIMIT_NOTE}
    assert result.citations[0].file == "orbit.pdf"


def test_tool_calls_on_last_step_are_ignored(run):
    result, _, _ = run([_resp("Partial answer", [_call(1, "search_docs", {"query": "x"})])], max_steps=1)
    assert result.stopped_reason == "step_limit" and result.answer == "Partial answer"
    assert result.steps[0].tool_calls == []


def test_bad_json_and_unknown_tool_are_logged_as_errors(run):
    result, model, _ = run([
        _resp(None, [_call(1, "search_docs", "{not json"), _call(2, "delete_everything", {})]),
        _resp("I could not find it."),
    ])
    calls = result.steps[0].tool_calls
    assert [c.is_error for c in calls] == [True, True]
    assert "not valid JSON" in calls[0].output_preview and "unknown tool" in calls[1].output_preview
    tool_msgs = [m for m in model.requests[1]["messages"] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2"]  # every call gets a result


def _bad_generation():
    import httpx
    import openai

    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.BadRequestError("output_parse_failed", response=httpx.Response(400, request=req),
                                  body={"error": {"code": "output_parse_failed"}})


def test_malformed_tool_call_is_retried_then_falls_back_to_tools_off(run):
    err = _bad_generation()
    result, model, _ = run([err, err, _resp("Answer from what I know [S1].")], max_steps=4)
    temps_choices = [(r["temperature"], r["tool_choice"]) for r in model.requests]
    assert temps_choices == [(0, "auto"), (0.7, "auto"), (0.7, "none")]
    assert result.steps[0].note == "malformed tool calls twice; answered without tools"
    assert result.stopped_reason == "answered"


def test_hallucinated_labels_dropped(run):
    result, _, _ = run([_resp("Answer [S7].")])
    assert result.answer == "Answer." and result.citations == []


def test_tool_name_brackets_removed(run):
    result, _, _ = run([_resp("18 x 30 = 540 days【calculator】. Checked [search_docs].")])
    assert result.answer == "18 x 30 = 540 days. Checked."


# --- Claude loop -----------------------------------------------------------------

def _claude(stop, *blocks):
    return NS(model="claude-opus-5", stop_reason=stop, usage=NS(input_tokens=50, output_tokens=5), content=list(blocks))


def test_anthropic_loop(run):
    tool_use = NS(type="tool_use", id="tu_1", name="calculator", input={"expression": "0.06*300000"})
    result, model, _ = run([
        _claude("tool_use", NS(type="text", text="Computing."), tool_use),
        _claude("end_turn", NS(type="text", text="$18,000 per year.")),
    ], provider="anthropic")

    assert result.stopped_reason == "answered" and result.answer == "$18,000 per year."
    second = model.requests[1]["messages"]
    assert second[1]["role"] == "assistant"  # full content echoed back, including the tool_use block
    assert second[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "18000", "is_error": False}
    assert model.requests[0]["tool_choice"] == {"type": "auto"}
