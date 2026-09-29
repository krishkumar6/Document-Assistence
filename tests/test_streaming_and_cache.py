"""Streaming, answer cache and cache-token accounting, with fake models (no network)."""

import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from app import agent as agent_mod
from app import answer as answer_mod
from app import main as main_mod
from app import tools as tools_mod
from app.answer_cache import AnswerCache, answer_cache
from app.chunking import PageSpan
from app.store import Hit

TEXT = "Passwords must be at least 14 characters long."
HITS = [Hit(id="p::0", text=TEXT, source="policy.pdf", page_start=2, page_end=2, spans=[PageSpan(2, 0, len(TEXT))], distance=0.1)]


# --- fake OpenAI-compatible streaming ---------------------------------------------

def _chunk(content=None, tool_calls=None, finish=None, usage=None):
    choices = [] if usage else [NS(delta=NS(content=content, tool_calls=tool_calls), finish_reason=finish)]
    return NS(model="fake-model", choices=choices, usage=usage)


def _usage(prompt, cached, completion):
    return NS(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=NS(cached_tokens=cached))


class FakeOpenAI:
    """Replays scripted chunk lists for stream=True calls and counts requests."""

    def __init__(self, *streams):
        self.streams, self.calls = list(streams), []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs.get("stream") and kwargs["stream_options"] == {"include_usage": True}
        return iter(self.streams.pop(0))


def _patch_openai(monkeypatch, fake):
    monkeypatch.setattr(answer_mod, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(answer_mod, "_openai_client", lambda: fake)
    monkeypatch.setattr(agent_mod, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(agent_mod, "_openai_client", lambda: fake)


def test_stream_answer_emits_deltas_then_linked_answer_with_cached_tokens(monkeypatch):
    fake = FakeOpenAI([_chunk("At least 14 "), _chunk("characters "), _chunk("【S1】."), _chunk(finish="stop"),
                       _chunk(usage=_usage(600, 512, 12))])
    _patch_openai(monkeypatch, fake)
    events = list(answer_mod.stream_answer("min length?", HITS))

    assert [e["text"] for e in events if e["type"] == "delta"] == ["At least 14 ", "characters ", "【S1】."]
    done = events[-1]["answer"]
    assert done.answer == "At least 14 characters[1]."
    assert (done.input_tokens, done.cached_input_tokens, done.output_tokens) == (600, 512, 12)
    assert done.citations[0].file == "policy.pdf" and done.citations[0].pages == [2]


def test_answer_cache_skips_the_model_on_repeat(monkeypatch):
    fake = FakeOpenAI([_chunk("14 chars [S1]"), _chunk(finish="stop"), _chunk(usage=_usage(600, 0, 5))])
    _patch_openai(monkeypatch, fake)

    first = [e for e in answer_mod.stream_answer("Min length?", HITS)][-1]["answer"]
    # Same question modulo case/whitespace, same passages -> no second model call (fake has no more streams).
    again = [e for e in answer_mod.stream_answer("  min   LENGTH? ", HITS)]
    second = again[-1]["answer"]

    assert len(fake.calls) == 1 and not first.from_answer_cache and second.from_answer_cache
    assert second.answer == first.answer and second.citations == first.citations
    assert second.input_tokens == 0 and [e["type"] for e in again] == ["done"]
    assert answer_cache.stats()["hits"] == 1


def test_answer_cache_misses_when_passages_change(monkeypatch):
    other = [Hit(id="p::1", text=TEXT + " Updated.", source="policy.pdf", page_start=2, page_end=2,
                 spans=[PageSpan(2, 0, len(TEXT) + 9)], distance=0.1)]
    fake = FakeOpenAI(*[[_chunk("14 [S1]"), _chunk(finish="stop"), _chunk(usage=_usage(1, 0, 1))] for _ in range(2)])
    _patch_openai(monkeypatch, fake)
    list(answer_mod.stream_answer("q", HITS))
    list(answer_mod.stream_answer("q", other))
    assert len(fake.calls) == 2


def test_answer_cache_ttl_and_size():
    c = AnswerCache(max_size=2, ttl_seconds=60)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None and c.get("c") == 3  # oldest evicted
    c.ttl = 0.0
    assert c.get("c") is None and not c.enabled  # expired / disabled
    assert AnswerCache(0, 60).get("x") is None


def test_refusals_and_truncated_answers_are_not_cached(monkeypatch):
    fake = FakeOpenAI(*[[_chunk("partial"), _chunk(finish="length"), _chunk(usage=_usage(1, 0, 1))] for _ in range(2)])
    _patch_openai(monkeypatch, fake)
    list(answer_mod.stream_answer("q", HITS))
    list(answer_mod.stream_answer("q", HITS))
    assert len(fake.calls) == 2


def test_anthropic_usage_totals_include_cache_reads_and_writes():
    u = answer_mod.anthropic_usage(NS(input_tokens=50, cache_read_input_tokens=3000, cache_creation_input_tokens=400, output_tokens=80))
    assert (u.input_tokens, u.cached_input_tokens, u.output_tokens) == (3450, 3000, 80)
    u = answer_mod.anthropic_usage(NS(input_tokens=50, output_tokens=8))  # no cache fields at all
    assert (u.input_tokens, u.cached_input_tokens) == (50, 0)


class FakeClaudeStream:
    """Stands in for `client.beta.messages.stream(...)`: iterable events + get_final_message()."""

    def __init__(self, texts, final):
        self.texts, self.final = texts, final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for t in self.texts:
            yield NS(type="content_block_delta", delta=NS(type="text_delta", text=t))

    def get_final_message(self):
        return self.final


def test_claude_streaming_ask_and_agent_cache_control(monkeypatch, tmp_path):
    cite = NS(type="content_block_location", document_index=0, start_block_index=0, end_block_index=1, cited_text=TEXT)
    final = NS(model="claude-opus-5", stop_reason="end_turn",
               usage=NS(input_tokens=40, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=9),
               content=[NS(type="text", text="At least 14 characters", citations=[cite])])
    requests = []

    def stream(**kw):
        requests.append(kw)
        return FakeClaudeStream(["At least ", "14 characters"], final)

    client = NS(beta=NS(messages=NS(stream=stream)))
    monkeypatch.setattr(answer_mod, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(answer_mod, "_anthropic_client", lambda: client)
    events = list(answer_mod.stream_answer("min?", HITS))
    assert [e["text"] for e in events if e["type"] == "delta"] == ["At least ", "14 characters"]
    assert events[-1]["answer"].answer == "At least 14 characters[1]"
    assert "cache_control" not in requests[0]  # single-use /ask prompt: caching would only add write cost

    monkeypatch.setattr(agent_mod, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(agent_mod, "_anthropic_client", lambda: client)
    monkeypatch.setattr(agent_mod, "LOG_DIR", tmp_path)
    list(agent_mod.agent_events("min?", max_steps=2))
    assert requests[-1]["cache_control"] == {"type": "ephemeral"}  # agent loop re-reads its prefix every turn


# --- agent streaming ---------------------------------------------------------------

def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def test_agent_stream_reassembles_split_tool_calls_and_streams_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(tools_mod.retrieval, "search", lambda cfg, q, k, *a: HITS)
    monkeypatch.setattr(agent_mod, "LOG_DIR", tmp_path)
    fake = FakeOpenAI(
        # step 1: tool call whose JSON arguments arrive in three pieces
        [_chunk(tool_calls=[_tc(0, "call_1", "calculator", '{"expr')]), _chunk(tool_calls=[_tc(0, args='ession": "14')]),
         _chunk(tool_calls=[_tc(0, args=' * 2"}')]), _chunk(finish="tool_calls"), _chunk(usage=_usage(900, 0, 20))],
        # step 2: streamed final answer; prefix now cached
        [_chunk("That is 28 "), _chunk("characters [S9]."), _chunk(finish="stop"), _chunk(usage=_usage(1000, 896, 10))],
    )
    _patch_openai(monkeypatch, fake)
    events = list(agent_mod.agent_events("double the min length", max_steps=4))
    types = [e["type"] for e in events]

    assert types == ["start", "step_start", "tool_call", "tool_result", "step_end", "step_start", "delta", "delta", "step_end", "done"]
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["input"] == {"expression": "14 * 2"}
    assert next(e for e in events if e["type"] == "tool_result")["output_preview"] == "28"
    result = events[-1]["result"]
    assert result.answer == "That is 28 characters."  # hallucinated [S9] dropped
    assert (result.input_tokens, result.cached_input_tokens, result.output_tokens) == (1900, 896, 30)
    assert [s.cached_input_tokens for s in result.steps] == [0, 896]
    # Cache-friendly history: request 2 begins with request 1's messages, unchanged.
    first, second = fake.calls[0]["messages"], fake.calls[1]["messages"]
    assert second[: len(first)] == first


def test_agent_stream_retry_event_on_bad_generation(monkeypatch, tmp_path):
    import openai

    monkeypatch.setattr(agent_mod, "LOG_DIR", tmp_path)

    def broken_stream():
        yield _chunk("I will open the page")
        raise openai.APIError("output_parse_failed", request=None, body=None)

    fake = FakeOpenAI(broken_stream(), [_chunk("Answer."), _chunk(finish="stop"), _chunk(usage=_usage(1, 0, 1))])
    _patch_openai(monkeypatch, fake)
    events = list(agent_mod.agent_events("q", max_steps=3))
    assert [e["type"] for e in events if e["type"] in ("delta", "retry")] == ["delta", "retry", "delta"]
    assert events[-1]["result"].answer == "Answer." and events[-1]["result"].steps[0].note == "malformed tool call; retried"


# --- SSE endpoints -------------------------------------------------------------------

def _sse_events(resp) -> list[dict]:
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_ask_stream_endpoint(monkeypatch):
    monkeypatch.setattr(main_mod.retrieval, "search", lambda cfg, q, k, *a: HITS)
    fake = FakeOpenAI([_chunk("14 [S1]"), _chunk(finish="stop"), _chunk(usage=_usage(10, 0, 2))])
    _patch_openai(monkeypatch, fake)
    resp = TestClient(main_mod.app).post("/ask/stream", json={"question": "min?"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp)
    assert [e["type"] for e in events] == ["retrieved", "delta", "done"]
    assert events[-1]["answer"] == "14[1]" and events[-1]["citations"][0]["pages"] == [2]
    assert events[-1]["retrieved"][0]["file"] == "policy.pdf"


def test_stream_errors_arrive_as_error_events(monkeypatch):
    monkeypatch.setattr(main_mod.retrieval, "search", lambda cfg, q, k, *a: HITS)

    def no_key():
        raise answer_mod.LLMError(503, "groq needs a free API key in GROQ_API_KEY")

    monkeypatch.setattr(answer_mod, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(answer_mod, "_openai_client", no_key)
    events = _sse_events(TestClient(main_mod.app).post("/ask/stream", json={"question": "min?"}))
    assert events[-1] == {"type": "error", "status": 503, "message": "groq needs a free API key in GROQ_API_KEY"}
