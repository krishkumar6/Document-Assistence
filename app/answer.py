"""Turn retrieved chunks into an answer with file + page citations.

Two backends:
- anthropic: Claude's native citations (exact quoted spans, mapped to pages).
- ollama / groq / gemini: any OpenAI-compatible model. Each page excerpt is
  labelled [S1], [S2]... in the prompt, the model cites those labels, and the
  labels are validated and mapped back to file + page here.

Each backend has a blocking and a streaming entry point; both share prompt
building, parsing and error handling, so they return identical answers.

Caching:
- Prompts are laid out stable-part-first (fixed instructions, then sources, then
  the question) so provider-side prompt caching (automatic on Groq and Ollama)
  can reuse the longest possible prefix.
- /ask answers are also kept in a local answer cache (app/answer_cache.py):
  same question + same retrieved passages + same model -> no LLM call at all.
"""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache

from app.answer_cache import answer_cache
from app.config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, OPENAI_COMPATIBLE
from app.store import Hit

NOT_FOUND = "I could not find this in the documents."

CLAUDE_SYSTEM = (
    "You answer questions using only the documents provided in the user turn. "
    "Each document is an excerpt from a PDF; its title gives the file name. "
    "If the excerpts do not contain the answer, say you could not find it in the documents "
    "rather than drawing on outside knowledge. Keep answers short and direct."
)

OPENAI_SYSTEM = (
    "You answer questions using only the numbered sources you are given. "
    "After every sentence that uses a source, cite it in square brackets, e.g. [S2] or [S1, S3]. "
    "Only cite source labels that appear in the list. Do not use outside knowledge. "
    f'If the sources do not contain the answer, reply exactly: "{NOT_FOUND}" '
    "Keep answers short and direct, in plain language with no LaTeX or math markup."
)


class LLMError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Citation:
    id: int
    file: str | None  # None for web citations
    pages: list[int]
    quotes: list[str] = field(default_factory=list)
    kind: str = "doc"  # "doc" or "web" (agent only)
    url: str | None = None
    title: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0  # total prompt tokens, cached or not
    cached_input_tokens: int = 0  # part of input_tokens served from the provider's prompt cache (if reported)
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class Answer:
    answer: str
    citations: list[Citation]
    stop_reason: str | None
    model: str | None
    provider: str = LLM_PROVIDER
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    from_answer_cache: bool = False  # True: served from the local answer cache, no LLM call made


def _with_usage(answer: Answer, usage: Usage) -> Answer:
    return replace(answer, input_tokens=usage.input_tokens, cached_input_tokens=usage.cached_input_tokens,
                   output_tokens=usage.output_tokens)


# --- public entry points ---------------------------------------------------------

def answer_question(question: str, hits: list[Hit]) -> Answer:
    for event in stream_answer(question, hits, stream=False):
        if event["type"] == "done":
            return event["answer"]
    raise AssertionError("stream_answer always ends with a done event")


def stream_answer(question: str, hits: list[Hit], stream: bool = True) -> Iterator[dict]:
    """Yield {"type": "delta", "text"} events while the model writes (if `stream`),
    then {"type": "done", "answer": Answer}. Delta text is raw model output; the
    final answer has citation labels resolved to [n] markers."""
    if not hits:
        yield {"type": "done", "answer": Answer("No documents have been ingested yet.", [], None, None)}
        return

    key = answer_cache.key(LLM_PROVIDER, LLM_MODEL, question, hits)
    cached = answer_cache.get(key)
    if cached is not None:
        yield {"type": "done", "answer": replace(cached, from_answer_cache=True, input_tokens=0,
                                                 cached_input_tokens=0, output_tokens=0)}
        return

    if LLM_PROVIDER == "anthropic":
        answer = yield from _claude(question, hits, stream)
    elif LLM_PROVIDER in OPENAI_COMPATIBLE or LLM_BASE_URL:
        answer = yield from _openai_compatible(question, hits, stream)
    else:
        raise LLMError(500, f"Unknown LLM_PROVIDER {LLM_PROVIDER!r}; use anthropic, {', '.join(OPENAI_COMPATIBLE)}")

    if answer.stop_reason not in (None, "refusal", "max_tokens", "length"):
        answer_cache.put(key, answer)
    yield {"type": "done", "answer": answer}


# --- shared helpers ----------------------------------------------------------------

def _segments(hit: Hit) -> list[tuple[int, str]]:
    """(page, text) for each non-empty page segment of a chunk."""
    out = []
    for span in hit.spans:
        text = hit.text[span.start : span.end].strip()
        if text:
            out.append((span.page, text))
    return out


class _Citations:
    """Numbers each distinct (file, pages) once, in order of first citation."""

    def __init__(self):
        self.by_key: dict[tuple, Citation] = {}

    def add(self, file: str, pages: tuple[int, ...], quote: str) -> Citation:
        key = (file, pages)
        if key not in self.by_key:
            self.by_key[key] = Citation(id=len(self.by_key) + 1, file=file, pages=list(pages))
        return self._quote(self.by_key[key], quote)

    def add_web(self, url: str, title: str, snippet: str) -> Citation:
        key = ("web", url)
        if key not in self.by_key:
            self.by_key[key] = Citation(id=len(self.by_key) + 1, file=None, pages=[], kind="web", url=url, title=title)
        return self._quote(self.by_key[key], snippet)

    @staticmethod
    def _quote(cite: Citation, quote: str) -> Citation:
        quote = quote.strip()
        if quote and quote not in cite.quotes:
            cite.quotes.append(quote)
        return cite

    def list(self) -> list[Citation]:
        return list(self.by_key.values())


def _excerpt(text: str, limit: int = 300) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")


# --- Claude (native citations) -------------------------------------------------

@lru_cache
def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


@contextmanager
def anthropic_errors():
    """Translate Anthropic SDK failures into LLMError (also while iterating a stream)."""
    import anthropic

    try:
        yield
    except TypeError as e:
        # The SDK raises TypeError (not an API error) when no credentials resolve at all.
        if "authentication" in str(e):
            raise LLMError(503, "Anthropic credentials missing (set ANTHROPIC_API_KEY, or LLM_PROVIDER=ollama for a free local model)") from e
        raise
    except anthropic.AuthenticationError as e:
        raise LLMError(503, "Anthropic API key is invalid") from e
    except anthropic.RateLimitError as e:
        raise LLMError(429, "Rate limited by the Anthropic API; retry shortly") from e
    except anthropic.APIStatusError as e:
        raise LLMError(502, f"Anthropic API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError(502, "Could not reach the Anthropic API") from e


def anthropic_usage(u) -> Usage:
    """Anthropic's input_tokens is only the uncached remainder; the total prompt is the sum."""
    read = getattr(u, "cache_read_input_tokens", None) or 0
    written = getattr(u, "cache_creation_input_tokens", None) or 0
    return Usage(u.input_tokens + read + written, read, u.output_tokens)


def _claude_documents(hits: list[Hit]) -> tuple[list[dict], list[list[int]]]:
    """One document per retrieved chunk, one text block per page segment.

    Claude's content_block_location citations point at block indices, so
    `block_pages[doc][block]` turns a citation back into an exact page.
    """
    docs, block_pages = [], []
    for hit in hits:
        segs = _segments(hit)
        pages_label = f"p. {hit.page_start}" if hit.page_start == hit.page_end else f"pp. {hit.page_start}-{hit.page_end}"
        docs.append(
            {
                "type": "document",
                "source": {"type": "content", "content": [{"type": "text", "text": t} for _, t in segs]},
                "title": f"{hit.source} ({pages_label})",
                "citations": {"enabled": True},
            }
        )
        block_pages.append([p for p, _ in segs])
    return docs, block_pages


def _claude(question: str, hits: list[Hit], stream: bool):
    docs, block_pages = _claude_documents(hits)
    # No cache_control here: each /ask prompt holds different passages, so a cache
    # write (1.25x input price) would rarely be read back. Repeat questions are
    # handled by the local answer cache instead. The agent loop does use caching.
    kwargs = dict(
        model=LLM_MODEL,
        max_tokens=16000,
        # If a safety classifier declines, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": [*docs, {"type": "text", "text": question}]}],
    )
    client = _anthropic_client()
    with anthropic_errors():
        if stream:
            with client.beta.messages.stream(**kwargs) as s:
                for event in s:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield {"type": "delta", "text": event.delta.text}
                response = s.get_final_message()
        else:
            response = client.beta.messages.create(**kwargs)
    return _parse_claude(response, hits, block_pages)


def _parse_claude(response, hits: list[Hit], block_pages: list[list[int]]) -> Answer:
    usage = anthropic_usage(response.usage)
    if response.stop_reason == "refusal":
        return _with_usage(Answer("The model declined to answer this question.", [], response.stop_reason, response.model), usage)

    # Citations come back attached to text blocks; stitch the text together and
    # add [n] markers after each cited span.
    parts: list[str] = []
    cites = _Citations()
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        markers = []
        for c in block.citations or []:
            if c.type != "content_block_location":
                continue
            pages = tuple(sorted(set(block_pages[c.document_index][c.start_block_index : c.end_block_index])))
            cite = cites.add(hits[c.document_index].source, pages, c.cited_text)
            if f"[{cite.id}]" not in markers:
                markers.append(f"[{cite.id}]")
        parts.extend(markers)

    return _with_usage(Answer("".join(parts).strip(), cites.list(), response.stop_reason, response.model), usage)


# --- OpenAI-compatible (Ollama, Groq, Gemini) ------------------------------------

@lru_cache
def _openai_client():
    from openai import OpenAI

    key_env = OPENAI_COMPATIBLE.get(LLM_PROVIDER, (None, None, None))[2]
    api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None) or "not-needed"
    if key_env and api_key == "not-needed":
        raise LLMError(503, f"{LLM_PROVIDER} needs a free API key in {key_env}")
    # Free tiers rate-limit hard; the SDK retries 429s with backoff, honouring retry-after.
    return OpenAI(base_url=LLM_BASE_URL, api_key=api_key, timeout=300, max_retries=1 if LLM_PROVIDER == "ollama" else 4)


@contextmanager
def openai_errors():
    """Translate OpenAI-SDK failures into LLMError (also while iterating a stream).

    BadRequestError passes through untouched so callers can retry malformed generations.
    """
    import openai

    try:
        yield
    except openai.BadRequestError:
        raise
    except openai.APIConnectionError as e:
        hint = " Is Ollama running? Start it with `ollama serve`." if LLM_PROVIDER == "ollama" else ""
        raise LLMError(503, f"Could not reach {LLM_PROVIDER} at {LLM_BASE_URL}.{hint}") from e
    except openai.AuthenticationError as e:
        raise LLMError(503, f"{LLM_PROVIDER} rejected the API key") from e
    except openai.RateLimitError as e:
        raise LLMError(429, f"{LLM_PROVIDER} free-tier rate limit hit; retry shortly") from e
    except openai.NotFoundError as e:
        hint = f" Run `ollama pull {LLM_MODEL}`." if LLM_PROVIDER == "ollama" else ""
        raise LLMError(502, f"Model {LLM_MODEL!r} not found on {LLM_PROVIDER}.{hint}") from e
    except openai.APIStatusError as e:
        raise LLMError(502, f"{LLM_PROVIDER} error ({e.status_code}): {e.message}") from e


def openai_usage(u) -> Usage:
    if u is None:
        return Usage()
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
    return Usage(u.prompt_tokens or 0, cached, u.completion_tokens or 0)


def stream_openai_turn(client, **kwargs):
    """One streamed chat completion. Yields {"type": "delta"} events and returns
    (content, tool_calls, finish_reason, model, Usage) once the stream ends.
    tool_calls are dicts {id, name, arguments} reassembled from the deltas."""
    content: list[str] = []
    calls: dict[int, dict] = {}
    finish = model = None
    usage = Usage()
    for chunk in client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kwargs):
        model = chunk.model or model
        if chunk.usage:
            usage = openai_usage(chunk.usage)
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            content.append(delta.content)
            yield {"type": "delta", "text": delta.content}
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish = choice.finish_reason
    return "".join(content), [calls[i] for i in sorted(calls)], finish, model, usage


# A bracket that contains only source labels: [S1], [S1, S3], [Source 2], [2].
# gpt-oss models write full-width 【S1】 by habit, so accept those brackets too.
_TAG = re.compile(r"[\[【]\s*((?:S(?:ource)?\s*)?\d+(?:\s*[,;]\s*(?:S(?:ource)?\s*)?\d+)*)\s*[\]】]", re.IGNORECASE)


def _numbered_sources(hits: list[Hit]) -> list[tuple[str, int, str]]:
    """Flatten chunks into (file, page, text) sources, one per page segment, deduplicated."""
    sources, seen = [], set()
    for hit in hits:
        for page, text in _segments(hit):
            if (hit.source, text) not in seen:
                seen.add((hit.source, text))
                sources.append((hit.source, page, text))
    return sources


def _link_tags(text: str, sources: list[tuple[str, int, str]]) -> tuple[str, list[Citation]]:
    """Replace [S#] labels with [n] citation markers; drop labels that don't exist."""
    cites = _Citations()

    def resolve(idx: int) -> int | None:
        if not 0 <= idx < len(sources):
            return None
        file, page, src = sources[idx]
        return cites.add(file, (page,), _excerpt(src)).id

    return link_labels(text, resolve), cites.list()


def link_labels(text: str, resolve) -> str:
    """Rewrite source labels ([S3], [S1, S2], 【S4】) as [n] citation markers.

    `resolve(zero_based_source_index)` returns the citation id to show, or None
    if no such source exists (a hallucinated label, which is dropped).
    """

    def repl(m: re.Match) -> str:
        markers = []
        for num in re.findall(r"\d+", m.group(1)):
            cid = resolve(int(num) - 1)
            if cid is not None and f"[{cid}]" not in markers:
                markers.append(f"[{cid}]")
        if markers:
            return "\0" + "".join(markers)  # \0 marks our insertions for the tidy-up below
        # Invalid "S" labels are hallucinated citations: drop them. Bare numbers
        # that match no source ("[2023]") are probably just text: keep them.
        return "\0" if re.search(r"s", m.group(1), re.IGNORECASE) else m.group(0)

    # [^\S\r\n] = any space except line breaks, incl. the narrow no-break
    # spaces (U+202F) gpt-oss puts before brackets.
    linked = _TAG.sub(repl, text)
    linked = re.sub(r"[^\S\r\n]*\0", "", linked)  # "word [S1]" -> "word[1]"
    linked = re.sub(r"(\[\d+\])[^\S\r\n]+(?=[.,;:])", r"\1", linked)  # "[1] ." -> "[1]."
    linked = re.sub(r"(\[\d+\])\1+", r"\1", linked)  # overlapping chunks -> "[1][1]" -> "[1]"
    # Collapse doubled spaces left by removed labels, but not leading indentation (nested lists).
    return re.sub(r"(?<=\S)[^\S\r\n]{2,}", " ", linked).strip()


def _openai_compatible(question: str, hits: list[Hit], stream: bool):
    import openai

    sources = _numbered_sources(hits)
    context = "\n\n".join(f"[S{i}] ({file}, page {page})\n{text}" for i, (file, page, text) in enumerate(sources, 1))
    kwargs = dict(
        model=LLM_MODEL,
        temperature=0,
        # Stable-first layout for prefix caching: fixed instructions, then sources, then the question.
        messages=[
            {"role": "system", "content": OPENAI_SYSTEM},
            {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    client = _openai_client()
    with openai_errors():
        try:
            if stream:
                text, _, finish, model, usage = yield from stream_openai_turn(client, **kwargs)
            else:
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                text, finish, model, usage = choice.message.content or "", choice.finish_reason, response.model, openai_usage(response.usage)
        except openai.BadRequestError as e:
            raise LLMError(502, f"{LLM_PROVIDER} rejected the request: {e.message}") from e

    linked, citations = _link_tags(text, sources)
    return _with_usage(Answer(linked, citations, finish, model), usage)
