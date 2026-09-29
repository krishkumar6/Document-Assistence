"""Citation mapping tests with a stubbed Anthropic client (no API calls)."""

from types import SimpleNamespace as NS

from app import answer as answer_mod
from app.chunking import PageSpan
from app.store import Hit


def _hit(source, text, spans):
    return Hit(id=f"{source}::0", text=text, source=source, page_start=spans[0].page,
               page_end=spans[-1].page, spans=spans, distance=0.1)


# Chunk spanning pages 3 and 4: block 0 -> page 3, block 1 -> page 4.
TEXT = "Batteries retire at 300 cycles. Incidents are reported within 2 hours."
HITS = [
    _hit("manual.pdf", TEXT, [PageSpan(3, 0, 31), PageSpan(4, 32, len(TEXT))]),
    _hit("policy.pdf", "Passwords need 14 characters.", [PageSpan(2, 0, 29)]),
]


def _cite(doc, start, end, quote):
    return NS(type="content_block_location", document_index=doc, start_block_index=start,
              end_block_index=end, cited_text=quote)


class FakeMessages:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _run(monkeypatch, content, stop_reason="end_turn"):
    response = NS(content=content, stop_reason=stop_reason, model="claude-opus-5",
                  usage=NS(input_tokens=100, output_tokens=20))
    fake = FakeMessages(response)
    monkeypatch.setattr(answer_mod, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(answer_mod, "_anthropic_client", lambda: NS(beta=NS(messages=fake)))
    return answer_mod.answer_question("q?", HITS), fake.kwargs


def test_citations_map_to_exact_page(monkeypatch):
    content = [
        NS(type="text", text="Report within ", citations=None),
        NS(type="text", text="2 hours", citations=[_cite(0, 1, 2, "Incidents are reported within 2 hours.")]),
        NS(type="text", text="; passwords need ", citations=None),
        NS(type="text", text="14 characters", citations=[_cite(1, 0, 1, "Passwords need 14 characters.")]),
        NS(type="text", text=". Again ", citations=None),
        NS(type="text", text="2 hours", citations=[_cite(0, 1, 2, "Incidents are reported within 2 hours.")]),
    ]
    ans, kwargs = _run(monkeypatch, content)

    assert [(c.id, c.file, c.pages) for c in ans.citations] == [(1, "manual.pdf", [4]), (2, "policy.pdf", [2])]
    assert ans.answer == "Report within 2 hours[1]; passwords need 14 characters[2]. Again 2 hours[1]"
    assert ans.citations[0].quotes == ["Incidents are reported within 2 hours."]

    # One document per hit, one text block per page segment, titled with file + pages.
    docs = [b for b in kwargs["messages"][0]["content"] if b["type"] == "document"]
    assert docs[0]["title"] == "manual.pdf (pp. 3-4)"
    assert len(docs[0]["source"]["content"]) == 2
    assert all(d["citations"] == {"enabled": True} for d in docs)


def test_refusal_is_not_parsed(monkeypatch):
    ans, _ = _run(monkeypatch, [], stop_reason="refusal")
    assert ans.citations == [] and ans.stop_reason == "refusal"


def test_no_hits_skips_api(monkeypatch):
    boom = lambda: (_ for _ in ()).throw(AssertionError("called API"))  # noqa: E731
    monkeypatch.setattr(answer_mod, "_anthropic_client", boom)
    monkeypatch.setattr(answer_mod, "_openai_client", boom)
    assert answer_mod.answer_question("q?", []).citations == []


# --- OpenAI-compatible backend (Ollama / Groq / Gemini) -------------------------

class FakeCompletions:
    def __init__(self, text):
        self.text, self.kwargs = text, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return NS(model="qwen2.5:3b", usage=NS(prompt_tokens=300, completion_tokens=20),
                  choices=[NS(finish_reason="stop", message=NS(content=self.text))])


def _run_openai(monkeypatch, text):
    fake = FakeCompletions(text)
    monkeypatch.setattr(answer_mod, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(answer_mod, "_openai_client", lambda: NS(chat=NS(completions=fake)))
    return answer_mod.answer_question("q?", HITS), fake.kwargs


def test_source_labels_map_to_file_and_page(monkeypatch):
    # Sources: S1 = manual p3, S2 = manual p4, S3 = policy p2
    ans, kwargs = _run_openai(monkeypatch, "Report within 2 hours [S2]. Passwords need 14 chars [S3, S2].")
    assert [(c.id, c.file, c.pages) for c in ans.citations] == [(1, "manual.pdf", [4]), (2, "policy.pdf", [2])]
    assert ans.answer == "Report within 2 hours[1]. Passwords need 14 chars[2][1]."
    prompt = kwargs["messages"][1]["content"]
    assert "[S2] (manual.pdf, page 4)" in prompt and "[S3] (policy.pdf, page 2)" in prompt


def test_label_variants_and_hallucinated_labels(monkeypatch):
    ans, _ = _run_openai(monkeypatch, "Retire at 300 cycles [Source 1]. Also [S9] and [1].")
    assert [(c.file, c.pages) for c in ans.citations] == [("manual.pdf", [3])]
    assert ans.answer == "Retire at 300 cycles[1]. Also and[1]."  # [S9] doesn't exist -> dropped


def test_fullwidth_brackets_from_gpt_oss(monkeypatch):
    ans, _ = _run_openai(monkeypatch, "Passwords need 14 characters【S3】【S2】. Again【S3】【S3】.")
    assert [(c.file, c.pages) for c in ans.citations] == [("policy.pdf", [2]), ("manual.pdf", [4])]
    assert ans.answer == "Passwords need 14 characters[1][2]. Again[1]."


def test_narrow_nbsp_before_label_and_list_indentation(monkeypatch):
    ans, _ = _run_openai(monkeypatch, "- Fee:\n  * **$45,000** 【S1】\n  * Other  text")
    assert ans.answer == "- Fee:\n  * **$45,000**[1]\n  * Other text"


def test_non_citation_brackets_untouched(monkeypatch):
    ans, _ = _run_openai(monkeypatch, "See [the appendix] from [2023] and [S1].")
    assert ans.answer == "See [the appendix] from [2023] and[1]."
