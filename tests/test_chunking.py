import pytest

from app.chunking import chunk_pages

PAGES = [
    (1, "Alpha sentence one. Alpha sentence two is a little longer. Alpha three."),
    (2, "Beta page begins here. Beta has the answer: forty two. Beta ends."),
    (3, "Gamma closes the document. The end."),
]


def test_spans_map_back_to_source_pages():
    for c in chunk_pages(PAGES, "doc.pdf", size=60, overlap=15):
        for span in c.spans:
            segment = c.text[span.start : span.end].strip()
            assert segment and segment in dict(PAGES)[span.page]


def test_all_text_is_covered():
    chunks = chunk_pages(PAGES, "doc.pdf", size=60, overlap=15)
    joined = " ".join(c.text for c in chunks)
    for _, text in PAGES:
        for word in text.split():
            assert word in joined


def test_chunk_can_span_pages():
    chunks = chunk_pages(PAGES, "doc.pdf", size=1000, overlap=0)
    assert len(chunks) == 1
    assert [s.page for s in chunks[0].spans] == [1, 2, 3]
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 3)


def test_size_is_respected_and_indices_sequential():
    chunks = chunk_pages(PAGES, "doc.pdf", size=50, overlap=10)
    assert all(len(c.text) <= 50 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_mid_word_starts():
    full = " ".join(t for _, t in PAGES)
    for c in chunk_pages(PAGES, "doc.pdf", size=40, overlap=12)[1:]:
        idx = full.find(c.text)
        assert idx == 0 or full[idx - 1] == " "


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_pages(PAGES, "doc.pdf", size=100, overlap=100)
