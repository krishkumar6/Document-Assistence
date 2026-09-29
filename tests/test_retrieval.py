"""BM25, fusion and reranking tests (fake vector store and fake cross-encoder; offline)."""

import pytest

from app import retrieval
from app import store as store_mod
from app.chunking import PageSpan
from app.config import CHUNK_CONFIGS
from app.retrieval import BM25Index, rrf_fuse, tokenize
from app.store import Hit

CFG = CHUNK_CONFIGS["small"]


def _hit(i, text, distance=None):
    return Hit(id=f"doc::{i}", text=text, source="doc.pdf", page_start=1, page_end=1,
               spans=[PageSpan(1, 0, len(text))], distance=distance)


DOCS = [
    _hit(0, "The report is filed on form NW-IR-3 in the crew app."),
    _hit(1, "Batteries are kept at a storage charge of 3.85 volts per cell."),
    _hit(2, "Restricted data must be encrypted at rest with AES-256."),
    _hit(3, "Passwords must be at least 14 characters long."),
    _hit(4, "Pilots report every incident to the safety officer within 2 hours."),
]


def test_tokenize_keeps_codes_and_their_parts():
    toks = tokenize("File form NW-IR-3; charge 3.85 V; AES-256 encryption; Passwords")
    assert {"nw-ir-3", "nw", "ir", "3", "3.85", "85", "aes-256", "aes", "256", "password"} <= set(toks)
    assert "the" not in tokenize("the form")


def test_bm25_ranks_exact_terms_first():
    index = BM25Index(DOCS)
    assert index.search("which form is NW-IR-3", 3)[0].id == "doc::0"
    assert index.search("AES 256", 3)[0].id == "doc::2"  # matches "AES-256" via its parts
    assert index.search("3.85 volts", 3)[0].id == "doc::1"
    top = index.search("password length", 3)[0]
    assert top.id == "doc::3" and top.scores["bm25_rank"] == 1
    assert index.search("zebra", 3) == []  # no shared terms, no results


def test_rrf_rewards_agreement_between_retrievers():
    a, b, c = DOCS[0], DOCS[1], DOCS[2]
    vector = [
        Hit(**{**vars(a), "distance": 0.2, "scores": {"vector_rank": 1}}),
        Hit(**{**vars(b), "distance": 0.3, "scores": {"vector_rank": 2}}),
    ]
    keyword = [Hit(**{**vars(b), "scores": {"bm25_rank": 1}}), Hit(**{**vars(c), "scores": {"bm25_rank": 2}})]
    fused = rrf_fuse(vector, keyword)
    assert [h.id for h in fused] == ["doc::1", "doc::0", "doc::2"]  # b is in both lists
    assert fused[0].scores["vector_rank"] == 2 and fused[0].scores["bm25_rank"] == 1
    assert fused[0].distance == 0.3  # vector distance kept through fusion


@pytest.fixture
def fake_store(monkeypatch):
    # Vector search that is (deliberately) bad at exact codes: returns docs in reverse order.
    monkeypatch.setattr(store_mod, "vector_search", lambda cfg, q, k: [_hit(h.id[-1], h.text, 0.5) for h in DOCS[::-1][:k]])
    monkeypatch.setattr(store_mod, "all_chunks", lambda cfg: DOCS)
    monkeypatch.setattr(store_mod, "version", lambda cfg: (1, len(DOCS)))
    retrieval._bm25_cache.clear()


def test_modes(fake_store):
    q = "form NW-IR-3"
    assert retrieval.search(CFG, q, 1, "vector", False)[0].id == "doc::4"
    assert retrieval.search(CFG, q, 1, "bm25", False)[0].id == "doc::0"
    hybrid = retrieval.search(CFG, q, 5, "hybrid", False)
    assert "doc::0" in [h.id for h in hybrid[:2]] and "rrf" in hybrid[0].scores
    with pytest.raises(ValueError):
        retrieval.search(CFG, q, 3, "fuzzy", False)


def test_rerank_reorders_candidates(fake_store, monkeypatch):
    class FakeCrossEncoder:
        def rerank(self, query, docs):
            return [10.0 if "NW-IR-3" in d else -float(i) for i, d in enumerate(docs)]

    monkeypatch.setattr(retrieval, "_get_reranker", lambda: FakeCrossEncoder())
    hits = retrieval.search(CFG, "incident form", 2, "vector", True)
    assert hits[0].id == "doc::0" and hits[0].scores["rerank"] == 10.0
    assert hits[0].scores["vector_rank"] == 5  # was last by vector search, first after reranking


def test_bm25_index_rebuilds_when_documents_change(fake_store, monkeypatch):
    first = retrieval.bm25_index(CFG)
    assert retrieval.bm25_index(CFG) is first  # cached
    monkeypatch.setattr(store_mod, "version", lambda cfg: (2, len(DOCS)))
    assert retrieval.bm25_index(CFG) is not first
