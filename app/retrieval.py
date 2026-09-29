"""Retrieval pipeline: vector search, BM25 keyword search, hybrid fusion, reranking.

    search(cfg, query, k, mode="hybrid", rerank=True)

- vector: semantic similarity (Chroma + MiniLM embeddings). Good at paraphrases
  ("how long are logs kept" ~ "retained for 18 months").
- bm25:   keyword relevance. Good at exact terms the embedding blurs: codes
  ("NW-IR-3"), numbers ("3.85"), names, rare words.
- hybrid: both lists merged with Reciprocal Rank Fusion (RRF), which combines
  ranks rather than raw scores, so the two scales never need calibrating.
- rerank: a cross-encoder reads the query and each candidate together and
  re-scores them. More accurate than either retriever alone, but it scores
  pairs one by one, so it only runs on a short candidate pool.

Everything runs locally and free (ONNX models, no API keys).
"""

import logging
import math
import os
import re
import threading
from collections import Counter
from dataclasses import replace

from app import store
from app.config import (
    CANDIDATE_POOL,
    MODEL_CACHE_DIR,
    RERANK_MODEL,
    RETRIEVAL_MODE,
    RETRIEVAL_RERANK,
    ChunkConfig,
)
from app.store import Hit

log = logging.getLogger(__name__)

MODES = ("vector", "bm25", "hybrid")
RRF_K = 60  # standard RRF damping constant: 1 / (60 + rank)


def search(cfg: ChunkConfig, query: str, k: int, mode: str | None = None, rerank: bool | None = None) -> list[Hit]:
    mode = mode or RETRIEVAL_MODE
    rerank = RETRIEVAL_RERANK if rerank is None else rerank
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    # Candidates fetched from each retriever for fusion / reranking. The reranker
    # scores every candidate, so this is the main latency knob.
    pool = max(CANDIDATE_POOL, 2 * k) if (rerank or mode == "hybrid") else k

    if mode == "vector":
        hits = _ranked(store.vector_search(cfg, query, pool), "vector_rank")
    elif mode == "bm25":
        hits = bm25_index(cfg).search(query, pool)
    else:
        hits = rrf_fuse(_ranked(store.vector_search(cfg, query, pool), "vector_rank"), bm25_index(cfg).search(query, pool))

    if rerank and hits:
        hits = rerank_hits(query, hits[:pool])
    return hits[:k]


def _ranked(hits: list[Hit], key: str) -> list[Hit]:
    return [replace(h, scores={**h.scores, key: i}) for i, h in enumerate(hits, 1)]


def rrf_fuse(*ranked_lists: list[Hit]) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (RRF_K + rank of d)."""
    fused: dict[str, Hit] = {}
    totals: dict[str, float] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits, 1):
            totals[h.id] = totals.get(h.id, 0.0) + 1.0 / (RRF_K + rank)
            if h.id in fused:  # merge rank info, keep the vector distance if either side has it
                prev = fused[h.id]
                fused[h.id] = replace(prev, distance=prev.distance if prev.distance is not None else h.distance,
                                      scores={**prev.scores, **h.scores})
            else:
                fused[h.id] = h
    order = sorted(fused, key=lambda i: totals[i], reverse=True)
    return [replace(fused[i], scores={**fused[i].scores, "rrf": round(totals[i], 5)}) for i in order]


# --- BM25 ------------------------------------------------------------------------

_STOP = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was were what "
    "when where which who why will with do does did can could should would their there they them than then "
    "so if about into out up any all per each my your our".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens that keep codes and decimals whole ("nw-ir-3", "3.85", "aes-256")
    and also index their parts, so "AES 256" still matches "AES-256"."""
    out = []
    for tok in _TOKEN.findall(text.lower()):
        parts = re.split(r"[.\-]", tok) if ("-" in tok or "." in tok) else []
        for t in [tok, *parts]:
            if t and t not in _STOP:
                out.append(_stem(t))
    return out


def _stem(t: str) -> str:
    """Tiny plural folding (passwords -> password, batteries -> battery); leaves numbers alone."""
    if t.isdigit() or len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


class BM25Index:
    """Okapi BM25 over one collection's chunks (k1=1.5, b=0.75)."""

    def __init__(self, hits: list[Hit], k1: float = 1.5, b: float = 0.75):
        self.hits, self.k1, self.b = hits, k1, b
        self.docs = [Counter(tokenize(h.text)) for h in hits]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if hits else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(hits)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]
        if not terms:
            return []
        scored = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for t in terms:
                f = doc.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [replace(self.hits[i], scores={"bm25_rank": r, "bm25": round(s, 3)}) for r, (s, i) in enumerate(scored[:k], 1)]


_bm25_cache: dict[str, tuple[tuple, BM25Index]] = {}
_bm25_lock = threading.Lock()


def bm25_index(cfg: ChunkConfig) -> BM25Index:
    """Built from the collection on first use, rebuilt when documents change."""
    ver = store.version(cfg)
    cached = _bm25_cache.get(cfg.collection)
    if cached and cached[0] == ver:
        return cached[1]
    with _bm25_lock:
        cached = _bm25_cache.get(cfg.collection)
        if cached and cached[0] == ver:
            return cached[1]
        index = BM25Index(store.all_chunks(cfg))
        _bm25_cache[cfg.collection] = (ver, index)
        log.info("built BM25 index for %s: %d chunks", cfg.collection, len(index.hits))
        return index


# --- reranker --------------------------------------------------------------------

_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    """Load the cross-encoder once (first call downloads it, ~80 MB for the default)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _reranker = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=str(MODEL_CACHE_DIR))
    return _reranker


def rerank_hits(query: str, hits: list[Hit]) -> list[Hit]:
    scores = list(_get_reranker().rerank(query, [h.text for h in hits]))
    ranked = sorted(zip(scores, hits), key=lambda p: p[0], reverse=True)
    return [replace(h, scores={**h.scores, "rerank": round(float(s), 3)}) for s, h in ranked]
