import json
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb

from app.chunking import Chunk, PageSpan
from app.config import CHROMA_DIR, ChunkConfig

BATCH = 256


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page_start: int
    page_end: int
    spans: list[PageSpan]
    distance: float | None = None  # cosine distance; None if the hit came from keyword search only
    # How retrieval ranked it, e.g. {"vector_rank": 2, "bm25_rank": 1, "rrf": 0.032, "rerank": 5.1}
    scores: dict = field(default_factory=dict)


@lru_cache
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(cfg: ChunkConfig):
    # Chroma's default embedding function (all-MiniLM-L6-v2, ONNX) runs locally,
    # so ingestion and retrieval need no API key.
    return _client().get_or_create_collection(
        cfg.collection,
        metadata={"hnsw:space": "cosine", "chunk_size": cfg.size, "chunk_overlap": cfg.overlap},
    )


# Bumped on every write so in-memory indexes built from the collection (BM25)
# know to rebuild. Writes from another process are caught by the chunk count.
_versions: dict[str, int] = {}
_versions_lock = threading.Lock()


def version(cfg: ChunkConfig) -> tuple[int, int]:
    return _versions.get(cfg.collection, 0), _collection(cfg).count()


def replace_document(cfg: ChunkConfig, source: str, chunks: list[Chunk]) -> None:
    """Idempotent ingest: drop any previous chunks for this file, then add the new ones."""
    col = _collection(cfg)
    col.delete(where={"source": source})
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        col.add(
            ids=[f"{c.source}::{c.chunk_index}" for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    # Chroma metadata must be scalar, so spans travel as JSON.
                    "spans": json.dumps([[s.page, s.start, s.end] for s in c.spans]),
                }
                for c in batch
            ],
        )
    with _versions_lock:
        _versions[cfg.collection] = _versions.get(cfg.collection, 0) + 1


def _hit(id_: str, text: str, meta: dict, distance: float | None = None) -> Hit:
    return Hit(
        id=id_,
        text=text,
        source=meta["source"],
        page_start=meta["page_start"],
        page_end=meta["page_end"],
        spans=[PageSpan(*s) for s in json.loads(meta["spans"])],
        distance=distance,
    )


def vector_search(cfg: ChunkConfig, query: str, k: int) -> list[Hit]:
    """Semantic search: nearest chunks by embedding. Use retrieval.search for the full pipeline."""
    col = _collection(cfg)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[query], n_results=min(k, col.count()))
    return [
        _hit(id_, text, meta, dist)
        for id_, text, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def all_chunks(cfg: ChunkConfig) -> list[Hit]:
    """Every chunk in the collection (for building the keyword index)."""
    res = _collection(cfg).get(include=["documents", "metadatas"])
    return [_hit(id_, text, meta) for id_, text, meta in zip(res["ids"], res["documents"], res["metadatas"])]


def stats(cfg: ChunkConfig) -> dict:
    col = _collection(cfg)
    sources = {m["source"] for m in col.get(include=["metadatas"])["metadatas"]}
    return {"collection": cfg.collection, "chunks": col.count(), "documents": sorted(sources)}
