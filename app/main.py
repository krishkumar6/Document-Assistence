import json
import logging
import re
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app import retrieval, security, store
from app.agent import agent_events, run_agent
from app.answer import LLMError, answer_question, stream_answer
from app.answer_cache import answer_cache
from app.config import (
    CHUNK_CONFIGS,
    DEFAULT_CHUNK_CONFIG,
    DOCS_DIR,
    LLM_MODEL,
    LLM_PROVIDER,
    MAX_FILES_PER_UPLOAD,
    MAX_PDF_PAGES,
    MAX_UPLOAD_MB,
    RERANK_MODEL,
    RETRIEVAL_MODE,
    RETRIEVAL_RERANK,
)
from app.ingest import ingest_paths
from app.seed import seed_if_empty

security.check_startup_config()  # fail fast rather than start publicly without a password

# Show the agent's per-step log lines in the server console.
_agent_log = logging.getLogger("agent")
if not _agent_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("AGENT %(message)s"))
    _agent_log.addHandler(_handler)
    _agent_log.setLevel(logging.INFO)
    _agent_log.propagate = False

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fresh container on a host that wipes its disk: restore the permanent library first.
    seed_if_empty()
    # Load the models and the BM25 index in the background at startup (10-20s on a
    # small CPU) so the first user's search doesn't pay for it. Failure here is
    # non-fatal: search tries again, and reports the error, on first use.
    def preload():
        try:
            for cfg in CHUNK_CONFIGS.values():
                store.vector_search(cfg, "warm up", 1)  # loads the embedding model
                retrieval.bm25_index(cfg)
            if RETRIEVAL_RERANK:
                retrieval._get_reranker()
        except Exception:
            logging.getLogger("uvicorn.error").exception("model preload failed")

    threading.Thread(target=preload, daemon=True).start()
    yield


app = FastAPI(
    title="Document Assistant API",
    description="Developer reference. For the simple web page, open [/](/).",
    lifespan=lifespan,
)
app.middleware("http")(security.security_middleware)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def home():
    """The simple web page for everyday users (upload PDFs, ask questions)."""
    return FileResponse(STATIC_DIR / "index.html")


SearchMode = Literal["vector", "bm25", "hybrid"]
_MODE_HELP = f"vector, bm25 or hybrid (both, fused). Default: {RETRIEVAL_MODE}"
_RERANK_HELP = f"Re-score candidates with a cross-encoder. Default: {RETRIEVAL_RERANK}"


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    chunk_config: str = DEFAULT_CHUNK_CONFIG
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: SearchMode | None = Field(default=None, description=_MODE_HELP)
    rerank: bool | None = Field(default=None, description=_RERANK_HELP)


class CompareRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: SearchMode | None = Field(default=None, description=_MODE_HELP)
    rerank: bool | None = Field(default=None, description=_RERANK_HELP)


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    max_steps: int = Field(default=6, ge=1, le=12, description="Model turns allowed; the last one must answer")
    chunk_config: str = DEFAULT_CHUNK_CONFIG
    allow_web: bool = True
    search_mode: SearchMode | None = Field(default=None, description=_MODE_HELP)
    rerank: bool | None = Field(default=None, description=_RERANK_HELP)


def _config(name: str):
    if name not in CHUNK_CONFIGS:
        raise HTTPException(422, f"chunk_config must be one of {list(CHUNK_CONFIGS)}")
    return CHUNK_CONFIGS[name]


def _ask_response(question: str, cfg, hits, result) -> dict:
    return {
        "question": question,
        "chunk_config": {"name": cfg.name, "size": cfg.size, "overlap": cfg.overlap},
        **asdict(result),
        "retrieved": [_retrieved(h) for h in hits],
    }


def _retrieved(h) -> dict:
    return {"file": h.source, "page_start": h.page_start, "page_end": h.page_end,
            "distance": None if h.distance is None else round(h.distance, 4), "chars": len(h.text), "scores": h.scores}


def _search(cfg, question: str, top_k: int, mode, rerank):
    try:
        return retrieval.search(cfg, question, top_k, mode, rerank)
    except Exception as e:  # e.g. reranker model download failed
        logging.getLogger("uvicorn.error").exception("retrieval failed")
        raise HTTPException(503, f"Search failed: {type(e).__name__}: {e}")


def _ask(question: str, config_name: str, top_k: int, mode=None, rerank=None) -> dict:
    cfg = _config(config_name)
    hits = _search(cfg, question, top_k, mode, rerank)
    try:
        result = answer_question(question, hits)
    except LLMError as e:
        raise HTTPException(e.status, e.message)
    return _ask_response(question, cfg, hits, result)


def _sse(events: Iterator[dict]) -> StreamingResponse:
    """Server-Sent Events: one `data: {json}` message per event.

    The HTTP status is already 200 once streaming starts, so failures arrive as
    an {"type": "error"} event instead of an error status.
    """

    def body():
        try:
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except LLMError as e:
            yield f"data: {json.dumps({'type': 'error', 'status': e.status, 'message': e.message})}\n\n"
        except Exception:
            logging.getLogger("uvicorn.error").exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'status': 500, 'message': 'Internal error while answering'})}\n\n"

    # no-cache + X-Accel-Buffering stop proxies (e.g. nginx when deployed) from buffering the stream.
    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": LLM_PROVIDER, "llm_model": LLM_MODEL,
            "retrieval": {"mode": RETRIEVAL_MODE, "rerank": RETRIEVAL_RERANK, "rerank_model": RERANK_MODEL}}


@app.get("/stats")
def stats():
    return {**{name: store.stats(cfg) for name, cfg in CHUNK_CONFIGS.items()}, "answer_cache": answer_cache.stats()}


@app.delete("/cache")
def clear_cache():
    """Empty the local /ask answer cache (e.g. after changing the prompt or model settings)."""
    answer_cache.clear()
    return answer_cache.stats()


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(..., description="One or more PDF files")):
    """Upload PDFs and index them. Re-uploading a file with the same name replaces it.

    Limits (env-configurable): MAX_FILES_PER_UPLOAD files, MAX_UPLOAD_MB each,
    MAX_PDF_PAGES pages each. Files are checked before any existing document is
    replaced; if one file is rejected, none are saved.
    """
    # Required (not Optional) so Swagger UI renders a file picker instead of a text box.
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(413, f"Too many files: upload at most {MAX_FILES_PER_UPLOAD} at a time")
    names = [_safe_filename(f.filename) for f in files]
    bad = [f.filename or "(unnamed)" for f, n in zip(files, names) if not n.lower().endswith(".pdf")]
    if bad:
        raise HTTPException(422, f"Only PDF files are accepted, got: {', '.join(bad)}")

    incoming = DOCS_DIR / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for f, name in zip(files, names):
            tmp = incoming / name
            _save_limited(f, tmp, name)
            _check_pdf(tmp, name)
            staged.append((tmp, DOCS_DIR / name))
    except HTTPException:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        for f in incoming.iterdir():
            f.unlink(missing_ok=True)
        raise
    paths = []
    for tmp, dest in staged:
        tmp.replace(dest)
        paths.append(dest)
    return {"ingested": ingest_paths(paths, list(CHUNK_CONFIGS.values()))}


def _safe_filename(filename: str | None) -> str:
    """Keep only the base name and ordinary characters (no paths, no hidden files)."""
    name = re.sub(r"[^\w.\- ()]", "_", Path(filename or "").name).strip(" .")
    return name or "upload.pdf"


def _save_limited(upload: UploadFile, dest: Path, name: str) -> None:
    """Stream to disk, stopping as soon as the size limit is passed (never holds the file in memory)."""
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    written = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"{name} is larger than the {MAX_UPLOAD_MB:g} MB limit")
            out.write(chunk)


def _check_pdf(path: Path, name: str) -> None:
    with open(path, "rb") as f:
        header = f.read(1024)
    if b"%PDF-" not in header:  # closed first: Windows can't delete an open file
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"{name} is not a valid PDF")
    try:
        pages = len(PdfReader(path).pages)
    except (PdfReadError, ValueError, OSError) as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"{name} could not be read as a PDF ({type(e).__name__})")
    if pages > MAX_PDF_PAGES:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"{name} has {pages} pages; the limit is {MAX_PDF_PAGES}")


@app.post("/ingest/folder")
def ingest_folder():
    """(Re)index every PDF already in the data/pdfs folder."""
    paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not paths:
        raise HTTPException(400, f"No PDFs found in {DOCS_DIR}")
    return {"ingested": ingest_paths(paths, list(CHUNK_CONFIGS.values()))}


@app.post("/ask")
def ask(req: AskRequest):
    return _ask(req.question, req.chunk_config, req.top_k, req.search_mode, req.rerank)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Streaming /ask (Server-Sent Events).

    Events: `retrieved` (the passages found), `delta` {text} as the model writes
    (raw text, citation labels unresolved), then `done` with the same body as
    /ask (answer with [n] markers, citations, token usage), or `error`.
    """
    cfg = _config(req.chunk_config)
    hits = _search(cfg, req.question, req.top_k, req.search_mode, req.rerank)

    def events():
        yield {"type": "retrieved", "passages": [{"file": h.source, "page_start": h.page_start, "page_end": h.page_end} for h in hits]}
        for ev in stream_answer(req.question, hits):
            if ev["type"] == "done":
                yield {"type": "done", **_ask_response(req.question, cfg, hits, ev["answer"])}
            else:
                yield ev

    return _sse(events())


@app.post("/ask/compare")
def ask_compare(req: CompareRequest):
    """Answer the same question with each chunk config, side by side."""
    return {name: _ask(req.question, name, req.top_k, req.search_mode, req.rerank) for name in CHUNK_CONFIGS}


@app.post("/agent")
def agent(req: AgentRequest):
    """Multi-step agent with search_docs, calculator and web_search.

    Returns the answer with citations (file + page, or URL), a log of every step,
    and `stopped_reason`: "answered", or "step_limit" if it hit `max_steps` and
    had to answer with what it had gathered.
    """
    _config(req.chunk_config)
    try:
        return asdict(run_agent(req.question, req.max_steps, req.chunk_config, req.allow_web, req.search_mode, req.rerank))
    except LLMError as e:
        raise HTTPException(e.status, e.message)


@app.post("/agent/stream")
def agent_stream(req: AgentRequest):
    """Streaming /agent (Server-Sent Events), one event per thing that happens:

    `start`, `step_start` {step, forced}, `delta` {step, text} while the model
    writes, `retry` {step, note} (discard that step's streamed text),
    `tool_call` {step, tool, input}, `tool_result` {step, tool, output_preview,
    is_error, ms}, `step_end` {step, log}, then `done` {result} with the same
    body as /agent, or `error`.
    """
    _config(req.chunk_config)

    def events():
        for ev in agent_events(req.question, req.max_steps, req.chunk_config, req.allow_web,
                               search_mode=req.search_mode, rerank=req.rerank):
            yield {"type": "done", "result": asdict(ev["result"])} if ev["type"] == "done" else ev

    return _sse(events())
