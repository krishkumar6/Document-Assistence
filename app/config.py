import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # keys and LLM_PROVIDER can live in a git-ignored .env file

# Everything the app writes (uploaded PDFs, search index, logs) goes under one
# folder, so a single persistent volume keeps it all when deployed. Railway
# exposes its volume as RAILWAY_VOLUME_MOUNT_PATH; elsewhere set DATA_DIR.
_data_root = os.getenv("DATA_DIR") or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
if _data_root:
    _base = Path(_data_root)
    DOCS_DIR = Path(os.getenv("DOCS_DIR", _base / "pdfs"))
    CHROMA_DIR = Path(os.getenv("CHROMA_DIR", _base / "chroma"))
    LOG_DIR = Path(os.getenv("LOG_DIR", _base / "logs"))
else:  # local development layout
    DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "pdfs"))
    CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
    LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

# Cap ONNX Runtime threads (embedder + reranker). On a tiny CPU share, like Render
# free's 0.1 CPU, the default of one busy-waiting thread per visible core makes the
# threads fight over the share; 1 thread without spinning is much faster there.
ORT_THREADS = int(os.getenv("ORT_THREADS", 0))  # 0 = ONNX Runtime default


def _limit_onnx_threads(n: int) -> None:
    import onnxruntime as ort

    base = ort.SessionOptions

    class LimitedSessionOptions(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.intra_op_num_threads = n
            self.inter_op_num_threads = 1
            self.add_session_config_entry("session.intra_op.allow_spinning", "0")

    # Chroma and fastembed both build sessions from onnxruntime.SessionOptions().
    ort.SessionOptions = LimitedSessionOptions


if ORT_THREADS > 0:
    _limit_onnx_threads(ORT_THREADS)

# Prebuilt index of the permanent library/ PDFs (made at image build, see app/seed.py).
SEED_DIR = Path(os.getenv("SEED_DIR", ROOT / "seed"))

# --- access and abuse limits (matter once deployed) ---
# Password for the whole app (web page + API). Unset = open, which is refused on a
# hosting platform unless ALLOW_PUBLIC=true, so a deploy is never accidentally public.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
ALLOW_PUBLIC = os.getenv("ALLOW_PUBLIC", "").lower() in ("1", "true", "yes")
ON_HOSTING_PLATFORM = any(os.getenv(v) for v in ("RAILWAY_ENVIRONMENT", "RENDER", "DATA_DIR"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", 20))  # per file
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", 10))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 500))
# Question requests (/ask, /agent) per client IP per minute; protects the LLM quota. 0 disables.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", 20))

# Retrieval: "vector", "bm25" or "hybrid" (both, fused), optionally reranked by a
# local cross-encoder. Defaults chosen from eval/compare_retrieval.py results.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
RETRIEVAL_RERANK = os.getenv("RETRIEVAL_RERANK", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# Candidates per retriever before fusion/reranking (at least 2x top_k). 12 matched 20's
# accuracy on the eval at ~35% less rerank time.
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", 12))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", ROOT / "models"))

# Local /ask answer cache (0 disables either).
ANSWER_CACHE_TTL = float(os.getenv("ANSWER_CACHE_TTL", 3600))
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", 256))

# Which LLM writes the answers. "anthropic" uses Claude with native citations;
# the others are free options reached through the OpenAI-compatible API.
# Defaults to anthropic only when a key is present, otherwise local Ollama.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "ollama")

# provider -> (base_url, default model, env var holding the key)
OPENAI_COMPATIBLE = {
    "ollama": ("http://localhost:11434/v1", "qwen2.5:3b", None),  # free, local, no key
    # Free plan (checked 2026-09): 30 req/min, 1K req/day, 8K tokens/min. Llama models are enterprise-only now.
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash", "GEMINI_API_KEY"),  # free tier
}
LLM_MODEL = os.getenv("LLM_MODEL") or (
    "claude-opus-5" if LLM_PROVIDER == "anthropic" else OPENAI_COMPATIBLE.get(LLM_PROVIDER, (None, None, None))[1]
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or OPENAI_COMPATIBLE.get(LLM_PROVIDER, (None,))[0]


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    size: int  # characters
    overlap: int  # characters

    @property
    def collection(self) -> str:
        return f"docs_{self.name}"


# The two chunkings under comparison. Each gets its own Chroma collection so
# they can be queried side by side over the same corpus.
CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "small": ChunkConfig("small", size=500, overlap=100),
    "large": ChunkConfig("large", size=2000, overlap=200),
}
DEFAULT_CHUNK_CONFIG = os.getenv("DEFAULT_CHUNK_CONFIG", "small")
