"""Ingest PDFs into every chunk-config collection.

    python -m app.ingest                   # everything in data/pdfs
    python -m app.ingest a.pdf b.pdf       # specific files
    python -m app.ingest --config small    # one chunking only
"""

import argparse
import time
from pathlib import Path

from app import store
from app.chunking import chunk_pages
from app.config import CHUNK_CONFIGS, DOCS_DIR, ChunkConfig
from app.pdf_loader import extract_pages


def ingest_file(path: Path, configs: list[ChunkConfig]) -> dict:
    pages = extract_pages(path)
    result = {"file": path.name, "pages": len(pages), "chunks": {}}
    if not pages:
        result["warning"] = "no extractable text (scanned PDF? needs OCR)"
        return result
    for cfg in configs:
        chunks = chunk_pages(pages, path.name, cfg.size, cfg.overlap)
        store.replace_document(cfg, path.name, chunks)
        result["chunks"][cfg.name] = len(chunks)
    return result


def ingest_paths(paths: list[Path], configs: list[ChunkConfig]) -> list[dict]:
    return [ingest_file(p, configs) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--config", choices=list(CHUNK_CONFIGS), action="append")
    args = parser.parse_args()

    paths = args.paths or sorted(DOCS_DIR.glob("*.pdf"))
    if not paths:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}")
    configs = [CHUNK_CONFIGS[n] for n in (args.config or CHUNK_CONFIGS)]

    t0 = time.perf_counter()
    for r in ingest_paths(paths, configs):
        print(r)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
