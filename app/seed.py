"""Permanent document library for hosts whose disk is wiped on restart (e.g. Render free).

At image build time, PDFs committed to `library/` are indexed into SEED_DIR:

    DATA_DIR=/app/seed python -m app.seed build library

At startup, if the data folder has no index yet (fresh container, or a new empty
disk), the prebuilt index and PDFs are copied in: a file copy that takes seconds,
instead of re-embedding every PDF on a slow CPU. Files uploaded through the page
are added on top and last until the next restart unless the host has a disk.
"""

import logging
import shutil
import sys
from pathlib import Path

from app.config import CHROMA_DIR, CHUNK_CONFIGS, DOCS_DIR, SEED_DIR

log = logging.getLogger("uvicorn.error")


def seed_if_empty() -> bool:
    """Copy the prebuilt library into the data folder if it has no index. Returns True if seeded."""
    seed_chroma, seed_pdfs = SEED_DIR / "chroma", SEED_DIR / "pdfs"
    if not seed_chroma.is_dir():
        return False
    if CHROMA_DIR.is_dir() and any(CHROMA_DIR.iterdir()):
        return False  # existing index (e.g. on a persistent disk): never overwrite it
    shutil.copytree(seed_chroma, CHROMA_DIR, dirs_exist_ok=True)
    if seed_pdfs.is_dir():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for pdf in seed_pdfs.glob("*.pdf"):
            shutil.copy2(pdf, DOCS_DIR / pdf.name)
    log.info("Seeded the document library from %s", SEED_DIR)
    return True


def build(library: Path) -> None:
    """Index library/*.pdf into the current DATA_DIR (run with DATA_DIR=SEED_DIR at build time)."""
    from app.ingest import ingest_paths

    pdfs = sorted(library.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {library}; the app will start with an empty library.")
        return
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    copies = []
    for pdf in pdfs:
        shutil.copy2(pdf, DOCS_DIR / pdf.name)
        copies.append(DOCS_DIR / pdf.name)
    for result in ingest_paths(copies, list(CHUNK_CONFIGS.values())):
        print(result)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "build":
        raise SystemExit("usage: DATA_DIR=<seed dir> python -m app.seed build <library folder>")
    build(Path(sys.argv[2]))
