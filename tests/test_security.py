"""Password, rate limit, startup guard and upload limits."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app import main as main_mod
from app import security


def _four_page_pdf() -> bytes:
    """A small valid PDF built in memory (data/ is git-ignored, so tests can't rely on it)."""
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class SAMPLE:  # same interface the tests use: SAMPLE.read_bytes()
    read_bytes = staticmethod(_four_page_pdf)


def _basic(password: str, user: str = "anyone") -> dict:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "DOCS_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(main_mod, "ingest_paths", lambda paths, cfgs: [{"file": p.name} for p in paths])
    return TestClient(main_mod.app)


def test_password_required_except_health(client, monkeypatch):
    monkeypatch.setattr(security, "APP_PASSWORD", "s3cret")
    r = client.get("/")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    assert client.get("/", headers=_basic("wrong")).status_code == 401
    assert client.get("/", headers={"Authorization": "Basic !!notbase64"}).status_code == 401
    assert client.get("/", headers=_basic("s3cret")).status_code == 200
    assert client.get("/health").status_code == 200  # platform health checks stay open


def test_no_password_means_open(client, monkeypatch):
    monkeypatch.setattr(security, "APP_PASSWORD", "")
    assert client.get("/").status_code == 200


def test_rate_limit_per_ip(client, monkeypatch):
    monkeypatch.setattr(security, "limiter", security.RateLimiter(2))
    monkeypatch.setattr(main_mod.retrieval, "search", lambda *a, **k: [])
    codes = [client.post("/ask", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/ask", json={"question": "q"})
    assert int(r.headers["retry-after"]) > 0 and "wait" in r.json()["detail"]
    assert client.get("/stats").status_code == 200  # only questions are limited


def test_startup_guard(monkeypatch):
    monkeypatch.setattr(security, "ON_HOSTING_PLATFORM", True)
    monkeypatch.setattr(security, "APP_PASSWORD", "")
    monkeypatch.setattr(security, "ALLOW_PUBLIC", False)
    with pytest.raises(RuntimeError, match="APP_PASSWORD"):
        security.check_startup_config()
    monkeypatch.setattr(security, "ALLOW_PUBLIC", True)
    security.check_startup_config()  # explicit opt-in is allowed


def test_upload_accepts_real_pdf(client):
    r = client.post("/ingest", files=[("files", ("../../evil name?.pdf", SAMPLE.read_bytes(), "application/pdf"))])
    assert r.status_code == 200 and r.json()["ingested"][0]["file"] == "evil name_.pdf"


def test_upload_rejects_fake_oversized_and_too_many(client, monkeypatch):
    fake = client.post("/ingest", files=[("files", ("notes.pdf", b"hello, not a pdf", "application/pdf"))])
    assert fake.status_code == 422 and "not a valid PDF" in fake.json()["detail"]

    monkeypatch.setattr(main_mod, "MAX_UPLOAD_MB", 0.0001)  # ~100 bytes, below the test PDF's size
    big = client.post("/ingest", files=[("files", ("big.pdf", SAMPLE.read_bytes(), "application/pdf"))])
    assert big.status_code == 413

    monkeypatch.setattr(main_mod, "MAX_UPLOAD_MB", 20)
    monkeypatch.setattr(main_mod, "MAX_PDF_PAGES", 2)
    pages = client.post("/ingest", files=[("files", ("long.pdf", SAMPLE.read_bytes(), "application/pdf"))])
    assert pages.status_code == 413 and "4 pages" in pages.json()["detail"]

    monkeypatch.setattr(main_mod, "MAX_FILES_PER_UPLOAD", 1)
    many = client.post("/ingest", files=[("files", (f"{i}.pdf", SAMPLE.read_bytes(), "application/pdf")) for i in range(2)])
    assert many.status_code == 413


def test_rejected_batch_saves_nothing(client, monkeypatch, tmp_path):
    r = client.post("/ingest", files=[("files", ("good.pdf", SAMPLE.read_bytes(), "application/pdf")),
                                      ("files", ("bad.pdf", b"junk", "application/pdf"))])
    assert r.status_code == 422
    docs = tmp_path / "pdfs"
    assert not list(docs.glob("*.pdf")) and not list((docs / ".incoming").iterdir())
