from app import seed


def _patch(monkeypatch, tmp_path):
    seed_dir, data = tmp_path / "seed", tmp_path / "data"
    (seed_dir / "chroma").mkdir(parents=True)
    (seed_dir / "chroma" / "chroma.sqlite3").write_text("prebuilt index")
    (seed_dir / "pdfs").mkdir()
    (seed_dir / "pdfs" / "handbook.pdf").write_bytes(b"%PDF-1.4 library copy")
    monkeypatch.setattr(seed, "SEED_DIR", seed_dir)
    monkeypatch.setattr(seed, "CHROMA_DIR", data / "chroma")
    monkeypatch.setattr(seed, "DOCS_DIR", data / "pdfs")
    return data


def test_empty_data_folder_is_seeded(monkeypatch, tmp_path):
    data = _patch(monkeypatch, tmp_path)
    assert seed.seed_if_empty() is True
    assert (data / "chroma" / "chroma.sqlite3").read_text() == "prebuilt index"
    assert (data / "pdfs" / "handbook.pdf").exists()


def test_existing_index_is_never_overwritten(monkeypatch, tmp_path):
    data = _patch(monkeypatch, tmp_path)
    (data / "chroma").mkdir(parents=True)
    (data / "chroma" / "chroma.sqlite3").write_text("user's index with uploads")
    assert seed.seed_if_empty() is False
    assert (data / "chroma" / "chroma.sqlite3").read_text() == "user's index with uploads"


def test_no_seed_built_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.setattr(seed, "SEED_DIR", tmp_path / "missing")
    assert seed.seed_if_empty() is False
