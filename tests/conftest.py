import pytest

from app.answer_cache import answer_cache


@pytest.fixture(autouse=True)
def _empty_answer_cache():
    """The answer cache is process-wide; isolate every test from the others."""
    answer_cache.clear()
    yield
    answer_cache.clear()
