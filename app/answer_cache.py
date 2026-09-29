"""Local answer cache for /ask: same question + same retrieved passages -> reuse the answer.

This is separate from provider prompt caching (which discounts re-sent prompt
tokens but still runs the model). A hit here skips the LLM call entirely, so it
costs nothing and doesn't count against free-tier rate limits.

The key includes the exact text of every retrieved passage, so when a PDF is
added or changed and retrieval returns different passages, old entries simply
stop matching. No explicit invalidation is needed.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict

from app.config import ANSWER_CACHE_SIZE, ANSWER_CACHE_TTL


class AnswerCache:
    def __init__(self, max_size: int, ttl_seconds: float):
        self.max_size, self.ttl = max_size, ttl_seconds
        self._items: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()  # FastAPI runs sync endpoints on a thread pool
        self.hits = self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_size > 0 and self.ttl > 0

    @staticmethod
    def key(provider: str, model: str | None, question: str, hits) -> str:
        normalized = " ".join(question.split()).casefold()
        payload = [provider, model, normalized, [(h.source, h.text) for h in hits]]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl:
                self._items.pop(key, None)
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._items[key] = (time.monotonic(), value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "entries": len(self._items), "hits": self.hits,
                    "misses": self.misses, "ttl_seconds": self.ttl, "max_size": self.max_size}


answer_cache = AnswerCache(ANSWER_CACHE_SIZE, ANSWER_CACHE_TTL)
