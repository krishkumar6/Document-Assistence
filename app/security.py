"""Access control for deployment: a shared password and a per-IP rate limit.

Password: HTTP Basic auth with any username and APP_PASSWORD as the password.
Browsers show their own login prompt and then send the credentials with every
request from the page (including the streaming fetches), so the web page needs
no changes. /health stays open for the platform's health checks.

Rate limit: questions (/ask*, /agent*) per client IP per minute, so one visitor
can't burn the whole free LLM quota. Client IPs come from X-Forwarded-For when
uvicorn runs with --proxy-headers (as in the Dockerfile), since Railway and
Render put a proxy in front of the app.
"""

import base64
import binascii
import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

from app.config import ALLOW_PUBLIC, APP_PASSWORD, ON_HOSTING_PLATFORM, RATE_LIMIT_PER_MINUTE

OPEN_PATHS = {"/health"}
RATE_LIMITED_PREFIXES = ("/ask", "/agent")


def check_startup_config() -> None:
    """Refuse to run publicly without a password unless explicitly allowed."""
    if ON_HOSTING_PLATFORM and not APP_PASSWORD and not ALLOW_PUBLIC:
        raise RuntimeError(
            "APP_PASSWORD is not set. Without it anyone with the URL can read your documents and use "
            "your LLM quota. Set APP_PASSWORD (or ALLOW_PUBLIC=true to deliberately run without one)."
        )


def password_ok(header: str | None) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    _user, sep, password = decoded.partition(":")
    return bool(sep) and hmac.compare_digest(password.encode(), APP_PASSWORD.encode())


class RateLimiter:
    """Sliding one-minute window per key."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        """Returns (allowed, seconds until the next request would be allowed)."""
        if self.per_minute <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, int(60 - (now - q[0])) + 1
            q.append(now)
            if len(self._hits) > 10_000:  # forget idle clients so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v]:
                    del self._hits[k]
            return True, 0


limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


async def security_middleware(request: Request, call_next):
    path = request.url.path
    if APP_PASSWORD and path not in OPEN_PATHS and not password_ok(request.headers.get("authorization")):
        return JSONResponse({"detail": "Password required"}, status_code=401,
                            headers={"WWW-Authenticate": 'Basic realm="Document Assistant", charset="UTF-8"'})
    if request.method == "POST" and path.startswith(RATE_LIMITED_PREFIXES):
        client = request.client.host if request.client else "unknown"
        allowed, retry_after = limiter.allow(client)
        if not allowed:
            return JSONResponse(
                {"detail": f"Too many questions. Please wait {retry_after} seconds and try again."},
                status_code=429, headers={"Retry-After": str(retry_after)})
    return await call_next(request)
