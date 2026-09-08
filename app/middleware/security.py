"""
Security Headers & Rate Limiting Middleware for SpaceNetra API Server.
"""

from collections import defaultdict
import time
from typing import Callable, Dict, List
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects standard production security headers on all API responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window in-memory rate limiter per IP/User per route category.
    """

    # Category limits (requests per 60 seconds)
    LIMITS = {
        "/api/search": 60,
        "/api/predict": 20,
        "/api/imagery": 120,
        "DEFAULT": 100,
    }

    def __init__(self, app, window_seconds: int = 60, enabled: bool = True, limits: Dict[str, int] = None):
        super().__init__(app)
        self.window_seconds = window_seconds
        self.enabled = enabled
        self.limits = limits or self.LIMITS.copy()
        # Key: (client_identifier, category) -> List[timestamp]
        self.requests: Dict[tuple, List[float]] = defaultdict(list)

    def _get_category(self, path: str) -> str:
        for prefix in ["/api/search", "/api/predict", "/api/imagery"]:
            if path.startswith(prefix):
                return prefix
        return "DEFAULT"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled or request.url.path in ["/health", "/api/health", "/"]:
            return await call_next(request)

        client_ip = request.client.host if request.client else "127.0.0.1"
        category = self._get_category(request.url.path)
        limit = self.limits.get(category, self.limits.get("DEFAULT", 100))

        key = (client_ip, category)
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean timestamps older than window
        timestamps = [ts for ts in self.requests[key] if ts > cutoff]
        self.requests[key] = timestamps

        if len(timestamps) >= limit:
            retry_after = int(self.window_seconds - (now - timestamps[0])) if timestamps else self.window_seconds
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."},
                headers={"Retry-After": str(max(1, retry_after))},
            )

        self.requests[key].append(now)
        return await call_next(request)
