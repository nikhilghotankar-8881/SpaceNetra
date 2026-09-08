"""
Structured JSONL Audit Logging Middleware for SpaceNetra API Requests.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

AUDIT_LOG_DIR = Path("logs")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit.jsonl"


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware that captures every incoming HTTP request and response,
    logging detailed structured JSON audit records.
    """

    def __init__(self, app, log_filepath: Path = AUDIT_LOG_FILE):
        super().__init__(app)
        self.log_filepath = log_filepath
        self.log_filepath.parent.mkdir(parents=True, exist_ok=True)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        
        # Capture client IP & path
        client_ip = request.client.host if request.client else "127.0.0.1"
        path = request.url.path
        method = request.method

        user_id = "anonymous"
        user_role = "UNAUTHENTICATED"
        
        # Check authorization header if present for audit identity
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                from backend.services.auth_service import AuthService
                payload = AuthService.verify_access_token(token)
                user_id = payload.get("sub", payload.get("user_id", "authenticated_user"))
                user_role = payload.get("role", "OBSERVER")
            except Exception:
                user_id = "invalid_token"

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        status_code = response.status_code

        # Add request_id header to response
        response.headers["X-Request-ID"] = request_id

        # Determine log level
        level = "INFO"
        if status_code >= 500:
            level = "ERROR"
        elif status_code >= 400:
            level = "WARNING"

        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": "API_REQUEST",
            "request_id": request_id,
            "method": method,
            "path": path,
            "user_id": user_id,
            "role": user_role,
            "ip": client_ip,
            "status_code": status_code,
            "latency_ms": duration_ms,
        }

        # Write to JSONL log file
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log entry: {e}")

        return response
