"""Audit logging middleware for full request accountability with privacy-safe metadata.

Every request is logged to preserve an immutable forensic trail for incident response.
Client IP is hashed before logging to support correlation without storing raw network
identifiers that could expose sensitive personal data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, cast
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastapi import HTTPException

from api.auth import verify_token

LOGGER = logging.getLogger("absm.api.audit")


def _hashed_ip(raw_ip: str) -> str:
    """Hash a client IP value so logs never persist raw address data."""

    return hashlib.sha256(raw_ip.encode("utf-8")).hexdigest()


class AuditMiddleware(BaseHTTPMiddleware):
    """Capture structured, privacy-safe audit records for every HTTP request."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Log request/response metadata including principal context and request outcome."""

        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id
        response: Any = await call_next(request)

        authorization_header = request.headers.get("Authorization", "")
        user_id = "anonymous"
        role = "anonymous"
        if authorization_header.startswith("Bearer "):
            token = authorization_header.removeprefix("Bearer ").strip()
            if token:
                # Audit identity extraction must never block the request pipeline.
                try:
                    user_claims = verify_token(token)
                    user_id = user_claims["sub"]
                    role = user_claims["role"]
                except HTTPException:
                    user_id = "unauthenticated"
                    role = "unauthenticated"

        client_host = request.client.host if request.client is not None else "unknown"
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "role": role,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "request_id": request_id,
            "ip_hash": _hashed_ip(client_host),
        }
        # Structured JSON avoids ambiguous parsing and keeps PII controls explicit.
        LOGGER.info("%s", json.dumps(log_payload, separators=(",", ":")))
        return cast(Response, response)
