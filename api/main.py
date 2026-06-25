"""FastAPI application assembly for ABSM API with security and observability controls."""

from __future__ import annotations

from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from api.audit_logger import AuditMiddleware
from api.rate_limit import limiter
from api.routes.alerts import router as alerts_router
from api.routes.health import router as health_router
from config.settings import load_settings

settings = load_settings()
app = FastAPI(title="ABSM API")
app.state.limiter = limiter


async def _rate_limit_handler(request: Request, exc: Exception) -> Response:
    """Adapt slowapi's handler to Starlette's expected exception handler signature."""

    return _rate_limit_exceeded_handler(request, cast(RateLimitExceeded, exc))


app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(AuditMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Response:
    """Apply baseline browser hardening headers to every HTTP response."""

    response: Any = await cast(Any, call_next)(request)
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return cast(Response, response)


app.include_router(alerts_router)
app.include_router(health_router)
