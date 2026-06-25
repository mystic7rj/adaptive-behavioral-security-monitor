"""Service health endpoint used by external uptime and readiness monitors."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Expose lightweight status and current UTC time without authentication."""

    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
