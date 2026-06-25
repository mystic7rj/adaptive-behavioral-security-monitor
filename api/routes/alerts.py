"""Alerts endpoints with role checks via FastAPI dependency injection.

Using Depends-based role guards keeps authorization centralized, auditable, and reusable
across endpoints, avoiding duplicated inline checks that drift over time.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from api.auth import require_role
from api.rate_limit import limiter

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
@limiter.limit("100/minute")
def list_alerts(request: Request, user: dict[str, str] = Depends(require_role("analyst"))) -> dict[str, Any]:
    """Return the alerts collection to authenticated analysts."""

    del request
    return {"items": [], "requested_by": user["sub"]}


@router.get("/{alert_id}")
@limiter.limit("100/minute")
def get_alert(
    alert_id: str,
    request: Request,
    user: dict[str, str] = Depends(require_role("analyst")),
) -> dict[str, str]:
    """Return a single alert resource to authenticated analysts."""

    del request
    return {"id": alert_id, "status": "open", "requested_by": user["sub"]}


@router.post("/{alert_id}/feedback")
@limiter.limit("100/minute")
def submit_feedback(
    alert_id: str,
    request: Request,
    user: dict[str, str] = Depends(require_role("analyst")),
) -> dict[str, str]:
    """Accept analyst feedback for an alert resource."""

    del request
    return {"id": alert_id, "feedback": "accepted", "submitted_by": user["sub"]}


@router.delete("/{alert_id}")
@limiter.limit("100/minute")
def delete_alert(alert_id: str, request: Request, user: dict[str, str] = Depends(require_role("admin"))) -> dict[str, str]:
    """Delete an alert when requested by an administrator."""

    del request
    return {"id": alert_id, "deleted_by": user["sub"]}
