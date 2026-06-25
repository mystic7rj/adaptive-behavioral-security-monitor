"""Authentication helpers for strict JWT validation and role-based access control."""

from __future__ import annotations

from typing import Any, Callable, cast

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt  # type: ignore[import-untyped]

from config.settings import load_settings


def verify_token(token: str) -> dict[str, str]:
    """Validate JWT strictly and return principal claims without any fallback behavior.

    Authentication failures are intentionally never softened because partial trust would
    introduce privilege escalation risk; any signature/expiry/issuer/claim issue must
    terminate the request with a 401 response.
    """

    settings = load_settings()
    try:
        # python-jose decode output is untyped, so we cast before structured processing.
        payload = cast(
            dict[str, Any],
            jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=["HS256"],
                issuer=settings.JWT_ISSUER,
                options={"verify_aud": False},
            ),
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc

    subject_value = payload.get("sub")
    role_value = payload.get("role")
    issuer_value = payload.get("iss")
    if not isinstance(subject_value, str) or not isinstance(role_value, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    if issuer_value != settings.JWT_ISSUER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return {"sub": subject_value, "role": role_value}


def get_current_user(authorization: str = Header(default="")) -> dict[str, str]:
    """Extract and validate the JWT from the Authorization header."""

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return verify_token(token)


def require_role(required_role: str) -> Callable[..., dict[str, str]]:
    """Build a dependency that enforces role-based access with explicit 403 failures."""

    def _role_dependency(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
        """Enforce the required role through dependency injection at routing boundaries."""

        if user["role"] != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role",
            )
        return user

    return _role_dependency
