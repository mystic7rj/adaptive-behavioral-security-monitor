"""Rate limiter configuration keyed by authenticated JWT subject."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import get_current_user


def limiter_key_func(request: Request) -> str:
    """Use JWT subject as the limiter identity, falling back to remote address when absent."""

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return get_remote_address(request)
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        return get_remote_address(request)
    user = get_current_user(f"Bearer {token}")
    return user["sub"]


limiter = Limiter(key_func=limiter_key_func)
