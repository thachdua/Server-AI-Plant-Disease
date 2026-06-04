from __future__ import annotations

import time
from ipaddress import ip_address

from fastapi import HTTPException, Request

_hits: dict[str, list[float]] = {}


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        candidate = forwarded_for.split(",", 1)[0].strip()
        try:
            return str(ip_address(candidate))
        except ValueError:
            return "invalid-forwarded-for"
    if request.client:
        try:
            return str(ip_address(request.client.host))
        except ValueError:
            return request.client.host[:80]
    return "unknown"


def check_rate_limit(request: Request, namespace: str, limit_per_minute: int) -> None:
    if limit_per_minute <= 0:
        return

    key = f"{namespace}|{client_key(request)}"
    now = time.monotonic()
    window_start = now - 60
    hits = [t for t in _hits.get(key, []) if t >= window_start]
    if len(hits) >= limit_per_minute:
        raise HTTPException(status_code=429, detail="Too many requests")
    hits.append(now)
    _hits[key] = hits


def reset_rate_limits() -> None:
    _hits.clear()
