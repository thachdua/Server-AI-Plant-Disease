from __future__ import annotations

from datetime import datetime
from math import isfinite

from fastapi import HTTPException


def validate_coordinates(lat: float, lng: float) -> None:
    if not isfinite(lat) or not -90 <= lat <= 90:
        raise HTTPException(status_code=400, detail="lat must be between -90 and 90")
    if not isfinite(lng) or not -180 <= lng <= 180:
        raise HTTPException(status_code=400, detail="lng must be between -180 and 180")


def validate_severity(value: int | None, *, field: str = "severity") -> None:
    if value is None:
        return
    if value < 1 or value > 5:
        raise HTTPException(status_code=400, detail=f"{field} must be between 1 and 5")


def validate_limit(limit: int, *, max_limit: int = 1000) -> int:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be at least 1")
    if limit > max_limit:
        raise HTTPException(status_code=400, detail=f"limit must be at most {max_limit}")
    return limit


def validate_since(value: str | None) -> None:
    if not value:
        return
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="since must be an ISO-8601 datetime")


def validate_since_days(value: int) -> int:
    if value < 1 or value > 3650:
        raise HTTPException(status_code=400, detail="since_days must be between 1 and 3650")
    return value
