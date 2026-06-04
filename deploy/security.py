from __future__ import annotations

from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from deploy.config import (
    SECURITY_GLOBAL_RATE_LIMIT_PER_MINUTE,
    SECURITY_MAX_JSON_BYTES,
    SECURITY_MAX_REQUEST_BYTES,
)
from deploy.rate_limit import check_rate_limit

JSON_ENDPOINTS = {
    "/history/save",
    "/llm/chat",
    "/llm/advice/diagnosis",
    "/llm/advice/weather",
    "/llm/care-plan/diagnosis",
}

MULTIPART_ENDPOINTS = {
    "/predict",
    "/ai-feedback/low-confidence",
}


def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _content_length(request: Request) -> int | None:
    raw = request.headers.get("content-length")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return -1


def _normalized_content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


def _path_has_suspicious_segments(path: str) -> bool:
    lowered = path.lower()
    return any(
        token in lowered
        for token in (
            "\x00",
            "../",
            "..\\",
            "%2e%2e",
            "%00",
            "<script",
            "${",
            "{{",
        )
    )


def _validate_content_type(request: Request) -> JSONResponse | None:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return None

    content_type = _normalized_content_type(request)
    path = request.url.path

    if path in JSON_ENDPOINTS and content_type != "application/json":
        return _error(415, "Expected application/json")
    if path in MULTIPART_ENDPOINTS and content_type != "multipart/form-data":
        return _error(415, "Expected multipart/form-data")
    return None


def _validate_content_length(request: Request) -> JSONResponse | None:
    if request.method not in {"POST", "PUT", "PATCH"}:
        return None

    length = _content_length(request)
    if length == -1:
        return _error(400, "Invalid Content-Length")
    if length is None:
        return None

    path = request.url.path
    if path in JSON_ENDPOINTS and length > SECURITY_MAX_JSON_BYTES:
        return _error(413, "JSON body is larger than allowed")
    if length > SECURITY_MAX_REQUEST_BYTES:
        return _error(413, "Request body is larger than allowed")
    return None


def add_security_headers(response: Response) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault(
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    )


async def security_middleware(request: Request, call_next: Callable) -> Response:
    if _path_has_suspicious_segments(request.url.path):
        response = _error(400, "Suspicious request path")
        add_security_headers(response)
        return response

    if request.url.path not in {"/health", "/health/ready"}:
        try:
            check_rate_limit(
                request,
                "global",
                SECURITY_GLOBAL_RATE_LIMIT_PER_MINUTE,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", 429)
            detail = getattr(exc, "detail", "Too many requests")
            response = _error(status_code, detail)
            add_security_headers(response)
            return response

    response = _validate_content_type(request) or _validate_content_length(request)
    if response is not None:
        add_security_headers(response)
        return response

    response = await call_next(request)
    add_security_headers(response)
    return response
