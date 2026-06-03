from __future__ import annotations

import requests
from fastapi import HTTPException, Request

from deploy.config import SUPABASE_KEY, SUPABASE_URL


def extract_bearer_token(request: Request):
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].strip(), parts[1].strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def get_user_id_from_supabase(access_token: str):
    """Best-effort: validate access token and return user id."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Missing required environment variable: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY"
        )
    try:
        r = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "apikey": SUPABASE_KEY,
                "Accept": "application/json",
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("id")
    except Exception:
        return None


def require_authenticated_user(request: Request) -> str:
    access_token = extract_bearer_token(request)
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    user_id = get_user_id_from_supabase(access_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


def optional_authenticated_user(request: Request) -> str | None:
    access_token = extract_bearer_token(request)
    if not access_token:
        return None
    user_id = get_user_id_from_supabase(access_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id
