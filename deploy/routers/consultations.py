from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from deploy.auth import require_authenticated_user
from deploy.config import supabase
from deploy.routers.predict import (
    _sanitize_image,
    _supabase_feedback_error_detail,
    _validate_upload_metadata,
)

router = APIRouter(prefix="/consultations", tags=["consultations"])


def _clean_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _json_object(raw: str, field_name: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON")
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return value


def _upload_consultation_image(user_id: str, contents: bytes, content_type: str) -> str:
    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(content_type, "jpg")
    path = f"consultations/{user_id}/{uuid4().hex}.{extension}"
    try:
        supabase.storage.from_("plant-images").upload(
            path,
            contents,
            {
                "content-type": content_type,
                "cache-control": "3600",
                "upsert": "false",
            },
        )
        return supabase.storage.from_("plant-images").get_public_url(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))


@router.post("/expert-request")
async def create_expert_request(
    request: Request,
    metadata: str = Form(...),
    primary_photo: UploadFile = File(...),
    symptom_photos: Optional[List[UploadFile]] = File(default=None),
):
    user_id = await run_in_threadpool(require_authenticated_user, request)
    meta = _json_object(metadata, "metadata")

    files = [primary_photo] + list(symptom_photos or [])
    if len(files) > 4:
        raise HTTPException(
            status_code=400,
            detail="Upload at most 1 primary photo and 3 symptom photos",
        )

    photo_urls: List[str] = []
    for file in files:
        _validate_upload_metadata(file)
        contents = await file.read()
        contents, content_type = _sanitize_image(contents)
        url = await run_in_threadpool(
            _upload_consultation_image,
            user_id,
            contents,
            content_type,
        )
        photo_urls.append(url)

    if not photo_urls:
        raise HTTPException(status_code=400, detail="primary_photo is required")

    question = _clean_text(meta.get("question"))
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    expected_reply_at = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    row = {
        "created_by": user_id,
        "title": _clean_text(meta.get("title")) or "Hồ sơ hỏi chuyên gia",
        "question": question,
        "image_url": photo_urls[0],
        "primary_photo_url": photo_urls[0],
        "photo_urls": photo_urls,
        "contact_phone": _clean_text(meta.get("contact_phone")),
        "storage_consent": _as_bool(meta.get("storage_consent"), True),
        "questionnaire_json": meta.get("questionnaire_json") or {},
        "profile_snapshot": meta.get("profile_snapshot") or {},
        "diagnosis_context": meta.get("diagnosis_context") or {},
        "expected_reply_at": expected_reply_at,
        "notify_email": _as_bool(meta.get("notify_email"), False),
        "notify_local": _as_bool(meta.get("notify_local"), True),
        "user_email": _clean_text(meta.get("user_email")),
        "status": "pending",
        "diagnostic_flow_version": "expert_wizard_v1",
    }

    try:
        result = await run_in_threadpool(
            lambda: supabase.table("consultation_requests").insert(row).execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))

    inserted = result.data[0] if getattr(result, "data", None) else row
    return {
        "id": inserted.get("id"),
        "status": inserted.get("status", "pending"),
        "photo_urls": inserted.get("photo_urls") or photo_urls,
        "primary_photo_url": inserted.get("primary_photo_url") or photo_urls[0],
        "expected_reply_at": inserted.get("expected_reply_at") or expected_reply_at,
    }
