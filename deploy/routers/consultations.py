from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
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


def _http_url(value, field_name: str) -> str:
    text = _clean_text(value)
    if not text:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an http or https URL")
    return text


def _dict_or_empty(value, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a JSON object")
    return value


def _consultation_response(inserted: dict, photo_urls: List[str], expected_reply_at: str) -> dict:
    return {
        "id": inserted.get("id"),
        "status": inserted.get("status", "pending"),
        "photo_urls": inserted.get("photo_urls") or photo_urls,
        "primary_photo_url": inserted.get("primary_photo_url") or photo_urls[0],
        "expected_reply_at": inserted.get("expected_reply_at") or expected_reply_at,
    }


def _insert_consultation_request(row: dict) -> dict:
    try:
        result = supabase.table("consultation_requests").insert(row).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))
    return result.data[0] if getattr(result, "data", None) else row


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

    inserted = await run_in_threadpool(_insert_consultation_request, row)
    return _consultation_response(inserted, photo_urls, expected_reply_at)


@router.post("/expert-request-from-url")
async def create_expert_request_from_url(request: Request, payload: dict = Body(...)):
    user_id = await run_in_threadpool(require_authenticated_user, request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")

    image_url = _http_url(payload.get("image_url"), "image_url")
    question = _clean_text(payload.get("question"))
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    photo_urls = [image_url]
    expected_reply_at = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    row = {
        "created_by": user_id,
        "title": _clean_text(payload.get("title")) or "Hồ sơ hỏi chuyên gia",
        "question": question,
        "image_url": image_url,
        "primary_photo_url": image_url,
        "photo_urls": photo_urls,
        "contact_phone": _clean_text(payload.get("contact_phone")),
        "user_plant_id": _clean_text(payload.get("user_plant_id")),
        "storage_consent": _as_bool(payload.get("storage_consent"), True),
        "questionnaire_json": _dict_or_empty(payload.get("questionnaire_json"), "questionnaire_json"),
        "profile_snapshot": _dict_or_empty(payload.get("profile_snapshot"), "profile_snapshot"),
        "diagnosis_context": _dict_or_empty(payload.get("diagnosis_context"), "diagnosis_context"),
        "expected_reply_at": expected_reply_at,
        "notify_email": _as_bool(payload.get("notify_email"), False),
        "notify_local": _as_bool(payload.get("notify_local"), True),
        "user_email": _clean_text(payload.get("user_email")),
        "status": "pending",
        "diagnostic_flow_version": "recovery_followup_v1",
    }

    inserted = await run_in_threadpool(_insert_consultation_request, row)
    return _consultation_response(inserted, photo_urls, expected_reply_at)
