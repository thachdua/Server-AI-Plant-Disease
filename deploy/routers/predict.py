from __future__ import annotations

import io
from uuid import uuid4

import requests
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from deploy.auth import optional_authenticated_user, require_authenticated_user
from deploy.config import (
    HF_API_URL,
    PREDICT_MAX_UPLOAD_BYTES,
    PREDICT_RATE_LIMIT_PER_MINUTE,
    PREDICT_REQUIRE_AUTH,
    supabase,
)
from deploy.rate_limit import check_rate_limit

router = APIRouter()


def infer_plant_from_disease_label(label: str | None) -> str | None:
    if not isinstance(label, str):
        return None
    if "___" in label:
        plant_part = label.split("___", 1)[0].strip()
        return plant_part or None
    if "_" in label:
        plant_part = label.split("_", 1)[0].strip()
        return plant_part or None
    if " " in label:
        plant_part = label.split(" ", 1)[0].strip()
        return plant_part or None
    return None


def parse_confidence_percent(raw_confidence) -> str | None:
    confidence_value = None
    if isinstance(raw_confidence, (int, float)):
        confidence_value = float(raw_confidence)
    elif isinstance(raw_confidence, str):
        s = raw_confidence.strip().replace("%", "")
        try:
            confidence_value = float(s)
        except ValueError:
            confidence_value = None

    if confidence_value is None:
        return None
    if 0.0 <= confidence_value <= 1.0:
        confidence_value *= 100.0
    return f"{confidence_value:.2f}%"


def _validate_image(contents: bytes) -> str:
    if not contents:
        raise HTTPException(status_code=400, detail="Missing image file")
    if len(contents) > PREDICT_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than allowed")

    try:
        with Image.open(io.BytesIO(contents)) as image:
            image.verify()
            fmt = (image.format or "").lower()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    if fmt in {"jpeg", "jpg"}:
        return "image/jpeg"
    if fmt == "png":
        return "image/png"
    if fmt == "webp":
        return "image/webp"
    raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images are supported")


def _upload_image_to_supabase(contents: bytes, content_type: str) -> str:
    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(content_type, "jpg")
    file_name = f"predictions/{uuid4().hex}.{extension}"
    supabase.storage.from_("plant-images").upload(
        file_name, contents, {"content-type": content_type}
    )
    return supabase.storage.from_("plant-images").get_public_url(file_name)


@router.post("/predict")
async def predict(
    request: Request, selected_plant: str = Form(...), file: UploadFile = File(...)
):
    try:
        check_rate_limit(request, "predict", PREDICT_RATE_LIMIT_PER_MINUTE)
        auth_func = (
            require_authenticated_user if PREDICT_REQUIRE_AUTH else optional_authenticated_user
        )
        user_id = await run_in_threadpool(auth_func, request)

        selected_plant = selected_plant.strip()
        if not selected_plant:
            raise HTTPException(status_code=400, detail="selected_plant is required")

        contents = await file.read()
        content_type = _validate_image(contents)

        print(
            "🚀 Đang gửi yêu cầu sang Hugging Face cho cây:"
            f" {selected_plant} | authenticated={bool(user_id)}"
        )
        try:
            response = await run_in_threadpool(
                requests.post,
                HF_API_URL,
                files={"file": (file.filename or "image.jpg", contents, content_type)},
                data={"selected_plant": selected_plant},
                timeout=120,
            )
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="Hugging Face request timed out")
        except requests.RequestException:
            raise HTTPException(status_code=502, detail="Could not reach Hugging Face")

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail="Hugging Face không phản hồi hoặc đang bận",
            )

        try:
            result = response.json()
        except ValueError:
            raise HTTPException(status_code=502, detail="Hugging Face returned invalid JSON")

        if result.get("status") == "error":
            raise HTTPException(
                status_code=502,
                detail=result.get("message") or "Prediction service returned an error",
            )

        disease_name = result.get("disease")
        if not isinstance(disease_name, str) or not disease_name.strip():
            raise HTTPException(
                status_code=502,
                detail="Prediction service returned incomplete result",
            )
        disease_name = disease_name.strip()

        predicted_plant = (
            result.get("plant")
            or infer_plant_from_disease_label(disease_name)
            or selected_plant
        )

        confidence_percent_str = parse_confidence_percent(result.get("confidence"))
        if confidence_percent_str is None:
            raise HTTPException(
                status_code=502,
                detail="Prediction service returned invalid confidence",
            )

        image_url = await run_in_threadpool(
            _upload_image_to_supabase, contents, content_type
        )

        return {
            "status": "success",
            "plant": predicted_plant,
            "disease": disease_name,
            "confidence": confidence_percent_str,
            "image_url": image_url,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /predict error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
