from __future__ import annotations

import io
import json
import warnings
from typing import Optional
from uuid import uuid4

import requests
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError, DecompressionBombWarning
from starlette.concurrency import run_in_threadpool

from deploy.auth import optional_authenticated_user, require_authenticated_user
from deploy.config import (
    HF_API_URL,
    PREDICT_LOW_CONFIDENCE_THRESHOLD,
    PREDICT_MAX_UPLOAD_BYTES,
    PREDICT_RATE_LIMIT_PER_MINUTE,
    PREDICT_REQUIRE_AUTH,
    PREDICT_UNRECOGNIZED_THRESHOLD,
    SECURITY_MAX_IMAGE_PIXELS,
    supabase,
)
from deploy.rate_limit import check_rate_limit

router = APIRouter()
Image.MAX_IMAGE_PIXELS = SECURITY_MAX_IMAGE_PIXELS

UNRECOGNIZED_MESSAGE = (
    "xin lỗi, hiện tại ứng dụng của chúng tôi không nhận diện được loại cây này"
)


def _supabase_feedback_error_detail(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if "ai_feedback_cases" in lowered and (
        "does not exist" in lowered
        or "not found" in lowered
        or "could not find" in lowered
        or "relation" in lowered
    ):
        return (
            "Supabase chưa có bảng ai_feedback_cases. "
            "Hãy chạy supabase/sql/011_ai_feedback_cases.sql trước."
        )
    if "plant-images" in lowered or "bucket" in lowered:
        return (
            "Supabase Storage chưa có bucket plant-images hoặc backend không có quyền upload. "
            "Hãy tạo bucket plant-images và kiểm tra SUPABASE_SERVICE_ROLE_KEY trên Render."
        )
    if (
        "row-level security" in lowered
        or "rls" in lowered
        or "permission denied" in lowered
        or "violates row-level security" in lowered
        or "42501" in lowered
    ):
        return (
            "Backend không có quyền ghi Supabase. "
            "Hãy dùng SUPABASE_SERVICE_ROLE_KEY trên Render hoặc kiểm tra RLS của bảng ai_feedback_cases."
        )
    return "Could not save feedback"


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
    confidence_value = parse_confidence_value(raw_confidence)
    if confidence_value is None:
        return None
    return f"{confidence_value:.2f}%"


def parse_confidence_value(raw_confidence) -> float | None:
    confidence_value = None
    is_percent_string = False
    if isinstance(raw_confidence, (int, float)):
        confidence_value = float(raw_confidence)
    elif isinstance(raw_confidence, str):
        raw = raw_confidence.strip()
        is_percent_string = "%" in raw
        s = raw.replace("%", "")
        try:
            confidence_value = float(s)
        except ValueError:
            confidence_value = None

    if confidence_value is None:
        return None
    if not is_percent_string and 0.0 <= confidence_value <= 1.0:
        confidence_value *= 100.0
    return confidence_value


def _validate_image(contents: bytes) -> str:
    return _sanitize_image(contents)[1]


def _sanitize_image(contents: bytes) -> tuple[bytes, str]:
    if not contents:
        raise HTTPException(status_code=400, detail="Missing image file")
    if len(contents) > PREDICT_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than allowed")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                image.verify()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                fmt = (image.format or "").lower()
                image.load()
                width, height = image.size
                if width < 1 or height < 1:
                    raise HTTPException(status_code=400, detail="Invalid image dimensions")
                if width * height > SECURITY_MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="Image dimensions are larger than allowed")

                safe = image.convert("RGB")
                output = io.BytesIO()
                safe.save(output, format="JPEG", quality=90, optimize=True)
                sanitized = output.getvalue()
                if len(sanitized) > PREDICT_MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Sanitized image is larger than allowed")

                if fmt not in {"jpeg", "jpg", "png", "webp"}:
                    raise HTTPException(status_code=400, detail="Only JPEG, PNG, or WebP images are supported")
                return sanitized, "image/jpeg"

    except HTTPException:
        raise
    except (UnidentifiedImageError, DecompressionBombError, DecompressionBombWarning):
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")


def _validate_upload_metadata(file: UploadFile) -> None:
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, or WebP uploads are accepted")
    filename = (file.filename or "").lower()
    if filename and not filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, or WebP uploads are accepted")


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


def _parse_client_quality_json(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _log_low_confidence_case(
    *,
    user_id: str | None,
    plant: str | None,
    disease: str,
    confidence: float,
    image_url: str,
    quality_json: dict | None = None,
    client_flow_version: str | None = None,
) -> None:
    if not user_id or confidence >= PREDICT_LOW_CONFIDENCE_THRESHOLD:
        return
    payload = {
        "created_by": user_id,
        "plant": plant,
        "predicted_disease": disease,
        "confidence": confidence,
        "image_url": image_url,
        "source": "predict",
        "reason": "low_confidence",
        "review_status": "pending",
    }
    if quality_json is not None:
        payload["quality_json"] = quality_json
    if client_flow_version:
        payload["client_flow_version"] = client_flow_version.strip()
    try:
        supabase.table("ai_feedback_cases").insert(payload).execute()
    except Exception as e:
        print(f"⚠️ low-confidence feedback log failed: {e}")


def _prediction_job_response(row: dict) -> dict:
    status = row.get("status")
    result = None
    if status == "done":
        confidence_text = row.get("confidence_text")
        confidence = row.get("confidence")
        if not confidence_text and isinstance(confidence, (int, float)):
            confidence_text = f"{float(confidence):.2f}%"
        is_unrecognized = (
            isinstance(confidence, (int, float))
            and float(confidence) < PREDICT_UNRECOGNIZED_THRESHOLD
        )
        result = {
            "status": "unrecognized" if is_unrecognized else "success",
            "message": UNRECOGNIZED_MESSAGE if is_unrecognized else None,
            "plant": row.get("predicted_plant"),
            "disease": row.get("predicted_disease"),
            "confidence": confidence_text,
            "image_url": row.get("result_image_url"),
        }
    return {
        "job_id": row.get("id"),
        "status": status,
        "image_url": row.get("image_url"),
        "error_message": row.get("error_message"),
        "result": result,
    }


def _insert_prediction_job(row: dict) -> dict:
    try:
        result = supabase.table("prediction_jobs").insert(row).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))
    inserted = result.data[0] if getattr(result, "data", None) else row
    if not inserted.get("id"):
        raise HTTPException(status_code=500, detail="Prediction job was not created")
    return inserted


def _get_prediction_job(job_id: str) -> dict | None:
    try:
        result = (
            supabase.table("prediction_jobs")
            .select("*")
            .eq("id", job_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))
    rows = getattr(result, "data", None) or []
    return rows[0] if rows else None


def _update_prediction_job(job_id: str, payload: dict) -> None:
    try:
        (
            supabase.table("prediction_jobs")
            .update(payload)
            .eq("id", job_id)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(exc))


def _download_prediction_image(image_url: str) -> tuple[bytes, str]:
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        raise HTTPException(status_code=502, detail="Could not download prediction image")
    return _sanitize_image(response.content)


def _call_hugging_face(contents: bytes, content_type: str, selected_plant: str, filename: str = "image.jpg") -> dict:
    try:
        response = requests.post(
            HF_API_URL,
            files={"file": (filename, contents, content_type)},
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
    return result


def _normalize_prediction_result(result: dict, selected_plant: str) -> dict:
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

    confidence_value = parse_confidence_value(result.get("confidence"))
    if confidence_value is None:
        raise HTTPException(
            status_code=502,
            detail="Prediction service returned invalid confidence",
        )

    return {
        "predicted_plant": predicted_plant,
        "predicted_disease": disease_name,
        "confidence": confidence_value,
        "confidence_text": f"{confidence_value:.2f}%",
    }


def process_prediction_job(job_id: str) -> None:
    try:
        job = _get_prediction_job(job_id)
        if not job:
            return
        if job.get("status") not in {"pending", "processing"}:
            return

        _update_prediction_job(job_id, {"status": "processing", "error_message": None})
        image_url = job.get("image_url")
        selected_plant = (job.get("selected_plant") or "").strip()
        if not image_url or not selected_plant:
            raise HTTPException(status_code=400, detail="Prediction job is missing image or plant")

        contents, content_type = _download_prediction_image(image_url)
        raw_result = _call_hugging_face(contents, content_type, selected_plant)
        normalized = _normalize_prediction_result(raw_result, selected_plant)
        confidence = normalized["confidence"]
        result_image_url = None if confidence < PREDICT_UNRECOGNIZED_THRESHOLD else image_url

        _update_prediction_job(
            job_id,
            {
                "status": "done",
                "predicted_plant": normalized["predicted_plant"],
                "predicted_disease": normalized["predicted_disease"],
                "confidence": confidence,
                "confidence_text": normalized["confidence_text"],
                "result_image_url": result_image_url,
                "error_message": None,
            },
        )

        if result_image_url:
            _log_low_confidence_case(
                user_id=job.get("created_by"),
                plant=normalized["predicted_plant"],
                disease=normalized["predicted_disease"],
                confidence=confidence,
                image_url=result_image_url,
                quality_json=None,
                client_flow_version="prediction_job_v1",
            )
    except HTTPException as exc:
        _update_prediction_job(
            job_id,
            {
                "status": "failed",
                "error_message": str(exc.detail),
            },
        )
    except Exception as exc:
        print(f"❌ prediction job {job_id} failed: {exc}")
        try:
            _update_prediction_job(
                job_id,
                {
                    "status": "failed",
                    "error_message": "Prediction failed",
                },
            )
        except Exception:
            pass


@router.post("/ai-feedback/low-confidence")
async def submit_low_confidence_feedback(
    request: Request,
    file: UploadFile = File(...),
    selected_plant: Optional[str] = Form(None),
    predicted_plant: Optional[str] = Form(None),
    predicted_disease: Optional[str] = Form(None),
    confidence: Optional[str] = Form(None),
    user_note: Optional[str] = Form(None),
    client_quality_json: Optional[str] = Form(None),
    client_flow_version: Optional[str] = Form(None),
):
    try:
        user_id = await run_in_threadpool(require_authenticated_user, request)
        _validate_upload_metadata(file)
        contents = await file.read()
        contents, content_type = _sanitize_image(contents)
        confidence_value = parse_confidence_value(confidence)
        image_url = await run_in_threadpool(
            _upload_image_to_supabase, contents, content_type
        )
        quality_json = _parse_client_quality_json(client_quality_json)

        payload = {
            "created_by": user_id,
            "plant": (predicted_plant or selected_plant or "").strip() or None,
            "predicted_disease": (predicted_disease or "").strip() or None,
            "confidence": confidence_value,
            "image_url": image_url,
            "source": "predict",
            "reason": "low_confidence",
            "user_note": (user_note or "").strip() or None,
            "review_status": "pending",
        }
        if quality_json is not None:
            payload["quality_json"] = quality_json
        if client_flow_version:
            payload["client_flow_version"] = client_flow_version.strip()

        supabase.table("ai_feedback_cases").insert(payload).execute()

        return {"status": "success", "image_url": image_url}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /ai-feedback/low-confidence error: {e}")
        raise HTTPException(status_code=500, detail=_supabase_feedback_error_detail(e))


@router.post("/prediction-jobs")
async def create_prediction_job(
    request: Request,
    background_tasks: BackgroundTasks,
    selected_plant: str = Form(...),
    file: UploadFile = File(...),
):
    check_rate_limit(request, "prediction_jobs", PREDICT_RATE_LIMIT_PER_MINUTE)
    auth_func = (
        require_authenticated_user if PREDICT_REQUIRE_AUTH else optional_authenticated_user
    )
    user_id = await run_in_threadpool(auth_func, request)

    selected_plant = selected_plant.strip()
    if not selected_plant:
        raise HTTPException(status_code=400, detail="selected_plant is required")

    _validate_upload_metadata(file)
    contents = await file.read()
    contents, content_type = _sanitize_image(contents)
    image_url = await run_in_threadpool(_upload_image_to_supabase, contents, content_type)

    row = {
        "created_by": user_id,
        "selected_plant": selected_plant,
        "image_url": image_url,
        "status": "pending",
        "source": "ios",
    }
    inserted = await run_in_threadpool(_insert_prediction_job, row)
    job_id = inserted["id"]
    background_tasks.add_task(process_prediction_job, job_id)
    return {"status": "accepted", "job_id": job_id, "image_url": image_url}


@router.get("/prediction-jobs/{job_id}")
async def get_prediction_job(job_id: str, request: Request):
    job = await run_in_threadpool(_get_prediction_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Prediction job not found")

    owner = job.get("created_by")
    if owner:
        user_id = await run_in_threadpool(require_authenticated_user, request)
        if user_id != owner:
            raise HTTPException(status_code=403, detail="You cannot access this prediction job")

    return _prediction_job_response(job)


@router.post("/prediction-jobs/{job_id}/process")
async def process_prediction_job_endpoint(job_id: str, request: Request):
    check_rate_limit(request, "prediction_jobs_process", PREDICT_RATE_LIMIT_PER_MINUTE)
    job = await run_in_threadpool(_get_prediction_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Prediction job not found")

    owner = job.get("created_by")
    if owner:
        user_id = await run_in_threadpool(require_authenticated_user, request)
        if user_id != owner:
            raise HTTPException(status_code=403, detail="You cannot access this prediction job")

    await run_in_threadpool(process_prediction_job, job_id)
    refreshed = await run_in_threadpool(_get_prediction_job, job_id)
    return _prediction_job_response(refreshed or job)


@router.post("/predict")
async def predict(
    request: Request,
    selected_plant: str = Form(...),
    file: UploadFile = File(...),
    client_quality_json: Optional[str] = Form(None),
    client_flow_version: Optional[str] = Form(None),
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

        _validate_upload_metadata(file)
        contents = await file.read()
        contents, content_type = _sanitize_image(contents)
        quality_json = _parse_client_quality_json(client_quality_json)

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

        confidence_value = parse_confidence_value(result.get("confidence"))
        if confidence_value is None:
            raise HTTPException(
                status_code=502,
                detail="Prediction service returned invalid confidence",
            )
        confidence_percent_str = f"{confidence_value:.2f}%"

        if confidence_value < PREDICT_UNRECOGNIZED_THRESHOLD:
            return {
                "status": "unrecognized",
                "message": UNRECOGNIZED_MESSAGE,
                "plant": predicted_plant,
                "disease": disease_name,
                "confidence": confidence_percent_str,
                "image_url": None,
            }

        image_url = await run_in_threadpool(
            _upload_image_to_supabase, contents, content_type
        )
        await run_in_threadpool(
            _log_low_confidence_case,
            user_id=user_id,
            plant=predicted_plant,
            disease=disease_name,
            confidence=confidence_value,
            image_url=image_url,
            quality_json=quality_json,
            client_flow_version=client_flow_version,
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
