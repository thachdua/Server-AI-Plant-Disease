from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from deploy.auth import require_authenticated_user
from deploy.database import save_outbreak_case, save_to_db
from deploy.models import SaveHistoryRequest
from deploy.validation import validate_coordinates

router = APIRouter()

OUTBREAK_CONFIDENCE_THRESHOLD = 60.0


def _is_healthy_disease(disease: str | None) -> bool:
    normalized = (disease or "").strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return (
        "healthy" in normalized
        or "khỏe mạnh" in normalized
        or "khoẻ mạnh" in normalized
        or "khoe manh" in normalized
    )


def _should_create_outbreak(req: SaveHistoryRequest) -> bool:
    if req.lat is None or req.lng is None:
        return False
    if req.confidence is None or req.confidence < OUTBREAK_CONFIDENCE_THRESHOLD:
        return False
    if _is_healthy_disease(req.disease):
        return False
    return True


@router.post("/history/save")
async def save_history(req: SaveHistoryRequest, request: Request):
    created_by = await run_in_threadpool(require_authenticated_user, request)

    if not req.disease or not req.image_url:
        raise HTTPException(status_code=400, detail="Missing required fields")
    if (req.lat is None) != (req.lng is None):
        raise HTTPException(status_code=400, detail="lat and lng must be provided together")
    if req.lat is not None and req.lng is not None:
        validate_coordinates(req.lat, req.lng)

    try:
        history_id = await run_in_threadpool(
            save_to_db,
            req.plant,
            req.disease,
            req.confidence,
            req.image_url,
            created_by,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        print(f"❌ Lỗi lưu DB: {exc}")
        raise HTTPException(status_code=500, detail="Could not save history")

    outbreak_saved = False
    if _should_create_outbreak(req):
        try:
            await run_in_threadpool(
                save_outbreak_case,
                lat=req.lat,
                lng=req.lng,
                plant=req.plant,
                disease=req.disease,
                confidence=req.confidence,
                image_url=req.image_url,
                history_id=history_id,
                created_by=created_by,
                location_label=req.location_label,
                province_name=req.province_name,
            )
            outbreak_saved = True
        except Exception as exc:
            print(f"⚠️ Không thể tạo outbreak case từ history: {exc}")

    return {"status": "success", "outbreak_saved": outbreak_saved}
