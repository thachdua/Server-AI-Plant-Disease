from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from deploy.auth import require_authenticated_user
from deploy.database import save_to_db
from deploy.models import SaveHistoryRequest

router = APIRouter()


@router.post("/history/save")
async def save_history(req: SaveHistoryRequest, request: Request):
    created_by = await run_in_threadpool(require_authenticated_user, request)

    if not req.disease or not req.image_url:
        raise HTTPException(status_code=400, detail="Missing required fields")

    try:
        await run_in_threadpool(
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
    return {"status": "success"}
