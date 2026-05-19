from fastapi import APIRouter, HTTPException, Request

from deploy.auth import extract_bearer_token, get_user_id_from_supabase
from deploy.database import save_to_db
from deploy.models import SaveHistoryRequest

router = APIRouter()


@router.post("/history/save")
async def save_history(req: SaveHistoryRequest, request: Request):
    access_token = extract_bearer_token(request)
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    created_by = get_user_id_from_supabase(access_token)
    if not created_by:
        raise HTTPException(status_code=401, detail="Invalid token")

    if not req.disease or not req.image_url:
        raise HTTPException(status_code=400, detail="Missing required fields")

    save_to_db(req.plant, req.disease, req.confidence, req.image_url, created_by=created_by)
    return {"status": "success"}
