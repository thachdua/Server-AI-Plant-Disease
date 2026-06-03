from fastapi import APIRouter, Response, status

from deploy.config import HF_API_URL, config_status

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Render Bridge is Online", "target": "Hugging Face Space", "hf": HF_API_URL}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ready")
def readiness(response: Response):
    cfg = config_status()
    if not cfg["ok"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if cfg["ok"] else "degraded",
        "config": cfg,
    }
