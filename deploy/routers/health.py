from fastapi import APIRouter

from deploy.config import HF_API_URL

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Render Bridge is Online", "target": "Hugging Face Space", "hf": HF_API_URL}
