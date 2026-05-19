from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class SaveHistoryRequest(BaseModel):
    plant: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    image_url: Optional[str] = None


class LLMAdviceDiagnosisRequest(BaseModel):
    plant: Optional[str] = None
    disease: str
    confidence: Optional[float] = None
    user_note: Optional[str] = None
    weather_snapshot: Optional[Dict[str, Any]] = None


class LLMAdviceWeatherRequest(BaseModel):
    lat: float
    lng: float
    weather_snapshot: Optional[Dict[str, Any]] = None


class LLMChatRequest(BaseModel):
    messages: List[dict]
    mode: Optional[str] = None
