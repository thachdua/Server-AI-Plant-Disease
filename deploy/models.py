from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SaveHistoryRequest(APIModel):
    plant: Optional[str] = Field(default=None, max_length=120)
    disease: Optional[str] = Field(default=None, max_length=180)
    confidence: Optional[float] = None
    image_url: Optional[str] = Field(default=None, max_length=2000)
    lat: Optional[float] = None
    lng: Optional[float] = None


class LLMAdviceDiagnosisRequest(APIModel):
    plant: Optional[str] = Field(default=None, max_length=120)
    disease: str = Field(min_length=1, max_length=180)
    confidence: Optional[float] = None
    user_note: Optional[str] = Field(default=None, max_length=1200)
    weather_snapshot: Optional[Dict[str, Any]] = None


class LLMAdviceWeatherRequest(APIModel):
    lat: float
    lng: float
    plant: Optional[str] = Field(default=None, max_length=120)
    disease: Optional[str] = Field(default=None, max_length=180)
    care_context: Optional[Dict[str, Any]] = None
    weather_snapshot: Optional[Dict[str, Any]] = None


class LLMChatMessage(APIModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=2000)


class LLMChatRequest(APIModel):
    messages: List[LLMChatMessage] = Field(min_length=1, max_length=20)
    mode: Optional[Literal["agriculture", "general"]] = None


class LLMCarePlanDiagnosisRequest(APIModel):
    plant: Optional[str] = Field(default=None, max_length=120)
    disease: str = Field(min_length=1, max_length=180)
    confidence: Optional[float] = None
    user_note: Optional[str] = Field(default=None, max_length=1200)
    weather_snapshot: Optional[Dict[str, Any]] = None
