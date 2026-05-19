import requests
from fastapi import APIRouter, HTTPException

from deploy.cache import cache_get, cache_set
from deploy.config import GEMINI_CHAT_MODEL, GEMINI_MODEL, OPENWEATHER_API_KEY
from deploy.database import llm_cache_get, llm_cache_upsert
from deploy.gemini import (
    call_gemini_json,
    call_gemini_text,
    diagnosis_fallback_advice,
    is_cache_expired,
    validate_advice_json,
)
from deploy.models import LLMAdviceDiagnosisRequest, LLMAdviceWeatherRequest, LLMChatRequest
from deploy.prompts import DIAGNOSIS_SYSTEM_PROMPT, WEATHER_SYSTEM_PROMPT
from deploy.utils import canonical_json, sha256

router = APIRouter()


@router.post("/llm/chat")
async def llm_chat(req: LLMChatRequest):
    try:
        msgs = (req.messages or [])[-12:]
        convo = []
        for m in msgs:
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            if not text:
                continue
            if role == "assistant":
                convo.append(f"Trợ lý: {text}")
            else:
                convo.append(f"Người dùng: {text}")
        convo_text = "\n".join(convo).strip()
        mode = (req.mode or "agriculture").strip().lower()
        if mode == "general":
            system_prompt = (
                "Bạn là trợ lý AI tổng quát. Trả lời tiếng Việt, ngắn gọn, rõ ràng.\n"
                "Không dùng markdown (không dùng dấu * hoặc **), không dùng tiêu đề dạng ###.\n"
                "Nếu cần liệt kê, dùng ký tự '•' và xuống dòng.\n"
                "Nếu người dùng hỏi về nông nghiệp/bệnh cây, hãy trả lời theo hướng an toàn và thực tế."
            )
        else:
            system_prompt = (
                "Bạn là trợ lý nông nghiệp. Trả lời tiếng Việt, ngắn gọn, rõ ràng.\n"
                "Không dùng markdown (không dùng dấu * hoặc **), không dùng tiêu đề dạng ###.\n"
                "Nếu cần liệt kê, dùng ký tự '•' và xuống dòng.\n"
                "Tránh đưa liều lượng/hoá chất nguy hiểm; ưu tiên IPM; khuyến nghị hỏi khuyến nông địa phương khi cần."
            )
        reply = call_gemini_text(
            system_prompt, convo_text + "\nTrợ lý:", model_override=GEMINI_CHAT_MODEL
        )
        if not reply.strip():
            reply = "Mình chưa nhận được nội dung trả lời. Bạn thử hỏi lại giúp mình nhé."
        return {"status": "success", "reply": reply}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /llm/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/advice/diagnosis")
async def llm_advice_diagnosis(req: LLMAdviceDiagnosisRequest):
    try:
        payload = {
            "plant": req.plant,
            "disease": req.disease,
            "confidence": req.confidence,
            "user_note": req.user_note,
            "weather_snapshot": req.weather_snapshot,
            "lang": "vi",
        }
        input_hash = sha256(canonical_json(payload))
        cached = llm_cache_get("diagnosis", input_hash, "vi")
        if cached:
            return {
                "status": "success",
                "cached": True,
                "model": cached.get("model"),
                "advice": cached.get("content_json"),
                "summary_vi": cached.get("content_text"),
            }

        try:
            raw = call_gemini_json(DIAGNOSIS_SYSTEM_PROMPT, payload)
            advice = validate_advice_json(raw)
            llm_cache_upsert(
                "diagnosis", input_hash, "vi", GEMINI_MODEL, advice, advice.get("summary_vi")
            )
            return {
                "status": "success",
                "cached": False,
                "model": GEMINI_MODEL,
                "advice": advice,
                "summary_vi": advice.get("summary_vi"),
            }
        except HTTPException as e:
            if e.status_code == 502 and "503" in str(e.detail):
                fallback_key = f"llm_fallback|diagnosis|{input_hash}"
                cached_fb = cache_get(fallback_key)
                if cached_fb is None:
                    cached_fb = diagnosis_fallback_advice(
                        req.plant, req.disease, req.confidence
                    )
                    cache_set(fallback_key, cached_fb, ttl_seconds=300)
                return {
                    "status": "success",
                    "cached": False,
                    "fallback": True,
                    "model": "fallback",
                    "advice": cached_fb,
                    "summary_vi": cached_fb.get("summary_vi"),
                }
            raise e
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /llm/advice/diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/advice/weather")
async def llm_advice_weather(req: LLMAdviceWeatherRequest):
    try:
        snapshot = req.weather_snapshot
        if snapshot is None:
            if not OPENWEATHER_API_KEY:
                raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")
            url = "https://api.openweathermap.org/data/3.0/onecall"
            r = requests.get(
                url,
                params={
                    "lat": req.lat,
                    "lon": req.lng,
                    "appid": OPENWEATHER_API_KEY,
                    "units": "metric",
                    "lang": "vi",
                    "exclude": "minutely",
                },
                timeout=12,
            )
            if r.status_code != 200:
                raise HTTPException(
                    status_code=502, detail=f"OpenWeather error: {r.status_code} {r.text}"
                )
            data = r.json()
            current = data.get("current") or {}
            snapshot = {
                "temp": current.get("temp"),
                "humidity": current.get("humidity"),
                "wind_speed": current.get("wind_speed"),
                "rain_1h": (current.get("rain") or {}).get("1h"),
                "weather": (current.get("weather") or [])[:1],
            }

        payload = {
            "lat": round(req.lat, 3),
            "lng": round(req.lng, 3),
            "snapshot": snapshot,
            "lang": "vi",
        }
        input_hash = sha256(canonical_json(payload))
        cached = llm_cache_get("weather", input_hash, "vi")
        if cached and not is_cache_expired("weather", cached.get("updated_at")):
            return {
                "status": "success",
                "cached": True,
                "model": cached.get("model"),
                "advice": cached.get("content_json"),
                "summary_vi": cached.get("content_text"),
            }

        try:
            raw = call_gemini_json(WEATHER_SYSTEM_PROMPT, payload)
        except HTTPException as e:
            if cached:
                return {
                    "status": "success",
                    "cached": True,
                    "stale": True,
                    "model": cached.get("model"),
                    "advice": cached.get("content_json"),
                    "summary_vi": cached.get("content_text"),
                }
            raise e

        advice = validate_advice_json(raw)
        llm_cache_upsert(
            "weather", input_hash, "vi", GEMINI_MODEL, advice, advice.get("summary_vi")
        )
        return {
            "status": "success",
            "cached": False,
            "model": GEMINI_MODEL,
            "advice": advice,
            "summary_vi": advice.get("summary_vi"),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /llm/advice/weather error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
