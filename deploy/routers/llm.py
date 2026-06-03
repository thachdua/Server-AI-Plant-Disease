import requests
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from deploy.cache import cache_get, cache_set
from deploy.config import (
    GEMINI_CHAT_MODEL,
    GEMINI_MODEL,
    LLM_CHAT_MAX_CHARS,
    LLM_RATE_LIMIT_PER_MINUTE,
    OPENWEATHER_API_KEY,
)
from deploy.database import llm_cache_get, llm_cache_upsert
from deploy.gemini import (
    call_gemini_json,
    call_gemini_text,
    diagnosis_fallback_advice,
    is_cache_expired,
    validate_advice_json,
    validate_care_plan_json,
)
from deploy.models import (
    LLMCarePlanDiagnosisRequest,
    LLMAdviceDiagnosisRequest,
    LLMAdviceWeatherRequest,
    LLMChatRequest,
)
from deploy.prompts import CARE_PLAN_SYSTEM_PROMPT, DIAGNOSIS_SYSTEM_PROMPT, WEATHER_SYSTEM_PROMPT
from deploy.rate_limit import check_rate_limit
from deploy.utils import canonical_json, sha256
from deploy.validation import validate_coordinates

router = APIRouter()


@router.post("/llm/chat")
async def llm_chat(req: LLMChatRequest, request: Request):
    try:
        check_rate_limit(request, "llm", LLM_RATE_LIMIT_PER_MINUTE)
        msgs = (req.messages or [])[-12:]
        convo = []
        total_chars = 0
        for m in msgs:
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            if not text:
                continue
            total_chars += len(text)
            if total_chars > LLM_CHAT_MAX_CHARS:
                raise HTTPException(status_code=413, detail="Chat prompt is too long")
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
        reply = await run_in_threadpool(
            call_gemini_text,
            system_prompt,
            convo_text + "\nTrợ lý:",
            model_override=GEMINI_CHAT_MODEL,
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
async def llm_advice_diagnosis(req: LLMAdviceDiagnosisRequest, request: Request):
    try:
        check_rate_limit(request, "llm", LLM_RATE_LIMIT_PER_MINUTE)
        disease = (req.disease or "").strip()
        if not disease:
            raise HTTPException(status_code=400, detail="disease is required")
        payload = {
            "plant": req.plant,
            "disease": disease,
            "confidence": req.confidence,
            "user_note": req.user_note,
            "weather_snapshot": req.weather_snapshot,
            "lang": "vi",
        }
        input_hash = sha256(canonical_json(payload))
        cached = await run_in_threadpool(llm_cache_get, "diagnosis", input_hash, "vi")
        if cached:
            return {
                "status": "success",
                "cached": True,
                "model": cached.get("model"),
                "advice": cached.get("content_json"),
                "summary_vi": cached.get("content_text"),
            }

        try:
            raw = await run_in_threadpool(call_gemini_json, DIAGNOSIS_SYSTEM_PROMPT, payload)
            advice = validate_advice_json(raw)
            await run_in_threadpool(
                llm_cache_upsert,
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
async def llm_advice_weather(req: LLMAdviceWeatherRequest, request: Request):
    try:
        check_rate_limit(request, "llm", LLM_RATE_LIMIT_PER_MINUTE)
        validate_coordinates(req.lat, req.lng)
        snapshot = req.weather_snapshot
        if snapshot is None:
            if not OPENWEATHER_API_KEY:
                raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")
            url = "https://api.openweathermap.org/data/3.0/onecall"
            r = await run_in_threadpool(
                requests.get,
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
            "plant": req.plant,
            "disease": req.disease,
            "care_context": req.care_context,
            "snapshot": snapshot,
            "lang": "vi",
        }
        input_hash = sha256(canonical_json(payload))
        cached = await run_in_threadpool(llm_cache_get, "weather", input_hash, "vi")
        if cached and not is_cache_expired("weather", cached.get("updated_at")):
            return {
                "status": "success",
                "cached": True,
                "model": cached.get("model"),
                "advice": cached.get("content_json"),
                "summary_vi": cached.get("content_text"),
            }

        try:
            raw = await run_in_threadpool(call_gemini_json, WEATHER_SYSTEM_PROMPT, payload)
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
        await run_in_threadpool(
            llm_cache_upsert,
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


@router.post("/llm/care-plan/diagnosis")
async def llm_care_plan_diagnosis(req: LLMCarePlanDiagnosisRequest, request: Request):
    try:
        check_rate_limit(request, "llm", LLM_RATE_LIMIT_PER_MINUTE)
        disease = (req.disease or "").strip()
        if not disease:
            raise HTTPException(status_code=400, detail="disease is required")

        payload = {
            "plant": req.plant,
            "disease": disease,
            "confidence": req.confidence,
            "user_note": req.user_note,
            "weather_snapshot": req.weather_snapshot,
            "lang": "vi",
        }
        input_hash = sha256(canonical_json(payload))
        cached = await run_in_threadpool(llm_cache_get, "care_plan", input_hash, "vi")
        if cached:
            return {
                "status": "success",
                "cached": True,
                "model": cached.get("model"),
                "care_plan": cached.get("content_json"),
                "summary_vi": cached.get("content_text"),
            }

        raw = await run_in_threadpool(call_gemini_json, CARE_PLAN_SYSTEM_PROMPT, payload)
        care_plan = validate_care_plan_json(raw)
        await run_in_threadpool(
            llm_cache_upsert,
            "care_plan",
            input_hash,
            "vi",
            GEMINI_MODEL,
            care_plan,
            care_plan.get("summary_vi"),
        )
        return {
            "status": "success",
            "cached": False,
            "model": GEMINI_MODEL,
            "care_plan": care_plan,
            "summary_vi": care_plan.get("summary_vi"),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /llm/care-plan/diagnosis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
