from __future__ import annotations

import json

import requests
from fastapi import HTTPException

from deploy.config import (
    GEMINI_API_KEYS,
    GEMINI_API_VERSION,
    GEMINI_CHAT_MODEL,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
)
from deploy.utils import canonical_json


def clean_markdown(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n")
    t = t.replace("**", "")
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if s.startswith("* "):
            s = "• " + s[2:].strip()
        elif s.startswith("- "):
            s = "• " + s[2:].strip()
        elif s.startswith("• "):
            s = "• " + s[2:].strip()
        lines.append(s)
    out = "\n".join(lines).strip()
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")
    return out


def call_gemini_json(system_prompt: str, user_payload: dict) -> dict:
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": canonical_json(user_payload)}],
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }

    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    last_error = None
    for api_key in GEMINI_API_KEYS:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{model}:generateContent"
            try:
                r = requests.post(url, params={"key": api_key}, json=body, timeout=25)
            except Exception as e:
                last_error = f"request_failed: {e}"
                continue

            if r.status_code == 200:
                data = r.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    raise HTTPException(
                        status_code=502, detail=f"Gemini bad response: {str(data)[:300]}"
                    )

                text = (text or "").strip()
                try:
                    return json.loads(text)
                except Exception:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Gemini returned invalid JSON: {text[:300]}",
                    )

            if r.status_code == 503:
                last_error = f"Gemini error: 503 {r.text}"
                continue

            if r.status_code in (401, 403, 429):
                last_error = f"Gemini key error: {r.status_code} {r.text}"
                break

            raise HTTPException(status_code=502, detail=f"Gemini error: {r.status_code} {r.text}")

    raise HTTPException(status_code=502, detail=last_error or "Gemini unavailable")


def call_gemini_text(
    system_prompt: str, user_text: str, model_override: str | None = None
) -> str:
    if not GEMINI_API_KEYS:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.3},
    }
    primary = model_override or GEMINI_MODEL
    models_to_try = [primary] + [m for m in GEMINI_FALLBACK_MODELS if m != primary]
    last_error = None
    for api_key in GEMINI_API_KEYS:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{model}:generateContent"
            try:
                r = requests.post(url, params={"key": api_key}, json=body, timeout=25)
            except Exception as e:
                last_error = f"request_failed: {e}"
                continue
            if r.status_code == 200:
                data = r.json()
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception:
                    raise HTTPException(
                        status_code=502, detail=f"Gemini bad response: {str(data)[:300]}"
                    )
                return clean_markdown((text or "").strip())
            if r.status_code == 503:
                last_error = f"Gemini error: 503 {r.text}"
                continue
            if r.status_code in (401, 403, 429):
                last_error = f"Gemini key error: {r.status_code} {r.text}"
                break
            raise HTTPException(status_code=502, detail=f"Gemini error: {r.status_code} {r.text}")
    raise HTTPException(status_code=502, detail=last_error or "Gemini unavailable")


def validate_advice_json(advice: dict) -> dict:
    if not isinstance(advice, dict):
        raise HTTPException(status_code=502, detail="LLM output is not an object")
    required_lists = ["symptoms", "causes", "treatments", "prevention"]
    for k in required_lists:
        v = advice.get(k)
        if not isinstance(v, list):
            advice[k] = []
        else:
            advice[k] = [str(x) for x in v if str(x).strip()][:10]

    summary = advice.get("summary_vi")
    if not isinstance(summary, str) or not summary.strip():
        summary = (
            "Gợi ý tham khảo: theo dõi triệu chứng, vệ sinh vườn, "
            "và xử lý theo khuyến cáo địa phương."
        )
    advice["summary_vi"] = summary.strip()

    w = advice.get("when_to_seek_expert")
    if not isinstance(w, str) or not w.strip():
        w = (
            "Nếu triệu chứng lan nhanh, cây suy kiệt, hoặc bạn không chắc chắn về chẩn đoán, "
            "hãy liên hệ chuyên gia."
        )
    advice["when_to_seek_expert"] = w.strip()
    return advice


def diagnosis_fallback_advice(
    plant: str | None, disease: str, confidence: float | None
) -> dict:
    disease_norm = (disease or "").strip()
    plant_norm = (plant or "").strip()
    header = f"{plant_norm + ' - ' if plant_norm else ''}{disease_norm or 'Bệnh lá'}"
    conf_text = (
        f" (độ tin cậy ~{confidence:.0f}%)"
        if isinstance(confidence, (int, float))
        else ""
    )
    return validate_advice_json(
        {
            "summary_vi": (
                f"Gợi ý tham khảo cho {header}{conf_text}: ưu tiên vệ sinh vườn, "
                "cắt bỏ lá/cành bệnh, giảm ẩm và theo dõi lây lan. "
                "(Hệ thống tư vấn AI đang quá tải, đây là gợi ý mặc định.)"
            ),
            "symptoms": [
                "Đốm/loang màu bất thường trên lá, có thể kèm viền sẫm",
                "Lá vàng, khô mép, rụng lá sớm",
                "Vết bệnh lan rộng nhanh khi ẩm cao hoặc sau mưa",
            ],
            "causes": [
                "Độ ẩm cao, lá ướt lâu sau mưa/tưới phun",
                "Tán lá rậm, thông thoáng kém",
                "Nguồn bệnh còn tồn dư trên lá rụng/dụng cụ chưa khử trùng",
            ],
            "treatments": [
                "Cắt bỏ và tiêu huỷ phần lá/cành bệnh; khử trùng kéo/dụng cụ sau khi cắt",
                "Tăng thông thoáng: tỉa tán, làm sạch cỏ dại; hạn chế nước đọng",
                "Tránh tưới phun lên lá vào chiều tối; ưu tiên tưới gốc",
            ],
            "prevention": [
                "Duy trì vườn sạch, thu gom lá rụng; luân canh nếu phù hợp",
                "Theo dõi sau mưa/ẩm cao để phát hiện sớm",
                "Tham khảo cán bộ khuyến nông/nhà vườn địa phương khi cần phun phòng trị",
            ],
            "when_to_seek_expert": (
                "Nếu vết bệnh lan rất nhanh, cây suy kiệt, hoặc bạn không chắc chẩn đoán, "
                "hãy gửi ảnh và liên hệ chuyên gia để được hướng dẫn."
            ),
        }
    )


def is_cache_expired(kind: str, updated_at) -> bool:
    from datetime import datetime

    if kind != "weather":
        return False
    try:
        if not updated_at:
            return True
        age = datetime.now().astimezone() - updated_at
        return age.total_seconds() > 6 * 3600
    except Exception:
        return True
