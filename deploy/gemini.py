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


def validate_care_plan_json(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise HTTPException(status_code=502, detail="Care plan output is not an object")

    summary = plan.get("summary_vi")
    if not isinstance(summary, str) or not summary.strip():
        summary = "Lịch chăm sóc tham khảo sau chẩn đoán: theo dõi cây, giảm ẩm và xử lý an toàn."

    allowed_categories = {
        "watering",
        "misting",
        "fertilizing",
        "treatment",
        "inspection",
        "rotation",
        "repotting",
        "cleanup",
    }
    allowed_repeat = {"none", "daily", "weekly", "monthly", "yearly"}
    tasks = []
    raw_tasks = plan.get("tasks")
    if isinstance(raw_tasks, list):
        for raw in raw_tasks[:12]:
            if not isinstance(raw, dict):
                continue
            title = str(raw.get("title") or "").strip()
            if not title:
                continue
            detail = str(raw.get("detail") or "").strip()
            category = str(raw.get("category") or "inspection").strip().lower()
            if category not in allowed_categories:
                category = "inspection"
            repeat_rule = str(raw.get("repeat_rule") or "none").strip().lower()
            if repeat_rule not in allowed_repeat:
                repeat_rule = "none"
            try:
                due_in_days = int(raw.get("due_in_days") or 0)
            except Exception:
                due_in_days = 0
            due_in_days = max(0, min(due_in_days, 365))
            try:
                reminder_hour = int(raw.get("reminder_hour") or 7)
            except Exception:
                reminder_hour = 7
            reminder_hour = max(6, min(reminder_hour, 20))
            tasks.append(
                {
                    "title": title[:120],
                    "detail": detail[:500],
                    "category": category,
                    "due_in_days": due_in_days,
                    "repeat_rule": repeat_rule,
                    "reminder_hour": reminder_hour,
                }
            )

    if not tasks:
        tasks = [
            {
                "title": "Kiểm tra triệu chứng",
                "detail": "Quan sát lá/cành bị bệnh, chụp lại nếu vết bệnh lan nhanh.",
                "category": "inspection",
                "due_in_days": 0,
                "repeat_rule": "daily",
                "reminder_hour": 7,
            },
            {
                "title": "Tưới gốc vừa đủ",
                "detail": "Tránh tưới phun lên lá, đặc biệt vào chiều tối.",
                "category": "watering",
                "due_in_days": 1,
                "repeat_rule": "none",
                "reminder_hour": 7,
            },
        ]

    checklist = plan.get("checklist")
    if not isinstance(checklist, list):
        checklist = []
    checklist = [str(x).strip() for x in checklist if str(x).strip()][:12]
    if not checklist:
        checklist = ["Cắt bỏ lá bệnh nếu có", "Giữ vườn thông thoáng", "Theo dõi sau mưa/ẩm cao"]

    safety_note = plan.get("safety_note")
    if not isinstance(safety_note, str) or not safety_note.strip():
        safety_note = "Thông tin chỉ mang tính tham khảo; không tự ý dùng hoá chất liều cao."

    return {
        "summary_vi": summary.strip(),
        "tasks": tasks,
        "checklist": checklist,
        "safety_note": safety_note.strip(),
    }


def validate_care_metrics_json(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        raise HTTPException(status_code=502, detail="Care metrics output is not an object")

    def float_or_none(value):
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    water_ml = float_or_none(metrics.get("water_ml_per_day"))
    if water_ml is not None:
        water_ml = min(max(water_ml, 20), 3000)

    cups = float_or_none(metrics.get("cup_count_per_day"))
    if cups is None and water_ml is not None:
        cups = water_ml / 250.0
    if cups is not None:
        cups = min(max(cups, 0.1), 12.0)

    light_min = float_or_none(metrics.get("light_min_lux"))
    light_max = float_or_none(metrics.get("light_max_lux"))
    if light_min is None:
        light_min = 250
    if light_max is None:
        light_max = max(light_min + 500, 1000)
    light_min = min(max(light_min, 20), 5000)
    light_max = min(max(light_max, light_min + 50), 8000)

    status = str(metrics.get("light_status_vi") or "").strip()
    allowed_status = {"Thiếu sáng", "Phù hợp", "Quá sáng", "Chưa đo"}
    if status not in allowed_status:
        status = "Chưa đo"

    water_advice = str(metrics.get("water_advice_vi") or "").strip()
    if not water_advice:
        water_advice = "Tưới từ từ vào gốc, kiểm tra đất trước khi tưới và điều chỉnh nếu đất còn ướt."

    light_advice = str(metrics.get("light_advice_vi") or "").strip()
    if not light_advice:
        light_advice = "Đo ánh sáng tại vị trí đặt cây và điều chỉnh dần để tránh sốc nắng."

    return {
        "water_ml_per_day": round(water_ml, 1) if water_ml is not None else None,
        "cup_count_per_day": round(cups, 2) if cups is not None else None,
        "water_advice_vi": water_advice[:700],
        "light_min_lux": round(light_min),
        "light_max_lux": round(light_max),
        "light_status_vi": status,
        "light_advice_vi": light_advice[:700],
    }


def care_metrics_fallback(
    plant: str | None,
    disease: str | None,
    confidence: float | None = None,
    pot_diameter_cm: float | None = None,
    plant_height_cm: float | None = None,
    measured_lux: float | None = None,
) -> dict:
    text = f"{plant or ''} {disease or ''}".lower()

    def has_any(words):
        return any(w in text for w in words)

    plant_text = (plant or "").lower()
    disease_text = (disease or "").lower()
    disease_is_leaf_spot = has_any([
        "bacterial", "vi khuẩn", "spot", "đốm", "blight", "cháy lá", "mold",
        "mildew", "mốc", "nấm", "rot", "thối", "rust", "rỉ sắt"
    ])
    disease_is_wilt = has_any(["wilt", "héo", "nematode", "tuyến trùng"])

    if any(w in plant_text for w in ["rice", "lúa", "corn", "ngô", "maize", "mía", "sugercane"]):
        light_min, light_max = 1200, 3200
    elif any(w in plant_text for w in [
        "tomato", "cà chua", "pepper", "bell", "ớt", "potato", "khoai",
        "grape", "nho", "watermelon", "dưa", "strawberry", "dâu", "rose", "hồng"
    ]):
        light_min, light_max = 800, 2200
    elif any(w in plant_text for w in ["coffee", "cà phê", "blueberry", "việt quất"]):
        light_min, light_max = 450, 1400
    elif any(w in plant_text for w in ["cassava", "sắn", "soybean", "đậu"]):
        light_min, light_max = 700, 1800
    else:
        light_min, light_max = 350, 1200

    if disease_is_leaf_spot:
        light_min = int(light_min * 1.05)
        light_max = int(light_max * 0.95)

    if measured_lux is None or measured_lux <= 0:
        light_status = "Chưa đo"
        light_advice = (
            f"Khoảng tham khảo cho cây này là {light_min}-{light_max} lux. "
            "Hãy đo tại đúng vị trí đặt cây để có đánh giá cụ thể hơn."
        )
    elif measured_lux < light_min:
        light_status = "Thiếu sáng"
        light_advice = (
            f"Vị trí này thấp hơn khoảng {light_min}-{light_max} lux. "
            "Tăng sáng từ từ, ưu tiên ánh sáng tán xạ để cây hồi phục tốt hơn."
        )
    elif measured_lux > light_max:
        light_status = "Quá sáng"
        light_advice = (
            f"Vị trí này cao hơn khoảng {light_min}-{light_max} lux. "
            "Giảm nắng gắt bằng rèm mỏng hoặc đặt cây lùi xa nguồn nắng trực tiếp."
        )
    else:
        light_status = "Phù hợp"
        light_advice = (
            f"Vị trí này nằm trong khoảng {light_min}-{light_max} lux cho cây/bệnh hiện tại. "
            "Giữ vị trí này và theo dõi lá non trong vài ngày."
        )

    water_ml = None
    cups = None
    if pot_diameter_cm and pot_diameter_cm > 0:
        height_factor = min(max((plant_height_cm or 30) / 30.0, 0.7), 1.8)
        base = pot_diameter_cm * pot_diameter_cm * 0.55 * height_factor
        if any(w in plant_text for w in ["rice", "lúa"]):
            base *= 1.25
        if any(w in plant_text for w in ["cactus", "xương rồng", "succulent", "sen đá"]):
            base *= 0.45
        if disease_is_leaf_spot:
            base *= 0.88
        if disease_is_wilt:
            base *= 0.95
        water_ml = min(max(base, 60), 1800)
        cups = water_ml / 250.0

    if water_ml is None:
        water_advice = "Nhập đường kính chậu và chiều cao cây để ước lượng ml/ngày. Khi cây bệnh, luôn kiểm tra đất trước khi tưới."
    elif disease_is_leaf_spot:
        water_advice = (
            f"Tưới khoảng {water_ml:.0f} ml/ngày vào gốc, không tưới lên lá và tránh tưới chiều tối "
            "để giảm ẩm kéo dài trên tán lá."
        )
    elif disease_is_wilt:
        water_advice = (
            f"Tưới khoảng {water_ml:.0f} ml/ngày, chia chậm quanh gốc và theo dõi cây héo do thiếu nước hay do bệnh rễ."
        )
    else:
        water_advice = (
            f"Tưới khoảng {water_ml:.0f} ml/ngày, điều chỉnh giảm nếu đất còn ướt hoặc tăng nhẹ nếu đất khô nhanh."
        )

    return {
        "water_ml_per_day": round(water_ml, 1) if water_ml is not None else None,
        "cup_count_per_day": round(cups, 2) if cups is not None else None,
        "water_advice_vi": water_advice,
        "light_min_lux": light_min,
        "light_max_lux": light_max,
        "light_status_vi": light_status,
        "light_advice_vi": light_advice,
    }


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
