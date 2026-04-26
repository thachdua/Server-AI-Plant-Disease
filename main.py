import os
import io
import requests
import json
import hashlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import time
from supabase import create_client, Client
import psycopg2
from datetime import datetime
from datetime import timedelta
from google import genai
from google.genai import types

app = FastAPI()

# --- 1. CONFIG ---
# Link API Hugging Face mà bạn vừa tạo
HF_API_URL = "https://thachdua-plantdiseasedectect.hf.space/predict"

SUPABASE_URL = "https://wxmmfmvyefruyknymvdm.supabase.co"
SUPABASE_KEY = "sb_publishable_gW6BTa8IWvVjO6Rbe-OGxQ_WgD6knWz"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Simple in-memory cache (per instance) ---
_cache = {}  # key -> (expires_at_epoch, value)

def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        _cache.pop(key, None)
        return None
    return val

def cache_set(key: str, val, ttl_seconds: int):
    _cache[key] = (time.time() + ttl_seconds, val)

# Config DB dùng Pooler (Cổng 6543) cho ổn định trên Cloud
DB_CONFIG = {
    "host": "aws-1-ap-southeast-1.pooler.supabase.com",
    "database": "postgres",
    "user": "postgres.wxmmfmvyefruyknymvdm",
    "password": os.environ.get("DB_PASSWORD", "Nguyenyeuloc@123"),
    "port": 6543,
    "sslmode": "require"
}

# --- 2. HÀM LƯU DATABASE ---
def save_to_db(plant_name, disease_name, confidence, image_url, created_by=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        # Prefer inserting created_by (if you added the column). Fallback to old schema if needed.
        try:
            query = """
            INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(query, (plant_name, disease_name, confidence, image_url, datetime.now(), created_by))
        except Exception:
            conn.rollback()
            query = """
            INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(query, (plant_name, disease_name, confidence, image_url, datetime.now()))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Đã lưu lịch sử vào Supabase")
    except Exception as e:
        print(f"❌ Lỗi lưu DB: {e}")


def _canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _llm_cache_get(kind: str, input_hash: str, lang: str = "vi"):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT content_json, content_text, model, updated_at
            FROM llm_advice_cache
            WHERE kind = %s AND input_hash = %s AND lang = %s
            LIMIT 1
            """,
            (kind, input_hash, lang),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return None
        content_json, content_text, model, updated_at = row
        return {
            "content_json": content_json,
            "content_text": content_text,
            "model": model,
            "updated_at": updated_at,
        }
    except Exception as e:
        print(f"❌ Lỗi đọc llm_advice_cache: {e}")
        return None


def _llm_cache_upsert(kind: str, input_hash: str, lang: str, model: str, content_json, content_text: str | None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO llm_advice_cache (kind, input_hash, lang, model, content_json, content_text)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (kind, input_hash, lang)
            DO UPDATE SET
              model = EXCLUDED.model,
              content_json = EXCLUDED.content_json,
              content_text = EXCLUDED.content_text,
              updated_at = now()
            """,
            (kind, input_hash, lang, model, json.dumps(content_json, ensure_ascii=False), content_text),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi ghi llm_advice_cache: {e}")
        return False


def _call_gemini_json(system_prompt: str, user_payload: dict) -> dict:
    if not _gemini_client:
        raise HTTPException(status_code=500, detail="Missing GEMINI_API_KEY")

    user_text = _canonical_json(user_payload)
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_text,
        config=types.GenerateContentConfig(
            temperature=0.4,
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "summary_vi": {"type": "string"},
                    "symptoms": {"type": "array", "items": {"type": "string"}},
                    "causes": {"type": "array", "items": {"type": "string"}},
                    "treatments": {"type": "array", "items": {"type": "string"}},
                    "prevention": {"type": "array", "items": {"type": "string"}},
                    "when_to_seek_expert": {"type": "string"},
                },
                "required": ["summary_vi", "symptoms", "causes", "treatments", "prevention", "when_to_seek_expert"],
            },
        ),
    )
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid JSON: {text[:300]}")


def _validate_advice_json(advice: dict) -> dict:
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
        summary = "Gợi ý tham khảo: theo dõi triệu chứng, vệ sinh vườn, và xử lý theo khuyến cáo địa phương."
    advice["summary_vi"] = summary.strip()

    w = advice.get("when_to_seek_expert")
    if not isinstance(w, str) or not w.strip():
        w = "Nếu triệu chứng lan nhanh, cây suy kiệt, hoặc bạn không chắc chắn về chẩn đoán, hãy liên hệ chuyên gia."
    advice["when_to_seek_expert"] = w.strip()
    return advice


def _is_cache_expired(kind: str, updated_at) -> bool:
    # Weather: expire after 6 hours
    if kind != "weather":
        return False
    try:
        if not updated_at:
            return True
        age = datetime.now().astimezone() - updated_at
        return age.total_seconds() > 6 * 3600
    except Exception:
        return True


def extract_bearer_token(request: Request):
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts[0].strip(), parts[1].strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def get_user_id_from_supabase(access_token: str):
    """Best-effort: validate access token and return user id."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {access_token}",
                "apikey": SUPABASE_KEY,
                "Accept": "application/json",
            },
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("id")
    except Exception:
        return None

@app.get("/")
def home():
    return {"message": "Render Bridge is Online", "target": "Hugging Face Space"}

# --- Outbreak map (public read) ---
@app.get("/outbreaks")
def outbreaks(
    disease: Optional[str] = None,
    severity: Optional[int] = None,
    since: Optional[str] = None,
    limit: int = 500
):
    """
    Returns outbreak points from Supabase table `public.outbreak_cases`.
    Filters:
    - disease: ilike match (pass full string or pattern)
    - severity: 1..5
    - since: ISO timestamp (reported_at >= since)
    """
    cache_key = f"outbreaks|d={disease}|sev={severity}|since={since}|l={limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "items": cached}

    q = supabase.table("outbreak_cases").select("id,lat,lng,disease,severity,reported_at,note,source")
    if disease:
        q = q.ilike("disease", disease)
    if severity is not None:
        q = q.eq("severity", severity)
    if since:
        q = q.gte("reported_at", since)
    q = q.order("reported_at", desc=True).limit(min(max(limit, 1), 1000))

    resp = q.execute()
    items = resp.data or []
    cache_set(cache_key, items, ttl_seconds=20)
    return {"status": "success", "items": items}

def _point_in_bbox(lat: float, lng: float, bbox: list[float]) -> bool:
    # bbox format [minLng, minLat, maxLng, maxLat]
    return bbox[1] <= lat <= bbox[3] and bbox[0] <= lng <= bbox[2]

def _point_in_ring(lng: float, lat: float, ring: list[list[float]]) -> bool:
    # Ray casting on [lng, lat] points
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def _point_in_polygon(lng: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    # polygon: [ring1, ring2...] where ring1 is outer, others are holes
    if not polygon:
        return False
    if not _point_in_ring(lng, lat, polygon[0]):
        return False
    # holes
    for hole in polygon[1:]:
        if _point_in_ring(lng, lat, hole):
            return False
    return True

def _point_in_multipolygon(lng: float, lat: float, coords) -> bool:
    # coords: MultiPolygon -> [polygon]
    try:
        for polygon in coords:
            if _point_in_polygon(lng, lat, polygon):
                return True
    except Exception:
        return False
    return False

def _compute_level(count7d: int, max_sev: int | None) -> int:
    max_sev = max_sev or 0
    if count7d > 10 or max_sev >= 5:
        return 4
    if count7d >= 6 or max_sev >= 4:
        return 3
    if count7d >= 3 or max_sev >= 3:
        return 2
    if count7d >= 1:
        return 1
    return 0

# --- Outbreak areas: Phase 1 (province overlays) ---
@app.get("/outbreaks/areas")
def outbreak_areas(
    level: str = "province",
    since_days: int = 7,
    disease: Optional[str] = None,
    min_severity: Optional[int] = None,
    since: Optional[str] = None,
):
    """
    Phase 1: Province-only overlays for Hue (46), Da Nang (48), Quang Nam (49), Quang Ngai (51).
    Uses public GIS polygons from dvhcvn repo (level1 polygons) and assigns points to provinces by bbox.
    """
    if level != "province":
        raise HTTPException(status_code=400, detail="Only level=province supported in phase 1")

    cache_key = f"outbreak_areas|level=province|since_days={since_days}|d={disease}|minsev={min_severity}|since={since}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    provinces = [
        {"id": "46", "name": "Huế", "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/46.json"},
        {"id": "48", "name": "Đà Nẵng", "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/48.json"},
        {"id": "49", "name": "Quảng Nam", "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/49.json"},
        {"id": "51", "name": "Quảng Ngãi", "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/51.json"},
    ]

    # Load province polygons (cached by this endpoint cache)
    areas = []
    for p in provinces:
        r = requests.get(p["url"], timeout=12)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Boundary fetch failed for {p['id']}")
        data = r.json()
        areas.append({
            "area_id": p["id"],
            "name": p["name"],
            "type": data.get("type"),
            "bbox": data.get("bbox"),
            "coordinates": data.get("coordinates"),
        })

    # Fetch recent outbreak points (public) from Supabase table
    if since:
        since_iso = since
    else:
        since_iso = (datetime.now().astimezone().replace(microsecond=0) - timedelta(days=max(1, min(since_days, 30)))).isoformat()

    q = supabase.table("outbreak_cases").select("lat,lng,disease,severity,reported_at").gte("reported_at", since_iso)
    if disease:
        q = q.ilike("disease", disease)
    if min_severity is not None:
        q = q.gte("severity", min_severity)
    resp = q.execute()
    points = resp.data or []

    # Aggregate by province polygon (correct), fallback to bbox if needed
    out_items = []
    for a in areas:
        bbox = a.get("bbox")
        if not bbox:
            continue
        coords = a.get("coordinates") or []
        count7d = 0
        max_sev = 0
        disease_counts = {}
        for pt in points:
            try:
                lat = float(pt.get("lat"))
                lng = float(pt.get("lng"))
            except Exception:
                continue
            in_area = _point_in_multipolygon(lng, lat, coords)
            if not in_area:
                in_area = _point_in_bbox(lat, lng, bbox)
            if in_area:
                count7d += 1
                sev = int(pt.get("severity") or 0)
                max_sev = max(max_sev, sev)
                d = (pt.get("disease") or "").strip()
                if d:
                    disease_counts[d] = disease_counts.get(d, 0) + 1

        top_disease = None
        if disease_counts:
            top_disease = sorted(disease_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

        level_value = _compute_level(count7d, max_sev)
        out_items.append({
            "area_id": a["area_id"],
            "name": a["name"],
            "level": level_value,
            "count": count7d,
            "max_severity": max_sev if count7d else None,
            "top_disease": top_disease,
            "bbox": a.get("bbox"),
            "type": a.get("type"),
            "coordinates": a.get("coordinates"),
        })

    out = {"status": "success", "items": out_items, "since_days": since_days}
    cache_set(cache_key, out, ttl_seconds=60)
    return out
# --- 3. API CHÍNH ---
@app.post("/predict")
async def predict(request: Request, selected_plant: str = Form(...), file: UploadFile = File(...)):
    try:
        # 1. Đọc file ảnh
        contents = await file.read()

        # 2. Gửi ảnh sang Hugging Face để AI xử lý (Dùng 16GB RAM bên đó)
        print(f"🚀 Đang gửi yêu cầu sang Hugging Face cho cây: {selected_plant}")
        response = requests.post(
            HF_API_URL,
            files={"file": (file.filename, contents, file.content_type)},
            data={"selected_plant": selected_plant},
            timeout=120 # Đợi tối đa 2 phút nếu HF đang thức dậy
        )
        
        if response.status_code != 200:
            return {"status": "error", "message": "Hugging Face không phản hồi hoặc đang bận"}

        result = response.json()
        if result.get("status") == "error":
            return result

        # 3. Upload ảnh lên Supabase Storage như cũ
        file_name = f"{datetime.now().timestamp()}.jpg"
        supabase.storage.from_("plant-images").upload(
            file_name, contents, {"content-type": "image/jpeg"}
        )
        image_url = supabase.storage.from_("plant-images").get_public_url(file_name)

        # 4. Chuẩn hoá kết quả từ Hugging Face
        #    Lưu ý: trước đây API đang luôn trả plant = selected_plant (input),
        #    khiến app thấy "Tomato" hoài nếu UI gửi mặc định Tomato.
        disease_name = result.get("disease")

        def infer_plant_from_disease_label(label: str | None) -> str | None:
            if not isinstance(label, str):
                return None
            # common formats:
            # - "Apple___healthy"
            # - "Tomato___Late_blight"
            # - "Apple healthy" (fallback)
            if "___" in label:
                plant_part = label.split("___", 1)[0].strip()
                return plant_part or None
            if "_" in label:
                # if model returns "Apple_healthy" (single underscore)
                plant_part = label.split("_", 1)[0].strip()
                return plant_part or None
            if " " in label:
                plant_part = label.split(" ", 1)[0].strip()
                return plant_part or None
            return None

        predicted_plant = (
            result.get("plant")
            or infer_plant_from_disease_label(disease_name)
            or selected_plant
        )

        raw_confidence = result.get("confidence")
        confidence_value = None
        if isinstance(raw_confidence, (int, float)):
            confidence_value = float(raw_confidence)
        elif isinstance(raw_confidence, str):
            # chấp nhận "0.87", "87", "87%" ...
            s = raw_confidence.strip().replace("%", "")
            try:
                confidence_value = float(s)
            except Exception:
                confidence_value = None

        # chuẩn hoá về chuỗi phần trăm
        # - nếu model trả [0..1] thì đổi sang %
        # - nếu đã là [0..100] thì giữ nguyên
        confidence_percent_str = None
        confidence_percent_value = None
        if confidence_value is not None:
            if 0.0 <= confidence_value <= 1.0:
                confidence_percent_value = confidence_value * 100.0
                confidence_percent_str = f"{confidence_percent_value:.2f}%"
            else:
                confidence_percent_value = confidence_value
                confidence_percent_str = f"{confidence_percent_value:.2f}%"

        # 5. Trả kết quả cuối cùng về cho iOS App
        return {
            "status": "success",
            "plant": predicted_plant,
            "disease": disease_name,
            "confidence": confidence_percent_str,
            "image_url": image_url
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


class SaveHistoryRequest(BaseModel):
    plant: str | None = None
    disease: str | None = None
    confidence: float | None = None   # 0..100
    image_url: str | None = None


@app.post("/history/save")
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


# --- LLM advice (Gemini) with Supabase cache ---

class LLMAdviceDiagnosisRequest(BaseModel):
    plant: str | None = None
    disease: str
    confidence: float | None = None  # 0..100
    user_note: str | None = None
    weather_snapshot: dict | None = None


class LLMAdviceWeatherRequest(BaseModel):
    lat: float
    lng: float
    # optional: caller can provide snapshot to avoid extra weather call
    weather_snapshot: dict | None = None


_DIAGNOSIS_SYSTEM_PROMPT = """
Bạn là trợ lý nông nghiệp. Nhiệm vụ: tạo lời khuyên tiếng Việt dựa trên (cây trồng, bệnh dự đoán, độ tin cậy, ghi chú người dùng, thời tiết nếu có).
Yêu cầu đầu ra: CHỈ trả về JSON hợp lệ, theo schema:
{
  "summary_vi": "string",
  "symptoms": ["string", ...],
  "causes": ["string", ...],
  "treatments": ["string", ...],
  "prevention": ["string", ...],
  "when_to_seek_expert": "string"
}
Ràng buộc an toàn:
- Không đưa liều lượng/hoá chất cụ thể gây nguy hiểm; tránh chỉ định thuốc cấm.
- Ưu tiên IPM (quản lý dịch hại tổng hợp), vệ sinh vườn, thông thoáng, theo dõi.
- Nếu độ tin cậy thấp hoặc triệu chứng nặng/lan nhanh, khuyến nghị hỏi chuyên gia/khuyến nông địa phương.
"""

_WEATHER_SYSTEM_PROMPT = """
Bạn là trợ lý nông nghiệp. Nhiệm vụ: tạo lời khuyên tiếng Việt dựa trên thời tiết (nhiệt độ, độ ẩm, mưa, gió) để giảm rủi ro sâu bệnh.
Yêu cầu đầu ra: CHỈ trả về JSON hợp lệ theo schema giống:
{
  "summary_vi": "string",
  "symptoms": ["string", ...],   // có thể là dấu hiệu cần theo dõi ngoài đồng
  "causes": ["string", ...],     // yếu tố thời tiết làm tăng rủi ro
  "treatments": ["string", ...], // hành động khuyến nghị ngay (không nêu liều hoá chất)
  "prevention": ["string", ...],
  "when_to_seek_expert": "string"
}
Ràng buộc an toàn giống như trên.
"""


@app.post("/llm/advice/diagnosis")
async def llm_advice_diagnosis(req: LLMAdviceDiagnosisRequest):
    payload = {
        "plant": req.plant,
        "disease": req.disease,
        "confidence": req.confidence,
        "user_note": req.user_note,
        "weather_snapshot": req.weather_snapshot,
        "lang": "vi",
    }
    input_hash = _sha256(_canonical_json(payload))
    cached = _llm_cache_get("diagnosis", input_hash, "vi")
    if cached:
        return {
            "status": "success",
            "cached": True,
            "model": cached.get("model"),
            "advice": cached.get("content_json"),
            "summary_vi": cached.get("content_text"),
        }

    raw = _call_gemini_json(_DIAGNOSIS_SYSTEM_PROMPT, payload)
    advice = _validate_advice_json(raw)
    _llm_cache_upsert("diagnosis", input_hash, "vi", GEMINI_MODEL, advice, advice.get("summary_vi"))
    return {"status": "success", "cached": False, "model": GEMINI_MODEL, "advice": advice, "summary_vi": advice.get("summary_vi")}


@app.post("/llm/advice/weather")
async def llm_advice_weather(req: LLMAdviceWeatherRequest):
    snapshot = req.weather_snapshot
    if snapshot is None:
        # reuse logic from /weather endpoint (call OpenWeather directly)
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
            raise HTTPException(status_code=502, detail=f"OpenWeather error: {r.status_code} {r.text}")
        data = r.json()
        current = data.get("current") or {}
        snapshot = {
            "temp": current.get("temp"),
            "humidity": current.get("humidity"),
            "wind_speed": current.get("wind_speed"),
            "rain_1h": (current.get("rain") or {}).get("1h"),
            "weather": (current.get("weather") or [])[:1],
        }

    payload = {"lat": round(req.lat, 3), "lng": round(req.lng, 3), "snapshot": snapshot, "lang": "vi"}
    input_hash = _sha256(_canonical_json(payload))
    cached = _llm_cache_get("weather", input_hash, "vi")
    if cached and not _is_cache_expired("weather", cached.get("updated_at")):
        return {
            "status": "success",
            "cached": True,
            "model": cached.get("model"),
            "advice": cached.get("content_json"),
            "summary_vi": cached.get("content_text"),
        }

    raw = _call_gemini_json(_WEATHER_SYSTEM_PROMPT, payload)
    advice = _validate_advice_json(raw)
    _llm_cache_upsert("weather", input_hash, "vi", GEMINI_MODEL, advice, advice.get("summary_vi"))
    return {"status": "success", "cached": False, "model": GEMINI_MODEL, "advice": advice, "summary_vi": advice.get("summary_vi")}


# --- Weather (OpenWeather) ---
@app.get("/weather")
def weather(lat: float, lng: float):
    if not OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")

    lat_key = round(lat, 3)
    lng_key = round(lng, 3)
    cache_key = f"weather|{lat_key}|{lng_key}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # One Call API 3.0 (requires subscription / enabled key).
    url = "https://api.openweathermap.org/data/3.0/onecall"
    r = requests.get(
        url,
        params={
            "lat": lat,
            "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
            "exclude": "minutely"
        },
        timeout=12,
    )
    if r.status_code != 200:
        # include body for easier debugging (invalid key, not activated yet, plan restriction, etc.)
        try:
            body = r.text
        except Exception:
            body = ""
        raise HTTPException(status_code=502, detail=f"OpenWeather error: {r.status_code} {body}")
    data = r.json()

    current = data.get("current") or {}
    humidity = current.get("humidity")
    temp = current.get("temp")
    wind = current.get("wind_speed")
    rain_1h = (current.get("rain") or {}).get("1h")

    alerts = []
    if isinstance(humidity, (int, float)) and humidity >= 85:
        alerts.append({"code": "high_humidity", "title": "Độ ẩm cao", "message": "Độ ẩm cao dễ làm bệnh nấm phát triển. Hãy theo dõi ruộng/vườn kỹ hơn."})
    if isinstance(rain_1h, (int, float)) and rain_1h >= 2:
        alerts.append({"code": "rain", "title": "Có mưa", "message": "Mưa làm tăng ẩm và nguy cơ bệnh lan. Hạn chế tưới phun, đảm bảo thông thoáng."})
    if isinstance(temp, (int, float)) and temp >= 32:
        alerts.append({"code": "hot", "title": "Nắng nóng", "message": "Nhiệt độ cao, cây dễ stress. Tăng tưới hợp lý và che chắn khi cần."})
    if isinstance(wind, (int, float)) and wind >= 10:
        alerts.append({"code": "windy", "title": "Gió mạnh", "message": "Gió mạnh có thể làm phát tán mầm bệnh. Kiểm tra lá/cành sau gió lớn."})

    out = {
        "status": "success",
        "lat": lat,
        "lng": lng,
        "current": {
            "dt": current.get("dt"),
            "temp": temp,
            "humidity": humidity,
            "wind_speed": wind,
            "rain_1h": rain_1h,
            "weather": (current.get("weather") or [])[:1],
        },
        "hourly": (data.get("hourly") or [])[:24],
        "daily": (data.get("daily") or [])[:7],
        "alerts": alerts,
    }

    cache_set(cache_key, out, ttl_seconds=300)
    return out


@app.get("/weather/overview")
def weather_overview(lat: float, lng: float):
    """
    Returns human-friendly overview + advice from OpenWeather One Call 3.0 overview endpoint.
    """
    if not OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")

    lat_key = round(lat, 3)
    lng_key = round(lng, 3)
    cache_key = f"weather_overview|{lat_key}|{lng_key}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = "https://api.openweathermap.org/data/3.0/onecall/overview"
    r = requests.get(
        url,
        params={
            "lat": lat,
            "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
        },
        timeout=12,
    )
    if r.status_code != 200:
        try:
            body = r.text
        except Exception:
            body = ""
        raise HTTPException(status_code=502, detail=f"OpenWeather overview error: {r.status_code} {body}")

    data = r.json()
    overview_text = None
    if isinstance(data, dict):
        # OpenWeather may return different keys depending on version/locale.
        # Common candidates: 'weather_overview', 'overview'
        overview_text = data.get("weather_overview") or data.get("overview") or data.get("summary")

    def looks_vietnamese(s: str | None) -> bool:
        if not isinstance(s, str) or not s.strip():
            return False
        vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        t = s.lower()
        return any(ch in t for ch in vi_chars)

    def build_vi_advice() -> str:
        # Lightweight Vietnamese advice (no external translation dependency)
        parts: list[str] = []
        parts.append("Gợi ý nhanh cho canh tác (tham khảo):")
        parts.append("- Theo dõi lá/cành sau mưa hoặc độ ẩm cao; ưu tiên thông thoáng và vệ sinh vườn.")
        parts.append("- Tránh tưới phun vào chiều tối; tưới gốc để hạn chế nấm bệnh.")
        parts.append("- Nếu có gió mạnh, kiểm tra cây sau gió vì mầm bệnh dễ phát tán.")
        parts.append("- Quan sát triệu chứng sớm để xử lý kịp thời, tránh lây lan.")
        return "\n".join(parts)

    overview_text_vi = overview_text if looks_vietnamese(overview_text) else build_vi_advice()
    out = {
        "status": "success",
        "lat": lat,
        "lng": lng,
        "overview": data,
        "overview_text": overview_text,
        "overview_text_vi": overview_text_vi,
    }
    # cache longer; overview changes slower
    cache_set(cache_key, out, ttl_seconds=900)
    return out

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)