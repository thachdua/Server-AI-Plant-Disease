import os
import io
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import time
from supabase import create_client, Client
import psycopg2
from datetime import datetime

app = FastAPI()

# --- 1. CONFIG ---
# Link API Hugging Face mà bạn vừa tạo
HF_API_URL = "https://thachdua-plantdiseasedectect.hf.space/predict"

SUPABASE_URL = "https://wxmmfmvyefruyknymvdm.supabase.co"
SUPABASE_KEY = "sb_publishable_gW6BTa8IWvVjO6Rbe-OGxQ_WgD6knWz"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

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

    # NOTE: One Call 3.0 often requires a paid subscription.
    # For free-tier keys, use the legacy 2.5 endpoint.
    url = "https://api.openweathermap.org/data/2.5/onecall"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)