from __future__ import annotations

import requests
from fastapi import APIRouter, HTTPException

from deploy.cache import cache_get, cache_set
from deploy.config import OPENWEATHER_API_KEY

router = APIRouter()


@router.get("/weather")
def weather(lat: float, lng: float):
    if not OPENWEATHER_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENWEATHER_API_KEY")

    lat_key = round(lat, 3)
    lng_key = round(lng, 3)
    cache_key = f"weather|{lat_key}|{lng_key}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    url = "https://api.openweathermap.org/data/3.0/onecall"
    r = requests.get(
        url,
        params={
            "lat": lat,
            "lon": lng,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "lang": "vi",
            "exclude": "minutely",
        },
        timeout=12,
    )
    if r.status_code != 200:
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
        alerts.append(
            {
                "code": "high_humidity",
                "title": "Độ ẩm cao",
                "message": "Độ ẩm cao dễ làm bệnh nấm phát triển. Hãy theo dõi ruộng/vườn kỹ hơn.",
            }
        )
    if isinstance(rain_1h, (int, float)) and rain_1h >= 2:
        alerts.append(
            {
                "code": "rain",
                "title": "Có mưa",
                "message": "Mưa làm tăng ẩm và nguy cơ bệnh lan. Hạn chế tưới phun, đảm bảo thông thoáng.",
            }
        )
    if isinstance(temp, (int, float)) and temp >= 32:
        alerts.append(
            {
                "code": "hot",
                "title": "Nắng nóng",
                "message": "Nhiệt độ cao, cây dễ stress. Tăng tưới hợp lý và che chắn khi cần.",
            }
        )
    if isinstance(wind, (int, float)) and wind >= 10:
        alerts.append(
            {
                "code": "windy",
                "title": "Gió mạnh",
                "message": "Gió mạnh có thể làm phát tán mầm bệnh. Kiểm tra lá/cành sau gió lớn.",
            }
        )

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


@router.get("/weather/overview")
def weather_overview(lat: float, lng: float):
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
        raise HTTPException(
            status_code=502, detail=f"OpenWeather overview error: {r.status_code} {body}"
        )

    data = r.json()
    overview_text = None
    if isinstance(data, dict):
        overview_text = (
            data.get("weather_overview") or data.get("overview") or data.get("summary")
        )

    def looks_vietnamese(s: str | None) -> bool:
        if not isinstance(s, str) or not s.strip():
            return False
        vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        t = s.lower()
        return any(ch in t for ch in vi_chars)

    def build_vi_advice() -> str:
        parts: list[str] = []
        parts.append("Gợi ý nhanh cho canh tác (tham khảo):")
        parts.append(
            "- Theo dõi lá/cành sau mưa hoặc độ ẩm cao; ưu tiên thông thoáng và vệ sinh vườn."
        )
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
    cache_set(cache_key, out, ttl_seconds=900)
    return out
