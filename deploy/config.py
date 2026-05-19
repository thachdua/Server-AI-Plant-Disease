import os

from supabase import Client, create_client

HF_API_URL = "https://thachdua-plantdiseasedectect.hf.space/predict"

SUPABASE_URL = "https://wxmmfmvyefruyknymvdm.supabase.co"
SUPABASE_KEY = "sb_publishable_gW6BTa8IWvVjO6Rbe-OGxQ_WgD6knWz"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_KEYS = [
    k.strip()
    for k in os.environ.get("GEMINI_API_KEYS", "").split(",")
    if k.strip()
]
if not GEMINI_API_KEYS and GEMINI_API_KEY:
    GEMINI_API_KEYS = [GEMINI_API_KEY]

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-2.5-flash"
    ).split(",")
    if m.strip()
]
GEMINI_API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1beta")

DB_CONFIG = {
    "host": "aws-1-ap-southeast-1.pooler.supabase.com",
    "database": "postgres",
    "user": "postgres.wxmmfmvyefruyknymvdm",
    "password": os.environ.get("DB_PASSWORD", "Nguyenyeuloc@123"),
    "port": 6543,
    "sslmode": "require",
}
