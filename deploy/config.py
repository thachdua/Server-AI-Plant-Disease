import os
from typing import Optional

from supabase import Client, create_client

# Public URLs only — no secrets in source control.
HF_API_URL = os.environ.get(
    "HF_API_URL",
    "https://thachdua-plantdiseasedectect.hf.space/predict",
)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "") or os.environ.get("SUPABASE_ANON_KEY", "")

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
    "host": os.environ.get("DB_HOST", "aws-1-ap-southeast-1.pooler.supabase.com"),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", ""),
    "password": os.environ.get("DB_PASSWORD", ""),
    "port": int(os.environ.get("DB_PORT", "6543")),
    "sslmode": os.environ.get("DB_SSLMODE", "require"),
}


def _require(name: str, value: str) -> str:
    if not value or not str(value).strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def get_supabase() -> Client:
    url = _require("SUPABASE_URL", SUPABASE_URL)
    key = _require("SUPABASE_KEY (or SUPABASE_ANON_KEY)", SUPABASE_KEY)
    return create_client(url, key)


# Lazy client — fails at first use if env is not set (not at import).
_supabase: Optional[Client] = None


class _SupabaseProxy:
    def __getattr__(self, item):
        global _supabase
        if _supabase is None:
            _supabase = get_supabase()
        return getattr(_supabase, item)


supabase: Client = _SupabaseProxy()  # type: ignore[assignment]
