import os
from typing import Optional
from urllib.parse import urlparse

from supabase import Client, create_client


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Public URLs only — no secrets in source control.
HF_API_URL = os.environ.get(
    "HF_API_URL",
    "https://thachdua-plantdiseasedectect.hf.space/predict",
)

PREDICT_REQUIRE_AUTH = _env_bool("PREDICT_REQUIRE_AUTH", default=False)
PREDICT_MAX_UPLOAD_BYTES = _env_int("PREDICT_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
PREDICT_RATE_LIMIT_PER_MINUTE = _env_int("PREDICT_RATE_LIMIT_PER_MINUTE", 20)
PREDICT_LOW_CONFIDENCE_THRESHOLD = _env_int("PREDICT_LOW_CONFIDENCE_THRESHOLD", 70)
PREDICT_UNRECOGNIZED_THRESHOLD = _env_int("PREDICT_UNRECOGNIZED_THRESHOLD", 60)
LLM_RATE_LIMIT_PER_MINUTE = _env_int("LLM_RATE_LIMIT_PER_MINUTE", 10)
LLM_CHAT_MAX_CHARS = _env_int("LLM_CHAT_MAX_CHARS", 4000)
WEATHER_RATE_LIMIT_PER_MINUTE = _env_int("WEATHER_RATE_LIMIT_PER_MINUTE", 60)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY_SOURCE = (
    "SUPABASE_SERVICE_ROLE_KEY"
    if os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    else "SUPABASE_KEY"
    if os.environ.get("SUPABASE_KEY", "").strip()
    else "SUPABASE_ANON_KEY"
    if os.environ.get("SUPABASE_ANON_KEY", "").strip()
    else ""
)
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    or os.environ.get("SUPABASE_KEY", "")
    or os.environ.get("SUPABASE_ANON_KEY", "")
)

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

REQUIRED_ENV_VARS = (
    ("SUPABASE_URL", "SUPABASE_URL"),
    ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_BACKEND_KEY"),
    ("DB_USER", "DB_USER"),
    ("DB_PASSWORD", "DB_PASSWORD"),
)


def _is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _supabase_key_warnings() -> list[str]:
    if not SUPABASE_KEY:
        return []
    lowered = SUPABASE_KEY.lower()
    if SUPABASE_KEY_SOURCE == "SUPABASE_ANON_KEY":
        return ["backend_is_using_supabase_anon_key"]
    if lowered.startswith("sb_publishable_"):
        return ["backend_is_using_supabase_publishable_key"]
    if SUPABASE_KEY_SOURCE == "SUPABASE_KEY":
        return ["prefer_supabase_service_role_key_name"]
    return []


def config_status() -> dict:
    """Return non-secret configuration status for health checks."""
    env_values = {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_BACKEND_KEY": SUPABASE_KEY,
        "DB_USER": DB_CONFIG.get("user", ""),
        "DB_PASSWORD": DB_CONFIG.get("password", ""),
        "HF_API_URL": HF_API_URL,
        "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
        "GEMINI_API_KEYS": ",".join(GEMINI_API_KEYS),
    }
    missing_required = [
        display
        for display, env_key in REQUIRED_ENV_VARS
        if not str(env_values.get(env_key, "")).strip()
    ]
    invalid = []
    if HF_API_URL and not _is_valid_http_url(HF_API_URL):
        invalid.append("HF_API_URL")
    if SUPABASE_URL and not _is_valid_http_url(SUPABASE_URL):
        invalid.append("SUPABASE_URL")
    warnings = _supabase_key_warnings()

    optional = {
        "openweather_configured": bool(OPENWEATHER_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEYS),
    }
    predict = {
        "require_auth": PREDICT_REQUIRE_AUTH,
        "max_upload_bytes": PREDICT_MAX_UPLOAD_BYTES,
        "rate_limit_per_minute": PREDICT_RATE_LIMIT_PER_MINUTE,
        "low_confidence_threshold": PREDICT_LOW_CONFIDENCE_THRESHOLD,
        "unrecognized_threshold": PREDICT_UNRECOGNIZED_THRESHOLD,
    }
    limits = {
        "llm_rate_limit_per_minute": LLM_RATE_LIMIT_PER_MINUTE,
        "llm_chat_max_chars": LLM_CHAT_MAX_CHARS,
        "weather_rate_limit_per_minute": WEATHER_RATE_LIMIT_PER_MINUTE,
    }
    database = {
        "host_configured": bool(DB_CONFIG.get("host")),
        "database_configured": bool(DB_CONFIG.get("database")),
        "port": DB_CONFIG.get("port"),
        "sslmode": DB_CONFIG.get("sslmode"),
    }
    return {
        "ok": not missing_required and not invalid,
        "missing_required": missing_required,
        "invalid": invalid,
        "warnings": warnings,
        "supabase_key_source": SUPABASE_KEY_SOURCE or None,
        "optional": optional,
        "predict": predict,
        "limits": limits,
        "database": database,
    }


def _require(name: str, value: str) -> str:
    if not value or not str(value).strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def get_supabase() -> Client:
    url = _require("SUPABASE_URL", SUPABASE_URL)
    key = _require(
        "SUPABASE_SERVICE_ROLE_KEY (or legacy SUPABASE_KEY)",
        SUPABASE_KEY,
    )
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
