from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
from psycopg2 import errors

from deploy.config import DB_CONFIG


def _db_connect():
    if not DB_CONFIG.get("user") or not DB_CONFIG.get("password"):
        raise RuntimeError("Missing DB_USER or DB_PASSWORD environment variables")
    return psycopg2.connect(**DB_CONFIG)


@contextmanager
def _db_cursor(commit: bool = False):
    conn = _db_connect()
    try:
        with conn.cursor() as cur:
            yield conn, cur
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise
    finally:
        conn.close()


def _now_utc():
    return datetime.now(timezone.utc)


def save_to_db(plant_name, disease_name, confidence, image_url, created_by=None):
    try:
        with _db_cursor(commit=True) as (_, cur):
            query = """
            INSERT INTO history
                (plant_name, disease_name, confidence, image_url, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(
                query,
                (
                    plant_name,
                    disease_name,
                    confidence,
                    image_url,
                    _now_utc(),
                    created_by,
                ),
            )
            row = cur.fetchone()
            history_id = str(row[0]) if row else None
    except errors.UndefinedColumn:
        with _db_cursor(commit=True) as (_, cur):
            query = """
            INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            cur.execute(
                query,
                (plant_name, disease_name, confidence, image_url, _now_utc()),
            )
            row = cur.fetchone()
            history_id = str(row[0]) if row else None
    print("✅ Đã lưu lịch sử vào Supabase")
    return history_id


def save_outbreak_case(
    *,
    lat: float,
    lng: float,
    plant: str | None,
    disease: str,
    confidence: float | None,
    image_url: str | None,
    history_id: str | None,
    created_by: str | None,
    location_label: str | None = None,
    province_name: str | None = None,
) -> str | None:
    with _db_cursor(commit=True) as (_, cur):
        cur.execute(
            """
            INSERT INTO outbreak_cases
                (
                    lat, lng, plant, disease, confidence, image_url, history_id,
                    created_by, location_label, province_name, severity, reported_at,
                    note, source, review_status
                )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s, 'history_save', 'auto_accepted')
            RETURNING id
            """,
            (
                lat,
                lng,
                plant,
                disease,
                confidence,
                image_url,
                history_id,
                created_by,
                location_label,
                province_name,
                _now_utc(),
                "Tự động tạo từ lịch sử chẩn đoán. Cấp vùng dịch được tính theo số ca ghi nhận.",
            ),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None


def llm_cache_get(kind: str, input_hash: str, lang: str = "vi"):
    try:
        with _db_cursor() as (_, cur):
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


def llm_cache_upsert(
    kind: str, input_hash: str, lang: str, model: str, content_json, content_text: str | None
):
    try:
        with _db_cursor(commit=True) as (_, cur):
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
                (
                    kind,
                    input_hash,
                    lang,
                    model,
                    json.dumps(content_json, ensure_ascii=False),
                    content_text,
                ),
            )
        return True
    except Exception as e:
        print(f"❌ Lỗi ghi llm_advice_cache: {e}")
        return False
