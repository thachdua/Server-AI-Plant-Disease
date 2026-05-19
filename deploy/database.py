from __future__ import annotations

import json
from datetime import datetime

import psycopg2

from deploy.config import DB_CONFIG


def save_to_db(plant_name, disease_name, confidence, image_url, created_by=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        try:
            query = """
            INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(
                query,
                (plant_name, disease_name, confidence, image_url, datetime.now(), created_by),
            )
        except Exception:
            conn.rollback()
            query = """
            INSERT INTO history (plant_name, disease_name, confidence, image_url, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            cur.execute(
                query, (plant_name, disease_name, confidence, image_url, datetime.now())
            )
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Đã lưu lịch sử vào Supabase")
    except Exception as e:
        print(f"❌ Lỗi lưu DB: {e}")


def llm_cache_get(kind: str, input_hash: str, lang: str = "vi"):
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


def llm_cache_upsert(
    kind: str, input_hash: str, lang: str, model: str, content_json, content_text: str | None
):
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
            (
                kind,
                input_hash,
                lang,
                model,
                json.dumps(content_json, ensure_ascii=False),
                content_text,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Lỗi ghi llm_advice_cache: {e}")
        return False
