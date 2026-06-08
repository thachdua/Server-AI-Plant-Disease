from datetime import datetime, timedelta
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException

from deploy.cache import cache_get, cache_set
from deploy.config import supabase
from deploy.geo import compute_level, point_in_bbox, point_in_multipolygon
from deploy.validation import (
    validate_limit,
    validate_severity,
    validate_since,
    validate_since_days,
)

router = APIRouter()


@router.get("/outbreaks")
def outbreaks(
    disease: Optional[str] = None,
    severity: Optional[int] = None,
    since: Optional[str] = None,
    limit: int = 500,
):
    validate_severity(severity)
    validate_since(since)
    limit = validate_limit(limit)

    cache_key = f"outbreaks|d={disease}|sev={severity}|since={since}|l={limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "items": cached}

    q = supabase.table("outbreak_cases").select(
        "id,lat,lng,plant,disease,confidence,image_url,history_id,severity,reported_at,note,source"
    )
    if disease:
        q = q.ilike("disease", disease)
    if severity is not None:
        q = q.eq("severity", severity)
    if since:
        q = q.gte("reported_at", since)
    q = q.order("reported_at", desc=True).limit(limit)

    resp = q.execute()
    items = resp.data or []
    cache_set(cache_key, items, ttl_seconds=20)
    return {"status": "success", "items": items}


@router.get("/outbreaks/areas")
def outbreak_areas(
    level: str = "province",
    since_days: int = 7,
    disease: Optional[str] = None,
    min_severity: Optional[int] = None,
    since: Optional[str] = None,
):
    if level != "province":
        raise HTTPException(status_code=400, detail="Only level=province supported in phase 1")
    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)

    cache_key = (
        f"outbreak_areas|level=province|since_days={since_days}|d={disease}"
        f"|minsev={min_severity}|since={since}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    provinces = [
        {
            "id": "46",
            "name": "Huế",
            "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/46.json",
        },
        {
            "id": "48",
            "name": "Đà Nẵng",
            "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/48.json",
        },
        {
            "id": "49",
            "name": "Quảng Nam",
            "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/49.json",
        },
        {
            "id": "51",
            "name": "Quảng Ngãi",
            "url": "https://raw.githubusercontent.com/daohoangson/dvhcvn/master/data/gis/51.json",
        },
    ]

    areas = []
    for p in provinces:
        r = requests.get(p["url"], timeout=12)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Boundary fetch failed for {p['id']}")
        data = r.json()
        areas.append(
            {
                "area_id": p["id"],
                "name": p["name"],
                "type": data.get("type"),
                "bbox": data.get("bbox"),
                "coordinates": data.get("coordinates"),
            }
        )

    if since:
        since_iso = since
    else:
        since_iso = (
            datetime.now().astimezone().replace(microsecond=0)
            - timedelta(days=since_days)
        ).isoformat()

    q = supabase.table("outbreak_cases").select(
        "id,lat,lng,plant,disease,confidence,image_url,history_id,severity,reported_at,note,source"
    ).gte("reported_at", since_iso)
    if disease:
        q = q.ilike("disease", disease)
    if min_severity is not None:
        q = q.gte("severity", min_severity)
    resp = q.execute()
    points = resp.data or []

    out_items = []
    for a in areas:
        bbox = a.get("bbox")
        if not bbox:
            continue
        coords = a.get("coordinates") or []
        count7d = 0
        max_sev = 0
        disease_counts = {}
        recent_cases = []
        for pt in points:
            try:
                lat = float(pt.get("lat"))
                lng = float(pt.get("lng"))
            except Exception:
                continue
            in_area = point_in_multipolygon(lng, lat, coords)
            if not in_area:
                in_area = point_in_bbox(lat, lng, bbox)
            if in_area:
                count7d += 1
                sev = int(pt.get("severity") or 0)
                max_sev = max(max_sev, sev)
                d = (pt.get("disease") or "").strip()
                if d:
                    disease_counts[d] = disease_counts.get(d, 0) + 1
                recent_cases.append(
                    {
                        "id": str(pt.get("id") or ""),
                        "lat": lat,
                        "lng": lng,
                        "plant": pt.get("plant"),
                        "disease": pt.get("disease") or "",
                        "confidence": pt.get("confidence"),
                        "image_url": pt.get("image_url"),
                        "history_id": pt.get("history_id"),
                        "severity": sev,
                        "reported_at": pt.get("reported_at"),
                        "note": pt.get("note"),
                        "source": pt.get("source"),
                    }
                )

        top_disease = None
        if disease_counts:
            top_disease = sorted(disease_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

        recent_cases.sort(key=lambda item: item.get("reported_at") or "", reverse=True)
        level_value = compute_level(count7d, max_sev)
        out_items.append(
            {
                "area_id": a["area_id"],
                "name": a["name"],
                "level": level_value,
                "count": count7d,
                "max_severity": max_sev if count7d else None,
                "top_disease": top_disease,
                "bbox": a.get("bbox"),
                "type": a.get("type"),
                "coordinates": a.get("coordinates"),
                "recent_cases": recent_cases[:5],
            }
        )

    out = {"status": "success", "items": out_items, "since_days": since_days}
    cache_set(cache_key, out, ttl_seconds=60)
    return out
