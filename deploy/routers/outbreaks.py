from datetime import datetime, timedelta
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException

from deploy.cache import cache_get, cache_set
from deploy.config import supabase
from deploy.geo import compute_level, point_in_bbox, point_in_multipolygon

router = APIRouter()


@router.get("/outbreaks")
def outbreaks(
    disease: Optional[str] = None,
    severity: Optional[int] = None,
    since: Optional[str] = None,
    limit: int = 500,
):
    cache_key = f"outbreaks|d={disease}|sev={severity}|since={since}|l={limit}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "items": cached}

    q = supabase.table("outbreak_cases").select(
        "id,lat,lng,disease,severity,reported_at,note,source"
    )
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
            - timedelta(days=max(1, min(since_days, 30)))
        ).isoformat()

    q = supabase.table("outbreak_cases").select("lat,lng,disease,severity,reported_at").gte(
        "reported_at", since_iso
    )
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

        top_disease = None
        if disease_counts:
            top_disease = sorted(disease_counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

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
            }
        )

    out = {"status": "success", "items": out_items, "since_days": since_days}
    cache_set(cache_key, out, ttl_seconds=60)
    return out
