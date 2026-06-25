from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from deploy.cache import cache_get, cache_set
from deploy.config import supabase
from deploy.database import fetch_dicts
from deploy.geo import compute_level, compute_risk_level, compute_risk_score
from deploy.validation import (
    validate_limit,
    validate_severity,
    validate_since,
    validate_since_days,
)

router = APIRouter()


CASE_SELECT = (
    "id,lat,lng,plant,disease,confidence,image_url,history_id,created_by,"
    "location_label,province_id,province_name,ward_id,ward_name,severity,"
    "reported_at,note,source"
)

CASE_SQL_FIELDS = """
    id::text as id,
    lat,
    lng,
    plant,
    disease,
    confidence,
    image_url,
    history_id,
    created_by::text as created_by,
    location_label,
    province_id,
    province_name,
    ward_id,
    ward_name,
    severity,
    reported_at,
    note,
    source,
    coalesce(geom, st_setsrid(st_makepoint(lng, lat), 4326)) as case_geom
"""


def _since_iso(since: str | None, since_days: int) -> str:
    if since:
        return since
    return (
        datetime.now().astimezone().replace(microsecond=0) - timedelta(days=since_days)
    ).isoformat()


def _short_user_id(user_id: str | None) -> str:
    if not user_id:
        return ""
    return user_id[:8]


def _source_display(item: dict, profiles: dict[str, dict]) -> str:
    created_by = item.get("created_by")
    profile = profiles.get(str(created_by)) if created_by else None
    role = (profile or {}).get("role")
    name = ((profile or {}).get("display_name") or "").strip()
    if role == "expert":
        return f"Chuyên gia: {name}" if name else f"Chuyên gia {_short_user_id(created_by)}"
    if created_by:
        return f"Người dùng: {name}" if name else f"Người dùng {_short_user_id(created_by)}"
    return "Người dùng"


def _profiles_for_items(items: list[dict]) -> dict[str, dict]:
    ids = sorted({str(item.get("created_by")) for item in items if item.get("created_by")})
    if not ids:
        return {}
    try:
        resp = (
            supabase.table("profiles")
            .select("id,role,display_name")
            .in_("id", ids)
            .execute()
        )
    except Exception as exc:
        print(f"⚠️ Không thể tải profile nguồn vùng dịch: {exc}")
        return {}
    return {str(row.get("id")): row for row in (resp.data or []) if row.get("id")}


def _location_label(item: dict, *, fallback_province: str | None = None) -> str:
    label = (item.get("location_label") or "").strip()
    if label:
        return label
    ward = (item.get("ward_name") or "").strip()
    province = (item.get("province_name") or fallback_province or "").strip()
    if ward and province:
        return f"{ward}, {province}, Vietnam"
    if province:
        return f"{province}, Vietnam"
    return "Vietnam"


def _as_case(item: dict, profiles: dict[str, dict], *, fallback_province: str | None = None) -> dict:
    out = dict(item)
    out["id"] = str(out.get("id") or "")
    out["severity"] = int(out.get("severity") or 0)
    if out.get("created_by") is not None:
        out["created_by"] = str(out.get("created_by"))
    reported = out.get("reported_at")
    if isinstance(reported, (datetime, date)):
        out["reported_at"] = reported.isoformat()
    out["location_label"] = _location_label(out, fallback_province=fallback_province)
    out["source_display"] = _source_display(out, profiles)
    out.pop("case_geom", None)
    return out


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _iter_positions(value: Any):
    if not isinstance(value, list):
        return
    if len(value) >= 2 and all(isinstance(value[i], (int, float)) for i in (0, 1)):
        yield float(value[0]), float(value[1])
        return
    for child in value:
        yield from _iter_positions(child)


def _bbox_from_geojson(geojson: dict | None) -> list[float] | None:
    if not geojson:
        return None
    coords = list(_iter_positions(geojson.get("coordinates")))
    if not coords:
        return None
    lngs = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    return [min(lngs), min(lats), max(lngs), max(lats)]


def _centroid_from_geojson(geojson: dict | None) -> list[float] | None:
    if not geojson:
        return None
    coords = geojson.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2:
        return [float(coords[0]), float(coords[1])]
    return None


def _case_filters(
    *,
    since_iso: str,
    disease: str | None = None,
    min_severity: int | None = None,
    province_id: str | None = None,
    ward_id: str | None = None,
) -> tuple[str, list[Any]]:
    filters = ["reported_at >= %s"]
    params: list[Any] = [since_iso]
    if disease:
        filters.append("disease ilike %s")
        params.append(disease)
    if min_severity is not None:
        filters.append("severity >= %s")
        params.append(min_severity)
    if province_id:
        filters.append("province_id = %s")
        params.append(province_id)
    if ward_id:
        filters.append("ward_id = %s")
        params.append(ward_id)
    return " and ".join(filters), params


def _recent_cases_from_row(row: dict) -> list[dict]:
    raw = row.get("recent_cases") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _rows_to_areas(rows: list[dict]) -> list[dict]:
    recent_cases = [case for row in rows for case in _recent_cases_from_row(row)]
    profiles = _profiles_for_items(recent_cases)
    out: list[dict] = []

    for row in rows:
        geometry = _json_value(row.get("geojson")) or {}
        bbox_geometry = _json_value(row.get("bbox_geojson")) or geometry
        centroid_geometry = _json_value(row.get("centroid_geojson"))
        case_count = int(row.get("case_count") or 0)
        max_severity = row.get("max_severity")
        max_severity = int(max_severity) if max_severity is not None else None
        risk_level = compute_risk_level(case_count, max_severity)
        risk_score = compute_risk_score(case_count, max_severity)
        decorated_cases = [
            _as_case(case, profiles, fallback_province=row.get("name"))
            for case in _recent_cases_from_row(row)
        ]

        out.append(
            {
                "area_id": str(row.get("area_id") or ""),
                "name": row.get("name") or "",
                "full_name": row.get("full_name"),
                "parent_id": row.get("parent_id"),
                "parent_name": row.get("parent_name"),
                "province_id": row.get("province_id"),
                "level": compute_level(case_count, max_severity),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "count": case_count,
                "max_severity": max_severity if case_count else None,
                "top_disease": row.get("top_disease"),
                "bbox": _bbox_from_geojson(bbox_geometry),
                "type": geometry.get("type"),
                "coordinates": geometry.get("coordinates"),
                "centroid": _centroid_from_geojson(centroid_geometry),
                "ward_count": int(row.get("ward_count") or 0),
                "recent_cases": decorated_cases[:5],
            }
        )
    return out


def _fetch_area_rows(
    *,
    level: str,
    parent_id: str | None,
    since_iso: str,
    disease: str | None,
    min_severity: int | None,
) -> list[dict]:
    filters, params = _case_filters(
        since_iso=since_iso,
        disease=disease,
        min_severity=min_severity,
    )

    if level == "province":
        sql = f"""
            with case_scope as (
              select {CASE_SQL_FIELDS}
              from public.outbreak_cases
              where {filters}
            )
            select
              p.code::text as area_id,
              p.name::text as name,
              p.full_name::text as full_name,
              null::text as parent_id,
              null::text as parent_name,
              p.code::text as province_id,
              st_asgeojson(st_simplifypreservetopology(gp.geom, 0.006)) as geojson,
              st_asgeojson(gp.bbox) as bbox_geojson,
              st_asgeojson(st_pointonsurface(gp.geom)) as centroid_geojson,
              coalesce(stats.case_count, 0)::int as case_count,
              stats.max_severity::int as max_severity,
              top_disease.disease::text as top_disease,
              coalesce(recent.recent_cases, '[]'::json) as recent_cases,
              coalesce(ward_count.ward_count, 0)::int as ward_count
            from public.provinces p
            join public.gis_provinces gp on gp.province_code = p.code
            left join lateral (
              select count(*)::int as case_count, max(c.severity)::int as max_severity
              from case_scope c
              where c.case_geom && gp.bbox
                and st_covers(gp.geom, c.case_geom)
            ) stats on true
            left join lateral (
              select c.disease
              from case_scope c
              where c.case_geom && gp.bbox
                and st_covers(gp.geom, c.case_geom)
                and coalesce(btrim(c.disease), '') <> ''
              group by c.disease
              order by count(*) desc, c.disease
              limit 1
            ) top_disease on true
            left join lateral (
              select json_agg(row_to_json(r)) as recent_cases
              from (
                select
                  c.id,
                  c.lat,
                  c.lng,
                  c.plant,
                  c.disease,
                  c.confidence,
                  c.image_url,
                  c.history_id,
                  c.created_by,
                  c.location_label,
                  c.province_id,
                  c.province_name,
                  c.ward_id,
                  c.ward_name,
                  c.severity,
                  c.reported_at::text as reported_at,
                  c.note,
                  c.source
                from case_scope c
                where c.case_geom && gp.bbox
                  and st_covers(gp.geom, c.case_geom)
                order by c.reported_at desc
                limit 5
              ) r
            ) recent on true
            left join lateral (
              select count(*)::int as ward_count
              from public.wards w
              where w.province_code = p.code
            ) ward_count on true
            order by p.code
        """
        return fetch_dicts(sql, params)

    if not parent_id:
        raise HTTPException(status_code=400, detail="parent_id is required when level=ward")

    sql = f"""
        with case_scope as (
          select {CASE_SQL_FIELDS}
          from public.outbreak_cases
          where {filters}
        )
        select
          w.code::text as area_id,
          w.name::text as name,
          coalesce(w.full_name, w.name)::text as full_name,
          p.code::text as parent_id,
          p.name::text as parent_name,
          p.code::text as province_id,
          st_asgeojson(st_simplifypreservetopology(gw.geom, 0.0015)) as geojson,
          st_asgeojson(gw.bbox) as bbox_geojson,
          st_asgeojson(st_pointonsurface(gw.geom)) as centroid_geojson,
          coalesce(stats.case_count, 0)::int as case_count,
          stats.max_severity::int as max_severity,
          top_disease.disease::text as top_disease,
          coalesce(recent.recent_cases, '[]'::json) as recent_cases,
          0::int as ward_count
        from public.wards w
        join public.provinces p on p.code = w.province_code
        join public.gis_wards gw on gw.ward_code = w.code
        left join lateral (
          select count(*)::int as case_count, max(c.severity)::int as max_severity
          from case_scope c
          where c.case_geom && gw.bbox
            and st_covers(gw.geom, c.case_geom)
        ) stats on true
        left join lateral (
          select c.disease
          from case_scope c
          where c.case_geom && gw.bbox
            and st_covers(gw.geom, c.case_geom)
            and coalesce(btrim(c.disease), '') <> ''
          group by c.disease
          order by count(*) desc, c.disease
          limit 1
        ) top_disease on true
        left join lateral (
          select json_agg(row_to_json(r)) as recent_cases
          from (
            select
              c.id,
              c.lat,
              c.lng,
              c.plant,
              c.disease,
              c.confidence,
              c.image_url,
              c.history_id,
              c.created_by,
              c.location_label,
              c.province_id,
              c.province_name,
              c.ward_id,
              c.ward_name,
              c.severity,
              c.reported_at::text as reported_at,
              c.note,
              c.source
            from case_scope c
            where c.case_geom && gw.bbox
              and st_covers(gw.geom, c.case_geom)
            order by c.reported_at desc
            limit 5
          ) r
        ) recent on true
        where w.province_code = %s
        order by w.code
    """
    return fetch_dicts(sql, [*params, parent_id])


@router.get("/outbreaks")
def outbreaks(
    disease: Optional[str] = None,
    severity: Optional[int] = None,
    min_severity: Optional[int] = None,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 500,
):
    validate_severity(severity)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)
    limit = validate_limit(limit)

    cache_key = (
        f"outbreaks|d={disease}|sev={severity}|minsev={min_severity}|"
        f"province={province_id}|ward={ward_id}|since={since}|l={limit}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "items": cached}

    q = supabase.table("outbreak_cases").select(CASE_SELECT)
    if disease:
        q = q.ilike("disease", disease)
    if severity is not None:
        q = q.eq("severity", severity)
    if min_severity is not None:
        q = q.gte("severity", min_severity)
    if province_id:
        q = q.eq("province_id", province_id)
    if ward_id:
        q = q.eq("ward_id", ward_id)
    if since:
        q = q.gte("reported_at", since)
    q = q.order("reported_at", desc=True).limit(limit)

    resp = q.execute()
    raw_items = resp.data or []
    profiles = _profiles_for_items(raw_items)
    items = [_as_case(item, profiles) for item in raw_items]
    cache_set(cache_key, items, ttl_seconds=20)
    return {"status": "success", "items": items}


@router.get("/outbreaks/areas")
def outbreak_areas(
    level: str = "province",
    parent_id: Optional[str] = None,
    since_days: int = 7,
    disease: Optional[str] = None,
    min_severity: Optional[int] = None,
    since: Optional[str] = None,
):
    if level not in {"province", "ward"}:
        raise HTTPException(status_code=400, detail="level must be province or ward")
    if level == "ward" and not parent_id:
        raise HTTPException(status_code=400, detail="parent_id is required when level=ward")

    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)
    since_iso = _since_iso(since, since_days)

    cache_key = (
        f"outbreak_areas|level={level}|parent={parent_id}|since_days={since_days}|"
        f"d={disease}|minsev={min_severity}|since={since_iso}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    rows = _fetch_area_rows(
        level=level,
        parent_id=parent_id,
        since_iso=since_iso,
        disease=disease,
        min_severity=min_severity,
    )
    out = {
        "status": "success",
        "items": _rows_to_areas(rows),
        "level": level,
        "parent_id": parent_id,
        "since_days": since_days,
    }
    cache_set(cache_key, out, ttl_seconds=300)
    return out


def _summary_cases(
    *,
    since_iso: str,
    disease: str | None,
    min_severity: int | None,
    province_id: str | None,
    ward_id: str | None,
) -> list[dict]:
    filters, params = _case_filters(
        since_iso=since_iso,
        disease=disease,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
    )
    sql = f"""
        select
          id::text as id,
          plant,
          disease,
          severity,
          reported_at,
          province_id,
          province_name,
          ward_id,
          ward_name
        from public.outbreak_cases
        where {filters}
        order by reported_at desc
    """
    return fetch_dicts(sql, params)


def _top_counts(cases: list[dict], field: str, *, limit: int = 5) -> list[dict]:
    counts = Counter(
        str(item.get(field)).strip()
        for item in cases
        if str(item.get(field) or "").strip()
    )
    return [
        {"name": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    ]


def _trend(cases: list[dict]) -> list[dict]:
    counts: Counter[str] = Counter()
    for item in cases:
        raw = item.get("reported_at")
        if isinstance(raw, datetime):
            key = raw.date().isoformat()
        else:
            key = str(raw or "")[:10]
        if key:
            counts[key] += 1
    return [{"date": key, "count": counts[key]} for key in sorted(counts)]


def _risky_areas(cases: list[dict], *, limit: int = 8) -> list[dict]:
    grouped: dict[str, dict] = defaultdict(lambda: {"count": 0, "max_severity": 0})
    for item in cases:
        area_id = item.get("ward_id") or item.get("province_id")
        if not area_id:
            continue
        group = grouped[str(area_id)]
        group["area_id"] = str(area_id)
        group["name"] = item.get("ward_name") or item.get("province_name") or str(area_id)
        group["parent_id"] = item.get("province_id")
        group["parent_name"] = item.get("province_name")
        group["count"] += 1
        group["max_severity"] = max(group["max_severity"], int(item.get("severity") or 0))

    out = []
    for group in grouped.values():
        max_severity = group["max_severity"] or None
        risk_level = compute_risk_level(group["count"], max_severity)
        out.append(
            {
                "area_id": group["area_id"],
                "name": group["name"],
                "parent_id": group.get("parent_id"),
                "parent_name": group.get("parent_name"),
                "count": group["count"],
                "max_severity": max_severity,
                "risk_level": risk_level,
                "risk_score": compute_risk_score(group["count"], max_severity),
            }
        )
    return sorted(out, key=lambda item: (-item["risk_level"], -item["count"], item["name"]))[:limit]


@router.get("/outbreaks/summary")
def outbreak_summary(
    since_days: int = 7,
    disease: Optional[str] = None,
    min_severity: Optional[int] = None,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    since: Optional[str] = None,
):
    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)
    since_iso = _since_iso(since, since_days)

    cache_key = (
        f"outbreak_summary|since_days={since_days}|d={disease}|minsev={min_severity}|"
        f"province={province_id}|ward={ward_id}|since={since_iso}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    cases = _summary_cases(
        since_iso=since_iso,
        disease=disease,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
    )
    risky_areas = _risky_areas(cases)
    out = {
        "status": "success",
        "since_days": since_days,
        "totals": {
            "case_count": len(cases),
            "province_count": len({item.get("province_id") for item in cases if item.get("province_id")}),
            "ward_count": len({item.get("ward_id") for item in cases if item.get("ward_id")}),
            "high_risk_area_count": len([item for item in risky_areas if item["risk_level"] >= 3]),
        },
        "top_diseases": _top_counts(cases, "disease"),
        "top_plants": _top_counts(cases, "plant"),
        "trend": _trend(cases),
        "risky_areas": risky_areas,
    }
    cache_set(cache_key, out, ttl_seconds=60)
    return out
