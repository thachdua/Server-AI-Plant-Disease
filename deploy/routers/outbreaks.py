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
    validate_coordinates,
    validate_limit,
    validate_severity,
    validate_since,
    validate_since_days,
)

router = APIRouter()


CASE_SELECT = (
    "id,lat,lng,plant,disease,confidence,image_url,history_id,created_by,"
    "location_label,province_id,province_name,ward_id,ward_name,severity,"
    "reported_at,note,source,review_status"
)


def _case_sql_fields(*, near_lat: float | None = None, near_lng: float | None = None) -> tuple[str, list[Any]]:
    distance_sql = "null::double precision"
    params: list[Any] = []
    if near_lat is not None and near_lng is not None:
        distance_sql = """
            st_distance(
              coalesce(geom, st_setsrid(st_makepoint(lng, lat), 4326))::geography,
              st_setsrid(st_makepoint(%s, %s), 4326)::geography
            ) / 1000.0
        """
        params.extend([near_lng, near_lat])

    return f"""
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
        review_status,
        {distance_sql} as distance_km,
        coalesce(geom, st_setsrid(st_makepoint(lng, lat), 4326)) as case_geom
    """, params


def _since_iso(since: str | None, since_days: int) -> str:
    if since:
        return since
    return (
        datetime.now().astimezone().replace(microsecond=0) - timedelta(days=since_days)
    ).isoformat()


def _like_pattern(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    if "%" in text or "_" in text:
        return text
    return f"%{text}%"


def _validate_radius_km(value: float | None, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if value <= 0 or value > 500:
        raise HTTPException(status_code=400, detail="radius_km must be between 0 and 500")
    return value


def _validate_risk_min(value: int | None) -> None:
    if value is None:
        return
    if value < 0 or value > 4:
        raise HTTPException(status_code=400, detail="risk_min must be between 0 and 4")


def _validate_nearby(lat: float | None, lng: float | None) -> tuple[float | None, float | None]:
    if lat is None and lng is None:
        return None, None
    if lat is None or lng is None:
        raise HTTPException(status_code=400, detail="near_lat and near_lng must be provided together")
    validate_coordinates(lat, lng)
    return lat, lng


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
    if out.get("distance_km") is not None:
        out["distance_km"] = round(float(out.get("distance_km") or 0), 2)
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
    plant: str | None = None,
    min_severity: int | None = None,
    province_id: str | None = None,
    ward_id: str | None = None,
    source: str | None = None,
    review_status: str | None = None,
    confidence_min: float | None = None,
    near_lat: float | None = None,
    near_lng: float | None = None,
    radius_km: float | None = None,
) -> tuple[str, list[Any]]:
    filters = ["reported_at >= %s"]
    params: list[Any] = [since_iso]
    disease = _like_pattern(disease)
    plant = _like_pattern(plant)
    if disease:
        filters.append("disease ilike %s")
        params.append(disease)
    if plant:
        filters.append("plant ilike %s")
        params.append(plant)
    if min_severity is not None:
        filters.append("severity >= %s")
        params.append(min_severity)
    if province_id:
        filters.append("province_id = %s")
        params.append(province_id)
    if ward_id:
        filters.append("ward_id = %s")
        params.append(ward_id)
    if source:
        filters.append("source = %s")
        params.append(source)
    if review_status:
        filters.append("review_status = %s")
        params.append(review_status)
    if confidence_min is not None:
        filters.append("confidence >= %s")
        params.append(confidence_min)
    if near_lat is not None and near_lng is not None and radius_km is not None:
        filters.append(
            """
            st_dwithin(
              coalesce(geom, st_setsrid(st_makepoint(lng, lat), 4326))::geography,
              st_setsrid(st_makepoint(%s, %s), 4326)::geography,
              %s
            )
            """
        )
        params.extend([near_lng, near_lat, radius_km * 1000.0])
    return " and ".join(filters), params


def _recent_cases_from_row(row: dict) -> list[dict]:
    raw = row.get("recent_cases") or []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _rows_to_areas(rows: list[dict], *, include_geometry: bool = True, risk_min: int | None = None) -> list[dict]:
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
        if risk_min is not None and risk_level < risk_min:
            continue
        decorated_cases = [
            _as_case(case, profiles, fallback_province=row.get("name"))
            for case in _recent_cases_from_row(row)
        ]
        distance_km = row.get("distance_km")

        out.append(
            {
                "area_id": str(row.get("area_id") or ""),
                "name": row.get("name") or "",
                "full_name": row.get("full_name"),
                "parent_id": row.get("parent_id"),
                "parent_name": row.get("parent_name"),
                "province_id": row.get("province_id"),
                "unit_type": row.get("unit_type"),
                "area_km2": float(row.get("area_km2")) if row.get("area_km2") is not None else None,
                "selected": bool(row.get("selected") or False),
                "distance_km": round(float(distance_km), 2) if distance_km is not None else None,
                "level": compute_level(case_count, max_severity),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "count": case_count,
                "max_severity": max_severity if case_count else None,
                "top_disease": row.get("top_disease"),
                "bbox": _bbox_from_geojson(bbox_geometry),
                "type": geometry.get("type") if include_geometry else None,
                "coordinates": geometry.get("coordinates") if include_geometry else None,
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
    plant: str | None = None,
    min_severity: int | None,
    province_id: str | None = None,
    ward_id: str | None = None,
    source: str | None = None,
    review_status: str | None = None,
    confidence_min: float | None = None,
    near_lat: float | None = None,
    near_lng: float | None = None,
    radius_km: float | None = None,
    include_geometry: bool = True,
    selected_only: bool = False,
) -> list[dict]:
    case_fields, case_field_params = _case_sql_fields(near_lat=near_lat, near_lng=near_lng)
    scope_province_id = province_id
    scope_ward_id = ward_id
    if level == "ward" and parent_id and not scope_province_id:
        scope_province_id = parent_id

    filters, params = _case_filters(
        since_iso=since_iso,
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        province_id=scope_province_id,
        ward_id=scope_ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
    )
    geom_sql_province = "st_asgeojson(st_simplifypreservetopology(gp.geom, 0.006))" if include_geometry else "null"
    geom_sql_ward = "st_asgeojson(st_simplifypreservetopology(gw.geom, 0.0015))" if include_geometry else "null"
    area_distance_sql_province = "null::double precision"
    area_distance_sql_ward = "null::double precision"
    area_distance_params: list[Any] = []
    if near_lat is not None and near_lng is not None:
        area_distance_sql_province = """
          st_distance(
            st_pointonsurface(gp.geom)::geography,
            st_setsrid(st_makepoint(%s, %s), 4326)::geography
          ) / 1000.0
        """
        area_distance_sql_ward = """
          st_distance(
            st_pointonsurface(gw.geom)::geography,
            st_setsrid(st_makepoint(%s, %s), 4326)::geography
          ) / 1000.0
        """
        area_distance_params = [near_lng, near_lat]

    if level == "province":
        row_filters: list[str] = []
        row_params: list[Any] = []
        if selected_only and province_id:
            row_filters.append("p.code = %s")
            row_params.append(province_id)
        row_where = f"where {' and '.join(row_filters)}" if row_filters else ""
        sql = f"""
            with case_scope as (
              select {case_fields}
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
              au.short_name::text as unit_type,
              gp.area_km2,
              (p.code = %s)::boolean as selected,
              {area_distance_sql_province} as distance_km,
              {geom_sql_province} as geojson,
              st_asgeojson(gp.bbox) as bbox_geojson,
              st_asgeojson(st_pointonsurface(gp.geom)) as centroid_geojson,
              coalesce(stats.case_count, 0)::int as case_count,
              stats.max_severity::int as max_severity,
              top_disease.disease::text as top_disease,
              coalesce(recent.recent_cases, '[]'::json) as recent_cases,
              coalesce(ward_count.ward_count, 0)::int as ward_count
            from public.provinces p
            left join public.administrative_units au on au.id = p.administrative_unit_id
            join public.gis_provinces gp on gp.province_code = p.code
            left join lateral (
              select count(*)::int as case_count, max(c.severity)::int as max_severity
              from case_scope c
              where c.province_id = p.code
            ) stats on true
            left join lateral (
              select c.disease
              from case_scope c
              where c.province_id = p.code
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
                  c.source,
                  c.review_status,
                  c.distance_km
                from case_scope c
                where c.province_id = p.code
                order by c.reported_at desc
                limit 5
              ) r
            ) recent on true
            left join lateral (
              select count(*)::int as ward_count
              from public.wards w
              where w.province_code = p.code
            ) ward_count on true
            {row_where}
            order by p.code
        """
        return fetch_dicts(sql, [*case_field_params, *params, province_id, *area_distance_params, *row_params])

    if not parent_id:
        raise HTTPException(status_code=400, detail="parent_id is required when level=ward")

    row_filters = ["w.province_code = %s"]
    row_params: list[Any] = [parent_id]
    if selected_only and ward_id:
        row_filters.append("w.code = %s")
        row_params.append(ward_id)
    row_where = " and ".join(row_filters)
    sql = f"""
        with case_scope as (
          select {case_fields}
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
          au.short_name::text as unit_type,
          gw.area_km2,
          (w.code = %s)::boolean as selected,
          {area_distance_sql_ward} as distance_km,
          {geom_sql_ward} as geojson,
          st_asgeojson(gw.bbox) as bbox_geojson,
          st_asgeojson(st_pointonsurface(gw.geom)) as centroid_geojson,
          coalesce(stats.case_count, 0)::int as case_count,
          stats.max_severity::int as max_severity,
          top_disease.disease::text as top_disease,
          coalesce(recent.recent_cases, '[]'::json) as recent_cases,
          0::int as ward_count
        from public.wards w
        join public.provinces p on p.code = w.province_code
        left join public.administrative_units au on au.id = w.administrative_unit_id
        join public.gis_wards gw on gw.ward_code = w.code
        left join lateral (
          select count(*)::int as case_count, max(c.severity)::int as max_severity
          from case_scope c
          where c.ward_id = w.code
        ) stats on true
        left join lateral (
          select c.disease
          from case_scope c
          where c.ward_id = w.code
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
              c.source,
              c.review_status,
              c.distance_km
            from case_scope c
            where c.ward_id = w.code
            order by c.reported_at desc
            limit 5
          ) r
        ) recent on true
        where {row_where}
        order by w.code
    """
    return fetch_dicts(sql, [*case_field_params, *params, ward_id, *area_distance_params, *row_params])


def _fetch_cases(
    *,
    since_iso: str | None = None,
    disease: str | None = None,
    plant: str | None = None,
    severity: int | None = None,
    min_severity: int | None = None,
    province_id: str | None = None,
    ward_id: str | None = None,
    source: str | None = None,
    review_status: str | None = None,
    confidence_min: float | None = None,
    near_lat: float | None = None,
    near_lng: float | None = None,
    radius_km: float | None = None,
    limit: int = 500,
) -> list[dict]:
    fields, field_params = _case_sql_fields(near_lat=near_lat, near_lng=near_lng)
    if severity is not None:
        min_severity = None
    filters, params = _case_filters(
        since_iso=since_iso or "1900-01-01T00:00:00+00:00",
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
    )
    if since_iso is None:
        parts = filters.split(" and ")
        filters = " and ".join(parts[1:]) if len(parts) > 1 else ""
        params = params[1:]
    if severity is not None:
        filters = f"{filters} and severity = %s" if filters else "severity = %s"
        params.append(severity)
    where_sql = f"where {filters}" if filters else ""
    order_sql = "distance_km asc, reported_at desc" if near_lat is not None and near_lng is not None else "reported_at desc"
    sql = f"""
        select {fields}
        from public.outbreak_cases
        {where_sql}
        order by {order_sql}
        limit %s
    """
    return fetch_dicts(sql, [*field_params, *params, limit])


@router.get("/outbreaks")
def outbreaks(
    disease: Optional[str] = None,
    plant: Optional[str] = None,
    severity: Optional[int] = None,
    min_severity: Optional[int] = None,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    source: Optional[str] = None,
    review_status: Optional[str] = None,
    confidence_min: Optional[float] = None,
    near_lat: Optional[float] = None,
    near_lng: Optional[float] = None,
    radius_km: Optional[float] = None,
    since: Optional[str] = None,
    limit: int = 500,
):
    validate_severity(severity)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)
    near_lat, near_lng = _validate_nearby(near_lat, near_lng)
    radius_km = _validate_radius_km(radius_km, default=25.0 if near_lat is not None else None)
    limit = validate_limit(limit)

    cache_key = (
        f"outbreaks|d={disease}|p={plant}|sev={severity}|minsev={min_severity}|"
        f"province={province_id}|ward={ward_id}|src={source}|review={review_status}|"
        f"conf={confidence_min}|near={near_lat},{near_lng},{radius_km}|since={since}|l={limit}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "items": cached}

    raw_items = _fetch_cases(
        since_iso=since,
        disease=disease,
        plant=plant,
        severity=severity,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
        limit=limit,
    )
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
    plant: Optional[str] = None,
    min_severity: Optional[int] = None,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    source: Optional[str] = None,
    review_status: Optional[str] = None,
    risk_min: Optional[int] = None,
    confidence_min: Optional[float] = None,
    near_lat: Optional[float] = None,
    near_lng: Optional[float] = None,
    radius_km: Optional[float] = None,
    include_geometry: bool = True,
    selected_only: bool = False,
    since: Optional[str] = None,
):
    if level not in {"province", "ward"}:
        raise HTTPException(status_code=400, detail="level must be province or ward")
    if level == "ward" and not parent_id:
        raise HTTPException(status_code=400, detail="parent_id is required when level=ward")

    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    _validate_risk_min(risk_min)
    validate_since(since)
    near_lat, near_lng = _validate_nearby(near_lat, near_lng)
    radius_km = _validate_radius_km(radius_km, default=25.0 if near_lat is not None else None)
    since_iso = _since_iso(since, since_days)

    cache_key = (
        f"outbreak_areas|level={level}|parent={parent_id}|since_days={since_days}|"
        f"d={disease}|plant={plant}|minsev={min_severity}|province={province_id}|"
        f"ward={ward_id}|source={source}|review={review_status}|risk={risk_min}|"
        f"conf={confidence_min}|near={near_lat},{near_lng},{radius_km}|geom={include_geometry}|"
        f"selected={selected_only}|since={since_iso}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    rows = _fetch_area_rows(
        level=level,
        parent_id=parent_id,
        since_iso=since_iso,
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
        include_geometry=include_geometry,
        selected_only=selected_only,
    )
    items = _rows_to_areas(rows, include_geometry=include_geometry, risk_min=risk_min)
    out = {
        "status": "success",
        "items": items,
        "level": level,
        "parent_id": parent_id,
        "since_days": since_days,
    }
    cache_set(cache_key, out, ttl_seconds=300)
    return out


@router.get("/outbreaks/admin/provinces")
def admin_provinces():
    cache_key = "outbreak_admin_provinces"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    rows = fetch_dicts(
        """
        select
          p.code::text as id,
          p.name::text as name,
          p.full_name::text as full_name,
          null::text as parent_id,
          au.short_name::text as unit_type,
          count(w.code)::int as ward_count
        from public.provinces p
        left join public.administrative_units au on au.id = p.administrative_unit_id
        left join public.wards w on w.province_code = p.code
        group by p.code, p.name, p.full_name, au.short_name
        order by p.code
        """
    )
    out = {"status": "success", "items": rows}
    cache_set(cache_key, out, ttl_seconds=3600)
    return out


@router.get("/outbreaks/admin/wards")
def admin_wards(province_id: str):
    if not province_id:
        raise HTTPException(status_code=400, detail="province_id is required")
    cache_key = f"outbreak_admin_wards|province={province_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    rows = fetch_dicts(
        """
        select
          w.code::text as id,
          w.name::text as name,
          coalesce(w.full_name, w.name)::text as full_name,
          w.province_code::text as parent_id,
          au.short_name::text as unit_type,
          0::int as ward_count
        from public.wards w
        left join public.administrative_units au on au.id = w.administrative_unit_id
        where w.province_code = %s
        order by w.code
        """,
        [province_id],
    )
    out = {"status": "success", "items": rows, "province_id": province_id}
    cache_set(cache_key, out, ttl_seconds=3600)
    return out


@router.get("/outbreaks/filter-options")
def outbreak_filter_options(
    since_days: int = 3650,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    since: Optional[str] = None,
):
    since_days = validate_since_days(since_days)
    validate_since(since)
    since_iso = _since_iso(since, since_days)
    cache_key = f"outbreak_filter_options|since={since_iso}|province={province_id}|ward={ward_id}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    filters, params = _case_filters(
        since_iso=since_iso,
        province_id=province_id,
        ward_id=ward_id,
    )
    sql = f"""
        select
          coalesce(json_agg(distinct disease) filter (where coalesce(btrim(disease), '') <> ''), '[]'::json) as diseases,
          coalesce(json_agg(distinct plant) filter (where coalesce(btrim(plant), '') <> ''), '[]'::json) as plants,
          coalesce(json_agg(distinct source) filter (where coalesce(btrim(source), '') <> ''), '[]'::json) as sources,
          coalesce(json_agg(distinct review_status) filter (where coalesce(btrim(review_status), '') <> ''), '[]'::json) as review_statuses
        from public.outbreak_cases
        where {filters}
    """
    rows = fetch_dicts(sql, params)
    row = rows[0] if rows else {}
    out = {
        "status": "success",
        "diseases": sorted(_json_value(row.get("diseases")) or []),
        "plants": sorted(_json_value(row.get("plants")) or []),
        "sources": sorted(_json_value(row.get("sources")) or []),
        "review_statuses": sorted(_json_value(row.get("review_statuses")) or []),
        "risk_levels": [0, 1, 2, 3, 4],
    }
    cache_set(cache_key, out, ttl_seconds=300)
    return out


def _normalized_text(value: str | None) -> str:
    return str(value or "").strip().lower()


def _nearby_alerts(cases: list[dict], *, plant: str | None, disease: str | None) -> list[dict]:
    plant_key = _normalized_text(plant)
    disease_key = _normalized_text(disease)
    alerts: list[dict] = []
    for item in cases:
        distance = float(item.get("distance_km") or 0)
        severity = int(item.get("severity") or 0)
        same_plant = bool(plant_key and plant_key in _normalized_text(item.get("plant")))
        same_disease = bool(disease_key and disease_key in _normalized_text(item.get("disease")))
        risk_level = compute_risk_level(1, severity)
        score = (6 - min(distance, 5)) + severity * 1.8 + (4 if same_disease else 0) + (2 if same_plant else 0)
        title = item.get("ward_name") or item.get("province_name") or "Khu vực gần bạn"
        disease_name = item.get("disease") or "bệnh cây"
        plant_name = item.get("plant")
        match_text = "trùng bệnh đang theo dõi" if same_disease else ("trùng cây của bạn" if same_plant else "gần vị trí của bạn")
        alerts.append(
            {
                "id": item.get("id"),
                "title": f"{title}: {disease_name}",
                "message": f"Cách khoảng {distance:.1f} km, mức {severity}; {match_text}.",
                "recommended_action": "Kiểm tra lá non, mặt dưới lá và giảm ẩm tán cây trong 24-48 giờ tới.",
                "severity": severity,
                "risk_level": risk_level,
                "risk_score": round(score * 10, 1),
                "distance_km": round(distance, 2),
                "plant": plant_name,
                "disease": disease_name,
                "province_id": item.get("province_id"),
                "province_name": item.get("province_name"),
                "ward_id": item.get("ward_id"),
                "ward_name": item.get("ward_name"),
                "same_plant": same_plant,
                "same_disease": same_disease,
            }
        )
    return sorted(alerts, key=lambda item: (-item["risk_score"], item["distance_km"]))[:8]


@router.get("/outbreaks/nearby-advice")
def nearby_advice(
    lat: float,
    lng: float,
    radius_km: float = 25,
    since_days: int = 30,
    plant: Optional[str] = None,
    disease: Optional[str] = None,
    min_severity: Optional[int] = None,
):
    validate_coordinates(lat, lng)
    radius_km = _validate_radius_km(radius_km) or 25
    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    since_iso = _since_iso(None, since_days)
    cache_key = (
        f"outbreak_nearby_advice|lat={lat:.4f}|lng={lng:.4f}|r={radius_km}|"
        f"days={since_days}|plant={plant}|disease={disease}|minsev={min_severity}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    cases = _fetch_cases(
        since_iso=since_iso,
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        near_lat=lat,
        near_lng=lng,
        radius_km=radius_km,
        limit=80,
    )
    if not cases and (plant or disease):
        cases = _fetch_cases(
            since_iso=since_iso,
            min_severity=min_severity,
            near_lat=lat,
            near_lng=lng,
            radius_km=radius_km,
            limit=80,
        )
    alerts = _nearby_alerts(cases, plant=plant, disease=disease)
    nearest = min((float(item.get("distance_km") or 0) for item in cases), default=None)
    out = {
        "status": "success",
        "radius_km": radius_km,
        "matched_case_count": len(cases),
        "nearest_distance_km": round(nearest, 2) if nearest is not None else None,
        "alerts": alerts,
        "recommended_filters": {
            "plant": plant,
            "disease": disease,
            "min_severity": min_severity,
            "near_lat": lat,
            "near_lng": lng,
            "radius_km": radius_km,
            "since_days": since_days,
        },
    }
    cache_set(cache_key, out, ttl_seconds=60)
    return out


def _summary_cases(
    *,
    since_iso: str,
    disease: str | None,
    plant: str | None,
    min_severity: int | None,
    province_id: str | None,
    ward_id: str | None,
    source: str | None = None,
    review_status: str | None = None,
    confidence_min: float | None = None,
    near_lat: float | None = None,
    near_lng: float | None = None,
    radius_km: float | None = None,
) -> list[dict]:
    fields, field_params = _case_sql_fields(near_lat=near_lat, near_lng=near_lng)
    filters, params = _case_filters(
        since_iso=since_iso,
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
    )
    sql = f"""
        select {fields}
        from public.outbreak_cases
        where {filters}
        order by reported_at desc
    """
    return fetch_dicts(sql, [*field_params, *params])


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
    plant: Optional[str] = None,
    min_severity: Optional[int] = None,
    province_id: Optional[str] = None,
    ward_id: Optional[str] = None,
    source: Optional[str] = None,
    review_status: Optional[str] = None,
    confidence_min: Optional[float] = None,
    near_lat: Optional[float] = None,
    near_lng: Optional[float] = None,
    radius_km: Optional[float] = None,
    since: Optional[str] = None,
):
    since_days = validate_since_days(since_days)
    validate_severity(min_severity, field="min_severity")
    validate_since(since)
    near_lat, near_lng = _validate_nearby(near_lat, near_lng)
    radius_km = _validate_radius_km(radius_km, default=25.0 if near_lat is not None else None)
    since_iso = _since_iso(since, since_days)

    cache_key = (
        f"outbreak_summary|since_days={since_days}|d={disease}|plant={plant}|minsev={min_severity}|"
        f"province={province_id}|ward={ward_id}|source={source}|review={review_status}|"
        f"conf={confidence_min}|near={near_lat},{near_lng},{radius_km}|since={since_iso}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    cases = _summary_cases(
        since_iso=since_iso,
        disease=disease,
        plant=plant,
        min_severity=min_severity,
        province_id=province_id,
        ward_id=ward_id,
        source=source,
        review_status=review_status,
        confidence_min=confidence_min,
        near_lat=near_lat,
        near_lng=near_lng,
        radius_km=radius_km,
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
