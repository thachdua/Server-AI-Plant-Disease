from __future__ import annotations


def point_in_bbox(lat: float, lng: float, bbox: list[float]) -> bool:
    return bbox[1] <= lat <= bbox[3] and bbox[0] <= lng <= bbox[2]


def point_in_ring(lng: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (
            lng < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def point_in_polygon(lng: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon:
        return False
    if not point_in_ring(lng, lat, polygon[0]):
        return False
    for hole in polygon[1:]:
        if point_in_ring(lng, lat, hole):
            return False
    return True


def point_in_multipolygon(lng: float, lat: float, coords) -> bool:
    try:
        for polygon in coords:
            if point_in_polygon(lng, lat, polygon):
                return True
    except Exception:
        return False
    return False


def compute_level(count7d: int, max_sev: int | None = None) -> int:
    # Province outbreak level is based on case volume in the selected time window.
    # max_sev is kept for backwards-compatible callers, but does not affect level.
    if count7d > 10:
        return 4
    if count7d >= 6:
        return 3
    if count7d >= 3:
        return 2
    if count7d >= 1:
        return 1
    return 0


def compute_risk_level(case_count: int, max_sev: int | None = None) -> int:
    severity = max_sev or 0
    if case_count >= 11 or severity >= 5:
        return 4
    if case_count >= 6 or severity >= 4:
        return 3
    if case_count >= 3:
        return 2
    if case_count >= 1:
        return 1
    return 0


def compute_risk_score(case_count: int, max_sev: int | None = None) -> float:
    severity = max(0, min(max_sev or 0, 5))
    count_component = min(case_count, 20) / 20.0 * 70.0
    severity_component = severity / 5.0 * 30.0
    return round(count_component + severity_component, 2)
