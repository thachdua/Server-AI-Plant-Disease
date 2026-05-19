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


def compute_level(count7d: int, max_sev: int | None) -> int:
    max_sev = max_sev or 0
    if count7d > 10 or max_sev >= 5:
        return 4
    if count7d >= 6 or max_sev >= 4:
        return 3
    if count7d >= 3 or max_sev >= 3:
        return 2
    if count7d >= 1:
        return 1
    return 0
