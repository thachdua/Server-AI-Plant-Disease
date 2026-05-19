import time

_cache: dict[str, tuple[float, object]] = {}


def cache_get(key: str):
    item = _cache.get(key)
    if not item:
        return None
    exp, val = item
    if time.time() > exp:
        _cache.pop(key, None)
        return None
    return val


def cache_set(key: str, val, ttl_seconds: int):
    _cache[key] = (time.time() + ttl_seconds, val)
