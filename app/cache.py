"""
On-disk cache for forecast/validation artifacts, keyed by (model, horizon,
data-as-of timestamp) so revisiting a previously-viewed combination is
instant. Only a combination not already cached for the current data
snapshot computes live, in the request path.
"""
import diskcache

import config

_cache = diskcache.Cache(config.CACHE_DIR)

_MISSING = object()


def _key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def get_or_compute(key: str, compute_fn, expire: int = None):
    """Return the cached value for `key`, computing and storing it via
    `compute_fn()` on a miss."""
    value = _cache.get(key, default=_MISSING)
    if value is not _MISSING:
        return value
    value = compute_fn()
    _cache.set(key, value, expire=expire)
    return value


def forecast_cache_key(model_name: str, horizon_hours: int, data_as_of) -> str:
    return _key("forecast", model_name, horizon_hours, data_as_of)


def holdout_cache_key(data_as_of, model_names: tuple) -> str:
    return _key("holdout", data_as_of, ",".join(sorted(model_names)))


def clear_all() -> None:
    _cache.clear()
