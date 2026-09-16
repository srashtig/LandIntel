"""Fast "is this point in Madhya Pradesh" gate, backed by the simplified
boundary cache built by ``scripts/build_mp_boundary_cache.py``.

The cache is a single dissolved+simplified polygon in EPSG:4326 (a few MB),
loaded once per process and reused — milliseconds per call instead of
touching the 230 MB source ``vb_soi_mp.GeoJSON``.
"""

from __future__ import annotations

from shapely.geometry import Point

from . import config

_boundary_geometry = None


def _load_boundary():
    """Load and cache (module-level, once per process) the simplified MP
    outline geometry from ``config.MP_BOUNDARY_CACHE``.

    Returns:
        A single shapely (Multi)Polygon in EPSG:4326.

    Raises:
        FileNotFoundError: If the cache file hasn't been built yet (run
            ``scripts/build_mp_boundary_cache.py`` once first).
    """

    global _boundary_geometry
    if _boundary_geometry is not None:
        return _boundary_geometry

    import geopandas as gpd

    if not config.MP_BOUNDARY_CACHE.exists():
        raise FileNotFoundError(
            f"MP boundary cache not found at {config.MP_BOUNDARY_CACHE}. "
            "Run `python scripts/build_mp_boundary_cache.py` once first."
        )

    gdf = gpd.read_file(config.MP_BOUNDARY_CACHE)
    _boundary_geometry = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    return _boundary_geometry


def is_in_mp(lat: float, lon: float) -> bool:
    """Check whether a point falls within Madhya Pradesh.

    Fast (milliseconds), backed by a small cached/simplified boundary
    polygon rather than the 230 MB source village-boundary file — a UI-level
    gate for "this pipeline's datasets are MP-only", not a precise cadastral
    boundary check.

    Args:
        lat: Latitude in decimal degrees (WGS84 / EPSG:4326).
        lon: Longitude in decimal degrees (WGS84 / EPSG:4326).

    Returns:
        True if (lat, lon) falls within (or on) the cached Madhya Pradesh
        outline, False otherwise.
    """

    boundary = _load_boundary()
    return bool(boundary.covers(Point(lon, lat)))
