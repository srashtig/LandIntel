"""Roads, railways and OSM-mapped water via OSMnx.

Ports notebook cells 4-6 (feature queries) and the road/rail/water portion
of cell 61 (vector-layer statistics: nearest feature, lengths, density).
"""

from __future__ import annotations

from datetime import datetime, timezone

import geopandas as gpd
import osmnx as ox
import pandas as pd

from .aoi import AOI

# osmnx's cache_folder is set once, centrally, in config.py (importing
# `data_analysis_pipeline` at all already runs that module) — see the note
# there on why it must be an absolute path.

# OSMnx's default (180s) means a stalled/unreachable Overpass connection for
# one layer (roads, rail, or water — queried independently, see
# _fetch_osm_layer) can silently eat 3 minutes before this module's own
# try/except gives up on it and reports "no features found" anyway — e.g. a
# real run's railway query timed out after 180s and still returned zero
# features. Failing fast loses nothing a graceful degrade wasn't already
# going to report; it just stops waiting for it.
ox.settings.requests_timeout = 15

# Before every actual Overpass request, OSMnx separately pings the
# Overpass "/status" endpoint to see if it should pause for rate-limiting
# (osmnx/_overpass.py's _get_overpass_pause). When *that* status check
# itself can't connect — confirmed happening for real against the public
# instance — OSMnx falls back to a hardcoded 60s "be safe" sleep before
# even attempting the real request, on top of that request's own timeout.
# This isn't tunable via requests_timeout at all (it's a separate
# hardcoded default_pause=60 inside that function), so a real stall here
# was costing ~15s (failed status check) + 60s (fallback sleep) + 15s
# (failed real request) ≈ 90s per layer, regardless of the setting above.
# Disabling rate-limit checking entirely skips this whole pre-check — we
# make only 3 requests per analysis run, not a tight loop, so the
# politeness this buys isn't worth an unbounded 60s dead-sleep on a bad
# connection.
ox.settings.overpass_rate_limit = False

ROAD_TAGS = {
    "highway": [
        "motorway", "trunk", "primary", "secondary",
        "tertiary", "unclassified", "residential",
    ]
}
RAIL_TAGS = {"railway": ["rail", "light_rail", "subway", "tram"]}
WATER_TAGS = {
    "natural": ["water"],
    "waterway": ["river", "stream", "canal", "drain"],
}
MAJOR_ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary"]

# Expanding search radii (km) used to find the nearest major road when none
# exists within the AOI itself -- most sites are several/many km from a
# highway, so "no major road in a 5 km AOI" shouldn't mean "unknown
# distance to one." Stops at the first radius that finds anything.
_MAJOR_ROAD_SEARCH_RADII_KM = [15, 30, 60, 120]


def _clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reset the OSMnx multi-index to columns and drop rows with no geometry."""

    gdf = gdf.reset_index()
    return gdf[gdf.geometry.notna()].copy()


def _fetch_osm_layer(polygon, tags: dict, label: str, warnings: list[str]) -> gpd.GeoDataFrame:
    """Query Overpass (via OSMnx, its configured default endpoint) for
    ``tags`` within ``polygon``.

    Never raises: a failed query is treated as "no features found" for
    that layer (an empty GeoDataFrame) with a note appended to
    ``warnings`` — one layer's Overpass failure degrades just that layer
    (roads, or railways, or water), not the whole ``get_osm_features``
    call, since the three are otherwise independent.

    (A retry-with-fallback-mirror version of this was tried and reverted —
    it didn't clearly help and made a slow/rate-limited response take even
    longer in practice, likely compounding with OSMnx's own internal
    retry-on-429 backoff. Overpass's public instance being occasionally
    slow/unavailable is expected and self-resolving; this function's job is
    just to make sure that doesn't take the rest of the run down with it.)

    Args:
        polygon: AOI polygon in WGS84, as passed to
            ``ox.features.features_from_polygon``.
        tags: OSM tag filter dict for this layer (e.g. :data:`ROAD_TAGS`).
        label: Short human-readable name for this layer, used only in the
            warning message if the query fails.
        warnings: List this function appends a failure note to, in place,
            if the query fails.

    Returns:
        A cleaned GeoDataFrame (see :func:`_clean`) — possibly empty if the
        query failed or genuinely no matching features exist.
    """

    try:
        return _clean(ox.features.features_from_polygon(polygon, tags=tags))
    except Exception as error:  # noqa: BLE001 - network dependent, deliberately broad
        warnings.append(f"{label} query failed: {error}")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


def _nearest_feature(gdf: gpd.GeoDataFrame, utm_crs, site_point_utm):
    """Find the feature in ``gdf`` closest to a point.

    Args:
        gdf: Candidate features (any geometry type), in any CRS.
        utm_crs: Local metric CRS to measure distance in.
        site_point_utm: The reference point, already in ``utm_crs``.

    Returns:
        ``(distance_m, row)`` for the nearest feature, or ``(None, None)``
        if ``gdf`` is empty or has no valid geometries.
    """

    if gdf is None or len(gdf) == 0:
        return None, None
    valid = gdf[gdf.geometry.notna()]
    if valid.empty:
        return None, None
    distances = valid.to_crs(utm_crs).geometry.distance(site_point_utm)
    idx = distances.idxmin()
    return float(distances.loc[idx]), valid.loc[idx]


def _label(row, fields):
    """Return the first non-empty value of ``row`` among ``fields``, or None."""

    if row is None:
        return None
    for field in fields:
        if field in row.index:
            value = row.get(field)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return None


def _find_nearest_major_road_beyond_aoi(aoi: AOI, warnings: list[str]):
    """Expanding-radius search for the nearest major road (motorway/trunk/
    primary/secondary) when none was found within the AOI itself.

    Tries :data:`_MAJOR_ROAD_SEARCH_RADII_KM` in order, stopping at the
    first radius that turns up any major road at all -- this keeps the
    common case (a highway within 15-30 km) cheap, only reaching for a
    wider, heavier Overpass query if genuinely nothing closer exists.
    Never raises: an Overpass failure at any radius just tries the next
    one (or gives up) with a note appended to ``warnings``.

    Args:
        aoi: The area of interest.
        warnings: List this function appends a failure note to, in place.

    Returns:
        ``(distance_m, highway_class, name)`` for the nearest major road
        found, or ``(None, None, None)`` if none was found at any radius
        (or every attempt failed).
    """

    utm_crs = aoi.utm_crs
    site_point_utm = aoi.center_utm

    for radius_km in _MAJOR_ROAD_SEARCH_RADII_KM:
        try:
            found = ox.features.features_from_point(
                (aoi.lat, aoi.lon), tags={"highway": MAJOR_ROAD_CLASSES}, dist=radius_km * 1000,
            )
            found = _clean(found)
        except Exception as error:  # noqa: BLE001 - network dependent, deliberately broad
            warnings.append(f"expanded major-road search at {radius_km} km failed: {error}")
            continue

        lines = found[found.geometry.geom_type.isin(["LineString", "MultiLineString"])] if not found.empty else found
        if lines.empty:
            continue

        distance_m, row = _nearest_feature(lines, utm_crs, site_point_utm)
        if distance_m is not None:
            return distance_m, _label(row, ["highway"]), _label(row, ["name", "ref"])

    return None, None, None


def _envelope(dataset: str, aoi: AOI, observations: dict, warnings: list[str]) -> dict:
    """Build the standard envelope dict shared by :func:`get_roads`,
    :func:`get_railways` and :func:`get_water` — each is its own
    fault-isolated orchestrator stage (see the module docstring and
    ``orchestrator.py``'s "OSM roads"/"OSM railways"/"OSM water" stages),
    run separately so a stuck/rate-limited Overpass query for one layer
    (rail was the observed offender) doesn't block the other two, and so
    each can be *started* early and joined later, overlapped with unrelated
    GEE stages elsewhere in the pipeline, instead of all three being fetched
    back-to-back up front."""

    return {
        "dataset": dataset,
        "source": "OpenStreetMap via OSMnx",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
        "observations": observations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "high",
        "warnings": warnings,
        "limitations": ["OSM coverage/tagging completeness varies by area."],
    }


def get_roads(aoi: AOI) -> dict:
    """Fetch OSM roads within the AOI and compute nearest-feature/length/
    density statistics (including an expanding-radius search for the
    nearest major road beyond the AOI if none is inside it).

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"gdf": GeoDataFrame,
        "summary": {features, total_length_km, length_km_by_class,
        density_km_per_km2, nearest_road_m, nearest_road_class,
        nearest_road_name, nearest_major_road_m, nearest_major_road_class,
        nearest_major_road_name, nearest_major_road_beyond_aoi, source}}.
        Distances in meters, lengths in km.
    """

    warnings: list[str] = []
    roads = _fetch_osm_layer(aoi.polygon_4326, ROAD_TAGS, "roads", warnings)
    utm_crs = aoi.utm_crs
    site_point_utm = aoi.center_utm

    road_lines = pd.DataFrame()
    if not roads.empty:
        road_lines = roads[
            roads.geometry.notna()
            & roads.geometry.geom_type.isin(["LineString", "MultiLineString"])
        ].copy()

    if isinstance(road_lines, gpd.GeoDataFrame) and not road_lines.empty:
        road_lines_utm = road_lines.to_crs(utm_crs)
        road_lines["length_km"] = road_lines_utm.geometry.length / 1000

        length_by_class = (
            road_lines.groupby(road_lines["highway"].astype(str))["length_km"]
            .sum()
            .sort_values(ascending=False)
        )

        nearest_road_m, nearest_road = _nearest_feature(road_lines, utm_crs, site_point_utm)
        major_roads = road_lines[road_lines["highway"].astype(str).isin(MAJOR_ROAD_CLASSES)]
        nearest_major_m, nearest_major = _nearest_feature(major_roads, utm_crs, site_point_utm)
        nearest_major_class = _label(nearest_major, ["highway"])
        nearest_major_name = _label(nearest_major, ["name", "ref"])
        major_road_beyond_aoi = False

        if nearest_major_m is None:
            nearest_major_m, nearest_major_class, nearest_major_name = _find_nearest_major_road_beyond_aoi(aoi, warnings)
            major_road_beyond_aoi = nearest_major_m is not None

        road_summary = {
            "features": int(len(road_lines)),
            "total_length_km": float(road_lines["length_km"].sum()),
            "length_km_by_class": {name: float(v) for name, v in length_by_class.items()},
            "density_km_per_km2": float(road_lines["length_km"].sum() / aoi.area_km2),
            "nearest_road_m": nearest_road_m,
            "nearest_road_class": _label(nearest_road, ["highway"]),
            "nearest_road_name": _label(nearest_road, ["name", "ref"]),
            "nearest_major_road_m": nearest_major_m,
            "nearest_major_road_class": nearest_major_class,
            "nearest_major_road_name": nearest_major_name,
            "nearest_major_road_beyond_aoi": major_road_beyond_aoi,
            "source": "OpenStreetMap via OSMnx",
        }
    else:
        nearest_major_m, nearest_major_class, nearest_major_name = _find_nearest_major_road_beyond_aoi(aoi, warnings)
        road_summary = {
            "features": 0,
            "nearest_major_road_m": nearest_major_m,
            "nearest_major_road_class": nearest_major_class,
            "nearest_major_road_name": nearest_major_name,
            "nearest_major_road_beyond_aoi": nearest_major_m is not None,
            "source": "OpenStreetMap via OSMnx",
        }

    return _envelope("osm_roads", aoi, {"gdf": roads, "summary": road_summary}, warnings)


def get_railways(aoi: AOI) -> dict:
    """Fetch OSM railways within the AOI and find the nearest one.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"gdf": GeoDataFrame,
        "summary": {features, nearest_rail_m, nearest_rail_name, source}}.
        Distance in meters.
    """

    warnings: list[str] = []
    railways = _fetch_osm_layer(aoi.polygon_4326, RAIL_TAGS, "railways", warnings)
    utm_crs = aoi.utm_crs
    site_point_utm = aoi.center_utm

    if isinstance(railways, gpd.GeoDataFrame) and not railways.empty:
        nearest_rail_m, nearest_rail = _nearest_feature(railways, utm_crs, site_point_utm)
        rail_summary = {
            "features": int(len(railways)),
            "nearest_rail_m": nearest_rail_m,
            "nearest_rail_name": _label(nearest_rail, ["name", "railway"]),
            "source": "OpenStreetMap via OSMnx",
        }
    else:
        rail_summary = {"features": 0, "source": "OpenStreetMap via OSMnx"}

    return _envelope("osm_railways", aoi, {"gdf": railways, "summary": rail_summary}, warnings)


def get_water(aoi: AOI) -> dict:
    """Fetch OSM-mapped water (bodies + waterways) within the AOI and
    compute area/length/nearest-feature statistics.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"gdf": GeoDataFrame,
        "summary": {features, waterbody_area_km2, waterway_length_km,
        waterbodies, waterways, waterway_types, nearest_water_m,
        nearest_water_type, nearest_water_name, waterbody_share_of_aoi,
        source}}. Distance in meters, areas in km², lengths in km.
    """

    warnings: list[str] = []
    water = _fetch_osm_layer(aoi.polygon_4326, WATER_TAGS, "water", warnings)
    utm_crs = aoi.utm_crs
    site_point_utm = aoi.center_utm

    if not water.empty:
        water_valid = water[water.geometry.notna()].copy()
        water_utm = water_valid.to_crs(utm_crs)

        is_area = water_utm.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        is_line = water_utm.geometry.geom_type.isin(["LineString", "MultiLineString"])

        nearest_water_m, nearest_water = _nearest_feature(water_valid, utm_crs, site_point_utm)

        waterway_counts = (
            water_valid["waterway"].dropna().astype(str).value_counts().to_dict()
            if "waterway" in water_valid.columns else {}
        )

        water_summary = {
            "features": int(len(water_valid)),
            "waterbody_area_km2": float(water_utm.loc[is_area].geometry.area.sum() / 1e6),
            "waterway_length_km": float(water_utm.loc[is_line].geometry.length.sum() / 1000),
            "waterbodies": int(is_area.sum()),
            "waterways": int(is_line.sum()),
            "waterway_types": {k: int(v) for k, v in waterway_counts.items()},
            "nearest_water_m": nearest_water_m,
            "nearest_water_type": _label(nearest_water, ["waterway", "natural", "water"]),
            "nearest_water_name": _label(nearest_water, ["name"]),
            "source": "OpenStreetMap via OSMnx",
        }
        water_summary["waterbody_share_of_aoi"] = water_summary["waterbody_area_km2"] / aoi.area_km2
    else:
        water_summary = {"features": 0, "source": "OpenStreetMap via OSMnx"}

    return _envelope("osm_water", aoi, {"gdf": water, "summary": water_summary}, warnings)


