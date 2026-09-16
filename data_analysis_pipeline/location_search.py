"""Location search for the app's location pickers: a raw "lat, lon"
detector (:func:`parse_latlon_query`), a live Google Maps place/address
search (:func:`search_google_maps`, via the same SerpApi integration
:mod:`get_nearby_places` uses), and the original fuzzy village/subdistrict/
district search (:func:`search_places`, backed by the index built by
``scripts/build_location_search_index.py`` — MP-only, offline, no API
call) — kept for callers that specifically want a precise Survey-of-India
village centroid rather than a general Google Maps result (currently
the original notebook's chat-based custom-facts parser, resolving a place mentioned
in a chat message).

Uses ``rapidfuzz`` (not ``difflib``) for speed and substring behavior at
~56k rows.
"""

from __future__ import annotations

import re

import requests

from . import config

_LATLON_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_latlon_query(query: str) -> tuple[float, float] | None:
    """Parse a query string as raw "lat, lon" (or "lat lon") coordinates,
    e.g. typed directly into a location-search box instead of a place name.

    Args:
        query: The raw search text.

    Returns:
        ``(lat, lon)`` if ``query`` is exactly a comma/whitespace-separated
        pair of numbers within valid lat/lon ranges, else ``None`` (so the
        caller falls through to treating it as a place-name search).
    """

    match = _LATLON_PATTERN.match(query or "")
    if not match:
        return None
    lat, lon = float(match.group(1)), float(match.group(2))
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    return None


# A loose Madhya-Pradesh-centered ranking bias (SerpApi's `ll` is a hint,
# not a hard filter) — nudges locally relevant results up without
# preventing a specific, well-named query elsewhere from being found.
_GOOGLE_MAPS_SEARCH_BIAS_LAT = 23.2599
_GOOGLE_MAPS_SEARCH_BIAS_LON = 77.4126


def search_google_maps(query: str, limit: int = 10) -> list[dict]:
    """Free-text place/address search via SerpApi's Google Maps engine —
    genuinely "search like Google Maps" (any address, landmark, village,
    or business name, anywhere), rather than a fixed local index. The
    app's Madhya-Pradesh-only gate (``mp_boundary.is_in_mp``) is applied
    afterwards, on whatever coordinates the user ends up picking — not
    here, so this can freely return results outside MP too (e.g. a
    workplace used as a "custom facts" reference location).

    Args:
        query: Free-text search, e.g. "Statue of Unity" or "Budasa,
            Dewas". Check :func:`parse_latlon_query` first if the caller
            wants raw "lat, lon" text handled without an API call — this
            function always makes a live request.
        limit: Maximum number of results to return.

    Returns:
        Up to ``limit`` place dicts: ``{"name", "address", "centroid_lat",
        "centroid_lon", "place_id", "google_maps_url"}``. Empty list if
        ``query`` is blank, the request failed, or nothing was found.
    """

    from .get_nearby_places import google_maps_link

    query = (query or "").strip()
    if not query or config.SERPAPI_DISABLED:
        return []

    try:
        api_key = config.SERPAPI_KEY
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "engine": "google_maps", "type": "search", "q": query,
                "ll": f"@{_GOOGLE_MAPS_SEARCH_BIAS_LAT},{_GOOGLE_MAPS_SEARCH_BIAS_LON},6z",
                "hl": "en", "api_key": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:  # noqa: BLE001 - a search-box lookup degrades to "no results", not a crash
        return []

    # A single well-specified query (e.g. an exact address) sometimes comes
    # back as `place_results` (one place) instead of `local_results` (a list).
    candidates = data.get("local_results") or ([data["place_results"]] if data.get("place_results") else [])

    results = []
    for place in candidates[:limit]:
        coords = place.get("gps_coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        name = place.get("title") or "Unnamed place"
        results.append({
            "name": name,
            "address": place.get("address"),
            "centroid_lat": float(lat),
            "centroid_lon": float(lon),
            "place_id": place.get("place_id"),
            "google_maps_url": google_maps_link(name, lat, lon, place.get("place_id")),
        })
    return results


_index_df = None


def _load_index():
    """Load and cache (module-level, once per process) the location search
    index from ``config.LOCATION_SEARCH_INDEX``.

    Returns:
        A pandas DataFrame with columns "name", "level" ("village" |
        "subdistrict" | "district"), "centroid_lat", "centroid_lon",
        "district", "subdistric".

    Raises:
        FileNotFoundError: If the index hasn't been built yet (run
            ``scripts/build_location_search_index.py`` once first).
    """

    global _index_df
    if _index_df is not None:
        return _index_df

    import pandas as pd

    if not config.LOCATION_SEARCH_INDEX.exists():
        raise FileNotFoundError(
            f"Location search index not found at {config.LOCATION_SEARCH_INDEX}. "
            "Run `python scripts/build_location_search_index.py` once first."
        )

    _index_df = pd.read_parquet(config.LOCATION_SEARCH_INDEX)
    return _index_df


def search_places(query: str, limit: int = 10) -> list[dict]:
    """Fuzzy-search Madhya Pradesh village/subdistrict/district names and
    return their centroids, for resolving a free-text place name to
    coordinates (e.g. a location picker's search box).

    Args:
        query: Free-text place name to search for, e.g. "Budasa" or
            "Dewas district". Case-insensitive, tolerates typos/partial
            names (fuzzy matching via rapidfuzz).
        limit: Maximum number of matches to return. Defaults to 10.

    Returns:
        Up to ``limit`` matches, best match first, each a dict:
        {"name": str, "level": "village" | "subdistrict" | "district",
         "centroid_lat": float, "centroid_lon": float (decimal degrees,
         WGS84), "district": str | None, "subdistric": str | None
         (the containing subdistrict/tehsil name), "score": float (0-100
         fuzzy-match quality, higher is better)}. Empty list if ``query``
        is blank.
    """

    from rapidfuzz import fuzz, process

    query = (query or "").strip()
    if not query:
        return []

    index_df = _load_index()
    names = index_df["name"].tolist()

    matches = process.extract(query, names, scorer=fuzz.WRatio, limit=limit)

    results = []
    seen_rows = set()
    for _name, score, row_idx in matches:
        if row_idx in seen_rows:
            continue
        seen_rows.add(row_idx)
        row = index_df.iloc[row_idx]
        results.append({
            "name": row["name"],
            "level": row["level"],
            "centroid_lat": float(row["centroid_lat"]),
            "centroid_lon": float(row["centroid_lon"]),
            "district": row.get("district"),
            "subdistric": row.get("subdistric"),
            "score": float(score),
        })

    return results
