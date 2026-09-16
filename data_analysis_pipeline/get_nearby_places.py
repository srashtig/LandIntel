"""Nearby settlements (OSM) + SerpApi category searches within a fixed
10 km radius. Ports notebook cell 19 (settlements + SerpApi block)."""

from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from urllib.parse import quote_plus

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import Point

from . import config
from ._util import clean_text
from .aoi import AOI, haversine_km

NEARBY_RADIUS_KM = 10
SETTLEMENT_TAGS = {"place": ["city", "town", "village", "suburb", "neighbourhood", "hamlet", "isolated_dwelling"]}

SERPAPI_QUERIES = {
    "tourist_attractions": "tourist attractions",
    "restaurants": "restaurants",
    "hospitals": "hospitals",
    "schools": "schools",
    "markets": "markets",
    "hotels": "hotels",
    "industrial_businesses": "industrial businesses",
    "warehouses_logistics": "warehouses logistics",
}

PLACE_STYLES = {
    "settlement": {"icon": "home", "color": "purple"},
    "tourist_attractions": {"icon": "camera", "color": "cadetblue"},
    "restaurants": {"icon": "cutlery", "color": "red"},
    "hospitals": {"icon": "plus", "color": "darkred"},
    "schools": {"icon": "graduation-cap", "color": "blue"},
    "markets": {"icon": "shopping-cart", "color": "green"},
    "hotels": {"icon": "bed", "color": "orange"},
    "industrial_businesses": {"icon": "industry", "color": "gray"},
    "warehouses_logistics": {"icon": "truck", "color": "black"},
}


def google_maps_link(name, latitude, longitude, place_id=None) -> str:
    """Build a Google Maps search URL for a place.

    Args:
        name: Place name (used as the search query if ``place_id`` is given).
        latitude, longitude: Place location, decimal degrees.
        place_id: Google place id, if known, for a precise link.

    Returns:
        A ``https://www.google.com/maps/search/...`` URL.
    """

    if place_id:
        query = quote_plus(name or f"{latitude},{longitude}")
        return f"https://www.google.com/maps/search/?api=1&query={query}&query_place_id={quote_plus(str(place_id))}"
    return f"https://www.google.com/maps/search/?api=1&query={float(latitude):.7f},{float(longitude):.7f}"


def place_popup(place: dict, category: str) -> str:
    """Render one SerpApi place (as returned in a ``get_nearby_places``
    envelope) as an HTML popup body for a Folium marker.

    Args:
        place: A place dict with name/type/rating/reviews/distance_km/
            google_maps_url keys, as produced by :func:`_serpapi_search`.
        category: The SerpApi category key this place was found under
            (used as a fallback label if ``place["type"]`` is absent).

    Returns:
        An HTML string.
    """

    name = html_lib.escape(str(place.get("name") or "Unnamed place"))
    place_type = html_lib.escape(str(place.get("type") or category))
    link = html_lib.escape(place["google_maps_url"], quote=True)
    rating, reviews = place.get("rating"), place.get("reviews")
    if rating is not None:
        try:
            rating_text = f"{float(rating):g}/5 stars"
            if reviews is not None:
                rating_text += f" ({int(str(reviews).replace(',', '')):,})"
        except (TypeError, ValueError):
            rating_text = "n/a"
    else:
        rating_text = "n/a"
    return (
        f"<b>{name}</b><br>Category: {place_type}<br>Rating: {rating_text}<br>"
        f"Distance: {float(place['distance_km']):.2f} km<br>"
        f"<a href='{link}' target='_blank'>Open in Google Maps</a>"
    )


def _serpapi_search(lat, lon, api_key, query, radius_km, max_results=20):
    """Search Google Maps local results via SerpApi for one query, filtered
    to ``radius_km`` of (lat, lon).

    Args:
        lat, lon: Search center, decimal degrees.
        api_key: SerpApi API key.
        query: Free-text search query (e.g. "schools").
        radius_km: Only results within this radius are kept (SerpApi's own
            ``ll`` zoom-radius parameter is a hint, not a hard filter, so
            results are re-filtered here by haversine distance).
        max_results: Maximum number of results to return, nearest-first.

    Returns:
        A list of place dicts (name, type, rating, reviews, address,
        latitude, longitude, distance_km, place_id, google_maps_url),
        nearest-first, capped at ``max_results``.
    """

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "engine": "google_maps", "type": "search", "q": query,
            "ll": f"@{lat},{lon},{max(1000, int(radius_km * 2000))}m",
            "hl": "en", "api_key": api_key, "start": 0,
        },
        timeout=60,
    )
    response.raise_for_status()
    output = []
    for place in response.json().get("local_results", []):
        coordinates = place.get("gps_coordinates", {})
        latitude, longitude = coordinates.get("latitude"), coordinates.get("longitude")
        if latitude is None or longitude is None:
            continue
        distance_km = haversine_km(lat, lon, latitude, longitude)
        if distance_km > radius_km:
            continue
        name = place.get("title") or "Unnamed place"
        output.append({
            "name": name, "type": place.get("type"), "rating": place.get("rating"),
            "reviews": place.get("reviews"), "address": place.get("address"),
            "latitude": float(latitude), "longitude": float(longitude),
            "distance_km": round(distance_km, 3), "place_id": place.get("place_id"),
            "google_maps_url": google_maps_link(name, latitude, longitude, place.get("place_id")),
        })
    return sorted(output, key=lambda item: item["distance_km"])[:max_results]


def get_nearby_places(aoi: AOI, radius_km: float = NEARBY_RADIUS_KM) -> dict:
    """Fetch nearby settlements (from OpenStreetMap) and category place
    searches (tourist attractions, restaurants, hospitals, schools,
    markets, hotels, industrial businesses, warehouses/logistics — via
    SerpApi/Google Maps) within ``radius_km`` of the AOI's center. This
    search radius is independent of (and defaults larger than) the AOI's
    own radius, matching the notebook's fixed 10 km neighborhood search.

    Args:
        aoi: The area of interest; only its center (aoi.lat, aoi.lon) and
            ``utm_crs`` are used — the search radius here is independent
            of ``aoi.radius_km``.
        radius_km: Search radius in kilometers around the AOI center.
            Defaults to 10.0.

    Returns:
        A standard envelope dict. ``observations`` = {"settlements_gdf":
        GeoDataFrame, "serp_places": {category: [place, ...]}, "summary": {
        "settlements": {settlement_count_10km, distance_to_nearest_settlement_km,
        nearest_settlement_name, nearest_settlement_type, search_radius_km,
        settlements (list, nearest-first), source},
        "nearby_places": {search_radius_km, serpapi_enabled, categories:
        [{category, count, nearest, places (full list)}, ...], source}}}.
        Distances in kilometers.
    """

    warnings: list[str] = []

    center_metric = gpd.GeoSeries([Point(aoi.lon, aoi.lat)], crs="EPSG:4326").to_crs(aoi.utm_crs).iloc[0]
    nearby_aoi_geom = gpd.GeoSeries(
        [center_metric.buffer(radius_km * 1000)], crs=aoi.utm_crs
    ).to_crs("EPSG:4326").iloc[0]

    try:
        nearby_settlements = ox.features.features_from_polygon(nearby_aoi_geom, tags=SETTLEMENT_TAGS).reset_index()
        nearby_settlements = nearby_settlements[nearby_settlements.geometry.notna()].copy()
    except Exception as error:
        warnings.append(f"OSM settlement query failed: {error}")
        nearby_settlements = gpd.GeoDataFrame(columns=["name", "place", "geometry"], geometry="geometry", crs="EPSG:4326")

    nearby_settlements_metric = nearby_settlements.to_crs(aoi.utm_crs)
    if not nearby_settlements_metric.empty:
        settlement_points = nearby_settlements_metric.copy()
        settlement_points["geometry"] = settlement_points.geometry.representative_point()
        settlement_points["distance_km"] = settlement_points.geometry.distance(center_metric) / 1000
        nearest_row = settlement_points.sort_values("distance_km").iloc[0]
        nearest_settlement = {
            "distance_km": float(nearest_row["distance_km"]),
            "name": nearest_row.get("name"),
            "place_type": nearest_row.get("place"),
        }
    else:
        nearest_settlement = {"distance_km": None, "name": None, "place_type": None}

    # Full settlement list, nearest-first. Previously only the single
    # nearest settlement (plus a bare count) made it into the JSON; the
    # rest existed only as unlabeled map markers.
    nearby_settlements_list = []
    if not nearby_settlements_metric.empty:
        for _, row in settlement_points.sort_values("distance_km").iterrows():
            nearby_settlements_list.append({
                "name": clean_text(row.get("name")),
                "place_type": clean_text(row.get("place")),
                "distance_km": float(row["distance_km"]),
            })

    settlement_summary = {
        "settlement_count_10km": int(len(nearby_settlements)),
        "distance_to_nearest_settlement_km": nearest_settlement["distance_km"],
        "nearest_settlement_name": nearest_settlement["name"],
        "nearest_settlement_type": nearest_settlement["place_type"],
        "search_radius_km": radius_km,
        "settlements": nearby_settlements_list,
        "source": "OpenStreetMap via OSMnx",
    }

    serp_places: dict[str, list] = {}
    api_key = None
    if config.SERPAPI_DISABLED:
        warnings.append("SERPAPI_DISABLED is set - SerpApi searches skipped.")
    else:
        api_key = config.SERPAPI_KEY

    if api_key:
        for category, query in SERPAPI_QUERIES.items():
            try:
                serp_places[category] = _serpapi_search(aoi.lat, aoi.lon, api_key, query, radius_km)
            except Exception as error:
                warnings.append(f"SerpApi category '{category}' failed: {error}")
                serp_places[category] = []
    else:
        if not config.SERPAPI_DISABLED:
            warnings.append("SERPAPI_KEY not set - SerpApi searches skipped.")
        for category in SERPAPI_QUERIES:
            serp_places[category] = []

    nearby_places_summary = {
        "search_radius_km": radius_km,
        "serpapi_enabled": bool(api_key),
        # "places" is the full fetched list (previously only rendered as map
        # markers/popups); "nearest" is kept as a convenience duplicate of
        # places[0] for quick access.
        "categories": [
            {"category": category, "count": len(items), "nearest": items[0] if items else None, "places": items}
            for category, items in serp_places.items()
        ],
        "source": "SerpApi / Google Maps local results",
    }

    return {
        "dataset": "nearby_places",
        "source": "OpenStreetMap via OSMnx + SerpApi / Google Maps local results",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": radius_km},
        "observations": {
            "settlements_gdf": nearby_settlements,
            "serp_places": serp_places,
            "summary": {"settlements": settlement_summary, "nearby_places": nearby_places_summary},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "high" if api_key else "medium",
        "warnings": warnings,
        "limitations": ["SerpApi category counts/ratings can drift between runs (live search results)."],
    }
