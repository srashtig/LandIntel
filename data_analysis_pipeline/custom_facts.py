"""User-supplied, run-specific facts that no automated data source can
provide: distance/travel-time to a reference location the user cares
about, and price/area for the plot being evaluated.

Pure functions only (no Streamlit/notebook coupling) so both call sites —
Stage B's Streamlit "Custom facts" expander and Stage C's chat-driven
notebook flow — share one implementation. Callers are responsible for
persisting the returned ``site_summary``/map HTML to disk; nothing here
does file I/O itself.

:func:`merge_custom_facts` writes directly into the same ``site_summary``
keys the original notebook's scoring engine and :mod:`build_map` already know
how to read (a new ``reference_location`` section, and the existing
``land_price`` section) — so once saved, both the decision engine and the
Stage C notebook pick these facts up with no further plumbing.
"""

from __future__ import annotations

import math

import requests

from . import config

_OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"
_SERPAPI_URL = "https://serpapi.com/search"

# Icon filenames Google's transit directions use to mark a "Transit" leg as
# rail-like (train/subway/tram) vs. everything else (bus, private coach).
_TRANSIT_RAIL_ICON_HINTS = ("rail", "train", "subway", "tram")

# acres per 1 unit, for normalizing a user-entered price/area to INR/acre —
# the unit decision_engine.py's existing "land_price" factor already uses.
_ACRES_PER_UNIT = {
    "acre": 1.0,
    "hectare": 2.4710538146717,
    "sqft": 1.0 / 43_560.0,
    "sqm": 1.0 / 4_046.8564224,
}
_SQFT_PER_ACRE = 43_560.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""

    earth_radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def fetch_road_travel_time(
    site_lat: float, site_lon: float, ref_lat: float, ref_lon: float, timeout_s: float = 8.0
) -> dict | None:
    """Best-effort road driving time/distance via the free public OSRM
    demo routing server (no API key needed). Never raises — any failure
    (network error, timeout, non-200, malformed response, no route found)
    degrades to ``None`` so the caller can fall back to straight-line
    distance alone.

    This is a public demo endpoint, not guaranteed reliable or suited to
    heavy use; swap in Google Directions/Mapbox/a self-hosted OSRM here if
    that ever becomes a problem — nothing else needs to change.

    Args:
        site_lat: Analyzed site latitude.
        site_lon: Analyzed site longitude.
        ref_lat: Reference location latitude.
        ref_lon: Reference location longitude.
        timeout_s: Request timeout in seconds.

    Returns:
        ``{"travel_time_min": float, "travel_distance_km": float}``, or
        ``None`` if a route could not be obtained.
    """

    url = f"{_OSRM_ROUTE_URL}/{site_lon},{site_lat};{ref_lon},{ref_lat}"
    try:
        response = requests.get(url, params={"overview": "false"}, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        route = data["routes"][0]
        return {
            "travel_time_min": route["duration"] / 60.0,
            "travel_distance_km": route["distance"] / 1000.0,
        }
    except Exception:  # noqa: BLE001 - routing is best-effort; any failure just degrades to None
        return None


def fetch_transit_time(
    site_lat: float, site_lon: float, ref_lat: float, ref_lon: float, timeout_s: float = 15.0
) -> dict | None:
    """Best-effort public-transit itinerary via Google Maps Directions
    (through SerpApi's ``google_maps_directions`` engine, ``travel_mode=3``
    = transit), summarized into riding time by bus/train plus the
    connecting walk distance.

    IMPORTANT — unlike :func:`fetch_road_travel_time`, this is NOT a
    stable average: Google returns ONE specific scheduled itinerary (a
    real departure/arrival clock time for whatever service happens to run
    next), so both the duration and which bus/train it finds depend
    entirely on when this is called. Treat the result as "one real option
    found just now," never as a typical/average travel time — this is why
    it is surfaced for display only, never as a decision_engine scoring
    input. There is also no reliable way to force "bus only" vs "train
    only" as independent alternatives through this API (a ``transit_mode``
    param exists but was verified to have no effect); whichever mode(s)
    appear are just whatever Google's own routing happened to pick as
    fastest overall for that one itinerary.

    Args:
        site_lat: Analyzed site latitude.
        site_lon: Analyzed site longitude.
        ref_lat: Reference location latitude.
        ref_lon: Reference location longitude.
        timeout_s: Request timeout in seconds.

    Returns:
        ``{"bus_duration_min": float | None, "train_duration_min":
        float | None, "walk_km": float, "total_duration_min": float,
        "departs_at": str | None, "arrives_at": str | None}``, or ``None``
        if no transit itinerary could be found or the request failed.
        ``bus_duration_min``/``train_duration_min`` are ``None`` when that
        mode isn't part of the one itinerary Google returned (not
        necessarily proof no such service exists at all).
    """

    if config.SERPAPI_DISABLED:
        return None

    try:
        response = requests.get(
            _SERPAPI_URL,
            params={
                "engine": "google_maps_directions",
                "start_coords": f"{site_lat},{site_lon}",
                "end_coords": f"{ref_lat},{ref_lon}",
                "travel_mode": "3",
                "api_key": config.SERPAPI_KEY,
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        directions = data.get("directions") or []
        if not directions or directions[0].get("travel_mode") != "Transit":
            return None
        itinerary = directions[0]

        bus_s = train_s = walk_m = 0.0
        has_bus = has_train = False
        for trip in itinerary.get("trips", []):
            mode = trip.get("travel_mode")
            duration = trip.get("duration")
            if mode == "Walking":
                if trip.get("distance") is not None:
                    walk_m += trip["distance"]
            elif mode == "Transit" and duration is not None:
                icon = (trip.get("icon") or "").lower()
                if any(hint in icon for hint in _TRANSIT_RAIL_ICON_HINTS):
                    train_s += duration
                    has_train = True
                else:
                    bus_s += duration
                    has_bus = True

        total_duration = itinerary.get("duration")
        return {
            "bus_duration_min": round(bus_s / 60.0, 1) if has_bus else None,
            "train_duration_min": round(train_s / 60.0, 1) if has_train else None,
            "walk_km": round(walk_m / 1000.0, 2),
            "total_duration_min": round(total_duration / 60.0, 1) if total_duration else None,
            "departs_at": itinerary.get("start_time"),
            "arrives_at": itinerary.get("end_time"),
        }
    except Exception:  # noqa: BLE001 - transit lookup is best-effort; any failure just degrades to None
        return None


def price_per_acre(price: float, area: float, area_unit: str) -> float:
    """Normalize a price + area (in one of acre/hectare/sqft/sqm) to
    INR-per-acre — the unit the original notebook's scoring engine's existing
    ``land_price`` factor already expects.

    Raises:
        ValueError: If ``area_unit`` is not recognized, or ``area`` is not
            positive.
    """

    key = area_unit.strip().lower()
    if key not in _ACRES_PER_UNIT:
        raise ValueError(f"Unknown area_unit {area_unit!r}; expected one of {sorted(_ACRES_PER_UNIT)}")
    area_in_acres = area * _ACRES_PER_UNIT[key]
    if area_in_acres <= 0:
        raise ValueError("area must be positive")
    return price / area_in_acres


def area_to_sqft(area: float, area_unit: str) -> float:
    """Normalize a user-entered area (one of acre/hectare/sqft/sqm) to sqft.

    Raises:
        ValueError: If ``area_unit`` is not recognized.
    """

    key = area_unit.strip().lower()
    if key not in _ACRES_PER_UNIT:
        raise ValueError(f"Unknown area_unit {area_unit!r}; expected one of {sorted(_ACRES_PER_UNIT)}")
    return area * _ACRES_PER_UNIT[key] * _SQFT_PER_ACRE


def merge_custom_facts(
    site_summary: dict,
    *,
    reference_location: dict | None = None,
    plot_price: float | None = None,
    plot_area: float | None = None,
    area_unit: str | None = None,
) -> tuple[dict, str | None]:
    """Enrich ``site_summary`` (in place, and returned) with user-supplied
    facts, and build a small supplementary map for the reference location
    if one was given. Also attaches a best-effort public-transit itinerary
    (see :func:`fetch_transit_time`) alongside the driving time — a
    one-off scheduled snapshot, not a stable average, so it's for display
    only and never read by decision_engine.py.

    Args:
        site_summary: A ``site_summary`` dict (mutated in place). Must
            already have ``site.latitude``/``site.longitude`` (always true
            for a Stage A pipeline output).
        reference_location: ``{"lat": float, "lon": float, "label":
            str | None}``, or ``None`` to leave any existing
            ``reference_location`` section untouched.
        plot_price: Total asking/expected price for the plot, or ``None``
            to leave ``land_price`` untouched.
        plot_area: Plot area in ``area_unit``, or ``None``.
        area_unit: One of "acre", "hectare", "sqft", "sqm". Required if
            ``plot_price``/``plot_area`` are given.

    Returns:
        ``(site_summary, reference_map_html)`` — the mutated dict, and a
        self-contained HTML string for a small site/reference-location map
        (see :func:`data_analysis_pipeline.build_map.build_reference_map`),
        or ``None`` if no ``reference_location`` was given. The caller is
        responsible for writing both to disk (e.g. back over the run's
        ``site_summary.json``, and to a new ``reference_map.html``).

    Raises:
        ValueError: If ``site_summary`` has no ``site.latitude``/
            ``site.longitude``, or ``area_unit`` is invalid.
    """

    reference_map_html = None

    if reference_location is not None:
        site = site_summary.get("site") or {}
        site_lat, site_lon = site.get("latitude"), site.get("longitude")
        if site_lat is None or site_lon is None:
            raise ValueError(
                "site_summary is missing site.latitude/site.longitude; cannot compute a "
                "distance to a reference location."
            )

        ref_lat, ref_lon = reference_location["lat"], reference_location["lon"]
        ref_label = reference_location.get("label") or "Reference location"

        straight_line_km = haversine_km(site_lat, site_lon, ref_lat, ref_lon)
        travel = fetch_road_travel_time(site_lat, site_lon, ref_lat, ref_lon)
        transit = fetch_transit_time(site_lat, site_lon, ref_lat, ref_lon)

        site_summary["reference_location"] = {
            "lat": ref_lat,
            "lon": ref_lon,
            "label": ref_label,
            "straight_line_km": round(straight_line_km, 3),
            "travel_time_min": round(travel["travel_time_min"], 1) if travel else None,
            "travel_distance_km": round(travel["travel_distance_km"], 3) if travel else None,
            "transit": transit,
            "source": "user_supplied",
        }

        # Local import: build_map pulls in folium/geemap/branca, which
        # every other caller of this module (e.g. decision_engine.py) has
        # no reason to import transitively.
        from .build_map import build_reference_map

        reference_map_html = build_reference_map(
            site_lat, site_lon, ref_lat, ref_lon, ref_label,
            straight_line_km, site_summary["reference_location"]["travel_time_min"],
        )

    if plot_price is not None and plot_area is not None and area_unit is not None:
        price_acre = price_per_acre(plot_price, plot_area, area_unit)
        site_summary["land_price"] = {
            "observations": 1,
            "median_price_per_acre": round(price_acre, 2),
            "min_price_per_acre": round(price_acre, 2),
            "max_price_per_acre": round(price_acre, 2),
            "price_per_sqft": round(price_acre / _SQFT_PER_ACRE, 2),
            "total_price": round(plot_price, 2),
            "area_sqft": round(area_to_sqft(plot_area, area_unit), 2),
            "source": "User-supplied single-plot price entry",
        }

    return site_summary, reference_map_html
