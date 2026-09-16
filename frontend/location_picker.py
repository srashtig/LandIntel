"""A unified location picker: a single search box that accepts a
place/address search, a pasted Google Maps link, or raw "lat, lon" text,
plus an always-visible lat/lon readout, a Street/Satellite layer toggle, and
a pin-droppable map — all in one view rather than separate tabs.
"""

from __future__ import annotations

import re
from typing import Optional

import requests
import streamlit as st

from data_analysis_pipeline.custom_facts import haversine_km
from data_analysis_pipeline.location_search import parse_latlon_query, search_google_maps
from data_analysis_pipeline.runs import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM

MP_CENTER = (23.2599, 77.4126)  # Bhopal-ish, just a reasonable initial map center

# Esri World Imagery — a free, no-API-key satellite tile source folium can
# use directly via a plain XYZ URL template. Paired with a transparent
# "Labels" reference overlay (place names, roads, admin boundaries) so
# satellite view isn't just bare imagery — same idea as Google Maps'
# "Satellite" (hybrid) mode. The overlay is a separate togglable layer, not
# baked into the satellite tile itself, so it can be switched off too.
_SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_SATELLITE_ATTR = "Tiles &copy; Esri — Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP"
_LABELS_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
_LABELS_ATTR = "Labels &copy; Esri"

_GMAPS_SHORT_DOMAINS = ("maps.app.goo.gl", "goo.gl/maps", "g.co/kgs")

# Tried in order of precision: the !3d<lat>!4d<lon> pair embedded in a place
# URL's data= parameter is the actual pinned point (most accurate); @lat,lon
# is just the map viewport center (close, but drifts if the view was panned
# after the pin was dropped); ?q=/?ll= are older/simpler link forms.
_GMAPS_COORD_PATTERNS = [
    re.compile(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)"),
    re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)"),
    re.compile(r"[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)"),
    re.compile(r"[?&]ll=(-?\d+\.\d+),(-?\d+\.\d+)"),
]


def _looks_like_maps_url(text: str) -> bool:
    """Whether ``text`` looks like a URL (a Google Maps link, most often)
    rather than a place-name search query."""

    lowered = text.strip().lower()
    return (
        lowered.startswith("http://") or lowered.startswith("https://")
        or "google.com/maps" in lowered
        or any(domain in lowered for domain in _GMAPS_SHORT_DOMAINS)
    )


def resolve_google_maps_url(url: str, timeout: float = 10.0) -> str:
    """Follow redirects for a shortened Google Maps link so its coordinates
    can be parsed from the final, long-form URL. A long-form
    ``google.com/maps/...`` URL is returned unchanged — no network call.

    Args:
        url: A Google Maps URL, possibly shortened (``maps.app.goo.gl``,
            ``goo.gl/maps``).
        timeout: Request timeout in seconds.

    Returns:
        The final URL after following any redirects.

    Raises:
        requests.RequestException: If the shortened link can't be resolved
            (network error, timeout, etc.) — callers should catch this and
            fall back to parsing the original pasted text.
    """

    if any(domain in url for domain in _GMAPS_SHORT_DOMAINS):
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        return response.url
    return url


def parse_google_maps_url(url: str) -> Optional[tuple[float, float]]:
    """Extract ``(lat, lon)`` from a Google Maps URL pasted by the user.

    Handles both long-form URLs (``.../@23.04,76.21,15z/...`` or
    ``.../data=!3d23.04!4d76.21``) and shortened links
    (``maps.app.goo.gl/...``, ``goo.gl/maps/...``), which are resolved via
    an HTTP redirect lookup first (see :func:`resolve_google_maps_url`) —
    if that lookup fails, falls back to pattern-matching the original text
    as pasted, in case it happens to already contain a coordinate.

    Args:
        url: The pasted Google Maps URL (or any text containing one of the
            recognized coordinate patterns).

    Returns:
        ``(lat, lon)`` as floats, or None if no valid coordinate pattern
        was found (nothing matched, or a match was outside the valid
        lat/lon range).
    """

    url = url.strip()
    if not url:
        return None

    try:
        url = resolve_google_maps_url(url)
    except requests.RequestException:
        pass  # fall through and try to parse whatever was pasted as-is

    for pattern in _GMAPS_COORD_PATTERNS:
        match = pattern.search(url)
        if not match:
            continue
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    return None


def render_location_picker_with_satellite(
    key_prefix: str,
    heading: str,
    default_lat: float | None = None,
    default_lon: float | None = None,
    secondary_marker: tuple[float, float, str] | None = None,
) -> None:
    """Render one unified location picker (see module docstring).

    Reads/writes ``st.session_state[f"{key_prefix}_lat"]`` /
    ``f"{key_prefix}_lon"]``. Factored out so it can be instantiated twice
    on the same page — once for the main analysis site
    (``key_prefix="picked"``) and once for a "custom facts" reference
    location (``key_prefix="ref"``) — without the two instances' widget
    keys or session state colliding.

    Args:
        key_prefix: Session-state / widget key prefix; must be unique per
            instance rendered on the same page.
        heading: Short heading shown above the picker.
        default_lat: Map center used before anything has been picked.
        default_lon: Same, for longitude.
        secondary_marker: ``(lat, lon, label)`` for a second, fixed point
            to also show on this same map (e.g. the analyzed site, when
            this picker is choosing a "custom facts" reference location) —
            drawn with a connecting line and live straight-line distance,
            instead of building a separate route map elsewhere. ``None``
            to render a single-marker picker as usual.
    """

    if default_lat is None:
        default_lat = DEFAULT_LAT
    if default_lon is None:
        default_lon = DEFAULT_LON

    if heading:
        st.markdown(f"**{heading}**")
    lat_key, lon_key = f"{key_prefix}_lat", f"{key_prefix}_lon"
    matches_key = f"{key_prefix}_search_matches"
    if lat_key not in st.session_state:
        st.session_state[lat_key] = None
    if lon_key not in st.session_state:
        st.session_state[lon_key] = None
    if matches_key not in st.session_state:
        st.session_state[matches_key] = []

    st.markdown(
        "**Type any of these into the box below, then click Go — or skip it and just click "
        "the map to drop a pin:**"
    )
    st.caption(
        "- A place, village, or address — e.g. `Budasa, Dewas` or `Statue of Unity` "
        "(found anywhere, but only Madhya Pradesh locations can be analyzed here)\n"
        "- A Google Maps link — full (`google.com/maps/...`) or shortened (`maps.app.goo.gl/...`)\n"
        "- Coordinates directly — e.g. `23.04, 76.21`\n\n"
        "Use the layers icon on the map to switch between Street and Satellite view."
    )
    col_query, col_button = st.columns([5, 1])
    with col_query:
        query = st.text_input(
            "Search, Google Maps link, or lat, lon",
            key=f"{key_prefix}_query", label_visibility="collapsed",
            placeholder="Search, Google Maps link, or lat, lon",
        )
    with col_button:
        go_clicked = st.button("Go", key=f"{key_prefix}_query_go", use_container_width=True)

    if go_clicked and query.strip():
        st.session_state[matches_key] = []
        latlon = parse_latlon_query(query)
        if latlon is not None:
            st.session_state[lat_key], st.session_state[lon_key] = latlon
        elif _looks_like_maps_url(query):
            with st.spinner("Reading link..."):
                coords = parse_google_maps_url(query)
            if coords:
                st.session_state[lat_key], st.session_state[lon_key] = coords
            else:
                st.error(
                    "Couldn't find coordinates in that link. Try a link for a specific "
                    "dropped pin, or search by name / lat, lon instead."
                )
        else:
            with st.spinner("Searching..."):
                matches = search_google_maps(query, limit=5)
            if matches:
                st.session_state[lat_key] = matches[0]["centroid_lat"]
                st.session_state[lon_key] = matches[0]["centroid_lon"]
                st.session_state[matches_key] = matches
            else:
                st.warning("No matches found.")

    matches = st.session_state.get(matches_key) or []
    if len(matches) > 1:
        labels = [f"{m['name']}" + (f" — {m['address']}" if m.get("address") else "") for m in matches]
        with st.expander(f"{len(matches)} matches found — showing the top one; pick a different one here"):
            chosen = st.radio(
                "Matches", options=range(len(matches)), format_func=lambda i: labels[i],
                key=f"{key_prefix}_match_choice",
            )
            if st.button("Use this match instead", key=f"{key_prefix}_match_use_button"):
                st.session_state[lat_key] = matches[chosen]["centroid_lat"]
                st.session_state[lon_key] = matches[chosen]["centroid_lon"]

    lat, lon = st.session_state[lat_key], st.session_state[lon_key]
    if lat is not None:
        st.markdown(f"**Selected:** lat `{lat:.6f}`, lon `{lon:.6f}`")
    else:
        st.caption("No location selected yet — search above or click the map below.")

    try:
        from streamlit_folium import st_folium
        import folium

        if lat is not None:
            center = [lat, lon]
        elif secondary_marker is not None:
            center = [secondary_marker[0], secondary_marker[1]]
        else:
            center = [default_lat, default_lon]
        fmap = folium.Map(location=center, zoom_start=12 if (lat is not None or secondary_marker is not None) else 8, tiles=None)
        folium.TileLayer("OpenStreetMap", name="Street", overlay=False, control=True, show=True).add_to(fmap)
        folium.TileLayer(
            tiles=_SATELLITE_TILES, attr=_SATELLITE_ATTR, name="Satellite",
            overlay=False, control=True, show=False,
        ).add_to(fmap)
        folium.TileLayer(
            tiles=_LABELS_TILES, attr=_LABELS_ATTR, name="Labels (places, roads, borders)",
            overlay=True, control=True, show=False,
        ).add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)

        if lat is not None:
            folium.Marker([lat, lon], tooltip="Selected location").add_to(fmap)

        if secondary_marker is not None:
            sec_lat, sec_lon, sec_label = secondary_marker
            folium.Marker(
                [sec_lat, sec_lon], tooltip=sec_label,
                icon=folium.Icon(color="green", icon="home"),
            ).add_to(fmap)
            if lat is not None:
                distance_km = haversine_km(sec_lat, sec_lon, lat, lon)
                folium.PolyLine(
                    [[sec_lat, sec_lon], [lat, lon]], color="#d62728", weight=3, dash_array="6",
                    tooltip=f"{distance_km:.2f} km straight-line",
                ).add_to(fmap)
                fmap.fit_bounds([[sec_lat, sec_lon], [lat, lon]])

        map_result = st_folium(fmap, height=450, width=None, key=f"{key_prefix}_picker_map")
        clicked = (map_result or {}).get("last_clicked")
        if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
            new_lat, new_lon = clicked["lat"], clicked["lng"]
            if (new_lat, new_lon) != (lat, lon):
                st.session_state[lat_key] = new_lat
                st.session_state[lon_key] = new_lon
                st.rerun()
    except ImportError:
        st.error(
            "streamlit-folium is not installed in this environment. Install it with:\n"
            "conda run -n land_intel pip install streamlit-folium"
        )
