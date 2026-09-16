"""Folium map + HTML summary-panel injection.

Ports the notebook's map-building cells (27-53: base map, roads, rail,
water/SAC/river/village layers, groundwater markers, land-price points,
terrain/NDVI/land-cover/LST tile layers via geemap, nearby-places markers,
layer control) and the summary-panel cell (63), adapted to read the
polished ``site_summary`` schema (see :mod:`assemble_summary`) for panel
numbers and the raw ``results`` envelopes (see :mod:`orchestrator`) for
map geometries/images.
"""

from __future__ import annotations

import html as html_lib

import branca
import folium
import geopandas as gpd
import pandas as pd

from .get_nearby_places import PLACE_STYLES, place_popup

WORLDCOVER_COLORS = {
    "Tree cover": "#006400", "Shrubland": "#ffbb22", "Grassland": "#ffff4c",
    "Cropland": "#f096ff", "Built-up": "#fa0000", "Bare / sparse vegetation": "#b4b4b4",
    "Snow / ice": "#f0f0f0", "Permanent water": "#0064c8", "Herbaceous wetland": "#0096a0",
    "Mangroves": "#00cf75", "Moss / lichen": "#fae6a0",
}
SLOPE_COLORS = {
    "< 3° (nearly level)": "#86b6ef", "3–8° (gentle)": "#3987e5",
    "8–15° (moderate)": "#1c5cab", ">= 15° (steep)": "#0d366b",
}

PANEL_CSS = """
<style>
.li-summary {
  --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #898781; --gridline: #e1e0d9; --border: rgba(11, 11, 11, 0.10);
  --track: #f0efec;
  position: fixed; top: 12px; left: 60px; z-index: 9999; width: 384px;
  max-height: calc(100vh - 40px); display: flex; flex-direction: column;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 10px;
  box-shadow: 0 6px 24px rgba(11, 11, 11, 0.14);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 12.5px; line-height: 1.45;
}
.li-summary__head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; padding: 10px 12px; border-bottom: 1px solid var(--gridline); }
.li-summary__title { font-size: 13px; font-weight: 650; letter-spacing: 0.01em; }
.li-summary__place { color: var(--text-secondary); font-size: 11.5px; }
.li-summary__toggle { appearance: none; border: 1px solid var(--border); background: var(--surface-1); color: var(--text-secondary); border-radius: 6px; padding: 2px 8px; font: inherit; font-size: 11px; cursor: pointer; }
.li-summary__toggle:hover { background: var(--track); }
.li-summary__body { overflow-y: auto; padding: 4px 12px 12px; }
.li-summary.is-collapsed .li-summary__body { display: none; }
.li-summary.is-collapsed { width: 268px; }
.li-hero { padding: 10px 0 4px; border-bottom: 1px solid var(--gridline); }
.li-hero__value { font-size: 27px; font-weight: 620; letter-spacing: -0.01em; }
.li-hero__label { color: var(--text-secondary); font-size: 11.5px; }
.li-sec { margin-top: 14px; }
.li-sec__title { font-size: 10.5px; font-weight: 650; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-muted); padding-bottom: 5px; border-bottom: 1px solid var(--gridline); }
.li-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 3px 0; }
.li-row__label { color: var(--text-secondary); }
.li-row__value { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.li-row__note { color: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.li-meter { padding: 4px 0 5px; }
.li-meter__head { display: flex; justify-content: space-between; gap: 8px; }
.li-meter__value { font-variant-numeric: tabular-nums; }
.li-meter__track { height: 7px; margin-top: 3px; border-radius: 4px; background: var(--track); overflow: hidden; }
.li-meter__fill { height: 100%; border-radius: 0 4px 4px 0; box-shadow: inset 0 0 0 0.5px rgba(11, 11, 11, 0.12); }
.li-table { width: 100%; border-collapse: collapse; margin-top: 2px; }
.li-table th, .li-table td { text-align: right; padding: 3px 0; font-variant-numeric: tabular-nums; }
.li-table th { color: var(--text-muted); font-weight: 500; font-size: 11px; border-bottom: 1px solid var(--gridline); }
.li-table th:first-child, .li-table td:first-child { text-align: left; color: var(--text-secondary); }
.li-table tr.is-total td { border-top: 1px solid var(--gridline); font-weight: 600; }
.li-foot { margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--gridline); color: var(--text-muted); font-size: 10.5px; }
</style>
"""


def _esc(value) -> str:
    """HTML-escape a value for panel display, rendering None as "n/a"."""

    return html_lib.escape("n/a" if value is None else str(value))


def _section(title: str, inner: str) -> str:
    """Wrap ``inner`` HTML in one titled summary-panel section."""

    return f'<div class="li-sec"><div class="li-sec__title">{_esc(title)}</div>{inner}</div>'


def _row(label, value, note=None) -> str:
    """Render one label/value (optionally with a small note) panel row."""

    note_html = f'<div class="li-row__note">{_esc(note)}</div>' if note else ""
    return f'<div class="li-row"><div class="li-row__label">{_esc(label)}</div><div class="li-row__value">{_esc(value)}{note_html}</div></div>'


def _meter(label, percentage, color, value_text=None) -> str:
    """Render one labeled horizontal bar (e.g. a land-cover class share).

    Args:
        label: Row label.
        percentage: Value on a 0-100 scale; used for the bar width.
        color: CSS color for the filled portion.
        value_text: Text shown instead of the default "{pct:.1f}%".
    """

    pct = 0.0 if percentage is None or pd.isna(percentage) else float(percentage)
    width = max(min(pct, 100.0), 0.6)
    return (
        '<div class="li-meter"><div class="li-meter__head">'
        f'<div class="li-row__label">{_esc(label)}</div>'
        f'<div class="li-meter__value">{_esc(value_text or f"{pct:.1f}%")}</div></div>'
        f'<div class="li-meter__track"><div class="li-meter__fill" style="width:{width:.2f}%;background:{color};"></div></div></div>'
    )


def _fmt(value, spec="{:.1f}", suffix="", missing="n/a") -> str:
    """Format a possibly-None/NaN numeric value, e.g. _fmt(3.14, "{:.0f}", " km")."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return missing
    return spec.format(value) + suffix


def _fmt_distance(metres) -> str:
    """Format a distance in meters as "N m" (<1km) or "N.NN km" (>=1km)."""

    if metres is None or (isinstance(metres, float) and pd.isna(metres)):
        return "n/a"
    return f"{metres:.0f} m" if metres < 1000 else f"{metres / 1000:.2f} km"


_GUIDELINE_RATE_DISPLAY = [
    ("Residential plot", "plot_residential_sqm", "sqm"),
    ("Commercial plot", "plot_commercial_sqm", "sqm"),
    ("Industrial plot", "plot_industrial_sqm", "sqm"),
    ("Irrigated agri land", "agri_land_irrigated_per_ha", "ha"),
    ("Unirrigated agri land", "agri_land_unirrigated_per_ha", "ha"),
]


def _guideline_rates_html(rates: dict | None) -> str:
    """Render ``administration.govt_guideline_rates`` (see
    :func:`data_analysis_pipeline.get_admin_context._lookup_govt_guideline_rates`)
    as summary-panel rows. Always shows ``row_kind`` -- whether this is a
    specific matched village's rate, or a (district, tehsil) range for an
    urban area with no ward-boundary shapefile to resolve any finer."""

    if not rates:
        return _row("Govt guideline rate", "not available for this location")

    row_kind = rates.get("row_kind")
    rows = [_row("Row kind", row_kind)]
    if row_kind == "rural_village":
        rows.append(_row(
            "Matched village", rates.get("village"),
            f"vlcode {rates['vlcode']:.0f}" if rates.get("vlcode") is not None else None,
        ))
        rows.append(_row("Tehsil", rates.get("tehsil")))
        rows.append(_row(
            "Frontage / match confidence", rates.get("frontage_type"),
            _fmt(rates.get("match_score"), "{:.0f}", "% name-match score"),
        ))
        for label, key, unit in _GUIDELINE_RATE_DISPLAY:
            value = rates.get(key)
            if value is not None:
                rows.append(_row(label, f"₹{value:,.0f} / {unit}"))
    elif row_kind == "urban_range":
        rows.append(_row("Tehsil", rates.get("tehsil")))
        rows.append(_row(
            "Wards / rows summarized",
            f"{_fmt(rates.get('n_wards'), '{:.0f}')} / {_fmt(rates.get('n_rows'), '{:.0f}')}",
        ))
        for label, key, unit in _GUIDELINE_RATE_DISPLAY:
            lo, hi = rates.get(f"{key}_min"), rates.get(f"{key}_max")
            if lo is not None or hi is not None:
                rows.append(_row(label, f"₹{_fmt(lo, '{:,.0f}')} – ₹{_fmt(hi, '{:,.0f}')} / {unit}"))
    rows.append(_row("Source", rates.get("source")))
    return "".join(rows)


def _seasonal_table_html(seasonal_records: list) -> str:
    """Render a year x pre-monsoon/monsoon groundwater-level HTML table.

    Args:
        seasonal_records: A list as produced in ``groundwater.seasonal``
            (see :mod:`assemble_summary`'s schema doc) — each entry a dict
            with "year", "pre_monsoon", "monsoon", "observed_months",
            "fluctuation_m".

    Returns:
        An HTML ``<table>`` plus a small footnote, as a string.
    """

    def cell(entry):
        if entry is None:
            return '<span style="color:var(--text-muted);">–</span>'
        return f'{entry["level_m_bgl"]:.2f}<div class="li-row__note">{_esc("/".join(entry["months"]))}</div>'

    rows = []
    observed_parts = []
    for record in seasonal_records:
        rows.append(
            "<tr>"
            f"<td>{record['year']}</td>"
            f"<td>{cell(record['pre_monsoon'])}</td>"
            f"<td>{cell(record['monsoon'])}</td>"
            f"<td>{_esc(_fmt(record['fluctuation_m'], '{:+.2f}', '', '–'))}</td>"
            "</tr>"
        )
        observed_parts.append(f"{record['year']} {', '.join(record['observed_months']) or 'none'}")

    return (
        '<table class="li-table"><tr><th>Year</th><th>Pre-<br>monsoon</th>'
        "<th>Monsoon</th><th>Change</th></tr>"
        f"{''.join(rows)}</table>"
        '<div class="li-foot" style="margin-top:6px;border:0;padding:0;">'
        "m bgl · change = pre − monsoon (positive = monsoon recharge)"
        f"<br>Months observed: {_esc('; '.join(observed_parts))}"
        "</div>"
    )


def build_reference_map(
    site_lat: float,
    site_lon: float,
    ref_lat: float,
    ref_lon: float,
    ref_label: str,
    straight_line_km: float,
    travel_time_min: float | None = None,
) -> str:
    """Build a small, self-contained map showing the analyzed site, a
    user-supplied reference location, and a line between them labeled with
    distance (and road travel time, if available).

    Built from scratch rather than added to the main run's ``map.html``:
    that map's live ``folium.Map`` object isn't persisted after a run
    (only its rendered HTML is), so a reference location added after the
    fact — see :func:`data_analysis_pipeline.custom_facts.merge_custom_facts`
    — can't be injected into it without rerunning the full pipeline. This
    map only needs the two points, so it costs nothing to build fresh.

    Args:
        site_lat: Analyzed site latitude.
        site_lon: Analyzed site longitude.
        ref_lat: Reference location latitude.
        ref_lon: Reference location longitude.
        ref_label: Display label for the reference location.
        straight_line_km: Great-circle distance, for the line's tooltip.
        travel_time_min: Road travel time in minutes, or ``None`` if
            routing was unavailable — shown alongside the distance when
            present, with a plain note otherwise.

    Returns:
        A self-contained HTML string.
    """

    time_note = (
        f" · {travel_time_min:.0f} min by road" if travel_time_min is not None
        else " · road travel time unavailable"
    )
    tooltip = f"{straight_line_km:.2f} km straight-line{time_note}"

    m = folium.Map(location=[(site_lat + ref_lat) / 2, (site_lon + ref_lon) / 2], tiles="OpenStreetMap")
    folium.Marker(
        [site_lat, site_lon], tooltip="Analyzed site",
        icon=folium.Icon(icon="map-marker", color="blue"),
    ).add_to(m)
    folium.Marker(
        [ref_lat, ref_lon], tooltip=html_lib.escape(ref_label),
        icon=folium.Icon(icon="star", color="red"),
    ).add_to(m)
    folium.PolyLine(
        [[site_lat, site_lon], [ref_lat, ref_lon]], color="#7f1d1d", weight=3, tooltip=tooltip,
    ).add_to(m)
    m.fit_bounds([[site_lat, site_lon], [ref_lat, ref_lon]], padding=(40, 40))

    return m.get_root().render()


def build_map(aoi, results: dict, site_summary: dict):
    """Build the Folium map (roads/rail/water/hydrology/groundwater/terrain/
    NDVI/land-cover/heat/nearby-places layers, plus a collapsible HTML
    summary panel) for one analyzed site.

    A failed upstream ``get_*`` module never aborts the map build: that
    layer is simply omitted (or, for the summary panel, replaced with a
    short "unavailable" note) — see the try/except around each layer below.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        results: Dict of every ``get_*`` module's envelope (keyed by module
            name), from :func:`orchestrator.run_analysis` — supplies the raw
            geometries/images each layer is drawn from.
        site_summary: The assembled, JSON-safe dict from
            :func:`assemble_summary.assemble_summary` — supplies the numbers
            shown in the summary panel.

    Returns:
        ``(folium.Map, html_str)`` — the live Folium map object and its
        fully rendered, self-contained HTML (suitable for ``.write()`` to a
        file or embedding via ``st.components.v1.html``).
    """

    # Every "observations" access below tolerates a module that failed
    # upstream (orchestrator._safe_call replaces it with an envelope whose
    # observations == {}): each layer is built from an empty GeoDataFrame /
    # dict fallback rather than raising, and every layer-building block
    # further down is individually try/except-guarded so one bad layer
    # never blocks the rest of the map.
    _empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    def _obs(module_key: str) -> dict:
        return results.get(module_key, {}).get("observations", {})

    osm, hydro = _obs("osm"), _obs("hydrology")
    admin_obs, gw_obs = _obs("admin"), _obs("groundwater")
    terrain_obs = _obs("terrain")
    sentinel_obs, land_cover_obs = _obs("sentinel"), _obs("land_cover")
    lst_obs, nearby_obs = _obs("lst"), _obs("nearby_places")

    roads = osm.get("roads", {}).get("gdf", _empty_gdf)
    railways = osm.get("railways", {}).get("gdf", _empty_gdf)
    water = osm.get("water", {}).get("gdf", _empty_gdf)
    river_polygons = hydro.get("river_polygons_gdf", _empty_gdf)
    water_bodies_in_aoi = hydro.get("water_bodies_gdf", _empty_gdf)
    villages_in_aoi_gdf = admin_obs.get("villages_in_aoi_gdf", _empty_gdf)
    latest_wells = gw_obs.get("latest_wells_gdf", _empty_gdf)
    well_seasonal = gw_obs.get("well_seasonal", {})
    serp_places = nearby_obs.get("serp_places", {})

    lat, lon, radius_km = aoi.lat, aoi.lon, aoi.radius_km

    # --- 10A. Base map ---------------------------------------------------
    # Two selectable BASE layers (mutually exclusive, radio-style in the
    # layer control): classic OpenStreetMap street tiles (default), and
    # Esri World Imagery satellite. Plus two independent OVERLAY layers —
    # transparent roads and transparent place/locality labels — that can be
    # switched on together with EITHER base layer, so turning both on over
    # Satellite gives a hybrid "satellite + roads + names" view (Esri's
    # free reference tile services; all free, no API key required).
    m = folium.Map(location=[lat, lon], zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Street", overlay=False, control=True, show=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Roads (overlay)",
        overlay=True,
        control=True,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Place names (overlay)",
        overlay=True,
        control=True,
        show=False,
    ).add_to(m)
    folium.Marker([lat, lon], tooltip="Selected location", popup=f"{lat:.7f}, {lon:.7f}", icon=folium.Icon(icon="map-marker")).add_to(m)
    folium.GeoJson(aoi.aoi_gdf.to_json(), name=f"{radius_km} km AOI", style_function=lambda x: {"fill": False, "weight": 2}).add_to(m)

    # --- 10B. Roads --------------------------------------------------------
    # Every block below is independently try/except-guarded: a failed
    # upstream module (empty gdf) or unexpectedly-shaped data in one layer
    # must never abort the rest of the map build.
    try:
        if not roads.empty:
            road_fg = folium.FeatureGroup(name="Roads", show=True)
            folium.GeoJson(
                roads.to_json(), style_function=lambda f: {"color": "#444444", "weight": 2},
                tooltip=folium.GeoJsonTooltip(fields=["highway"], aliases=["Type"], localize=True, sticky=False),
            ).add_to(road_fg)
            road_fg.add_to(m)
    except Exception as error:
        print("Roads layer could not be added:", error)

    # --- 10C. Railways -------------------------------------------------
    try:
        if not railways.empty:
            rail_fg = folium.FeatureGroup(name="Railways", show=False)
            tooltip_fields = [f for f in ["name"] if f in railways.columns]
            folium.GeoJson(
                railways.to_json(), style_function=lambda f: {"color": "#7b1fa2", "weight": 3},
                tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=["Name"], localize=True) if tooltip_fields else None,
            ).add_to(rail_fg)
            rail_fg.add_to(m)
    except Exception as error:
        print("Railways layer could not be added:", error)

    # --- 10D. Water (OSM + SAC) -----------------------------------------
    try:
        if (not water.empty) or (not water_bodies_in_aoi.empty):
            water_fg = folium.FeatureGroup(name="Water", show=False)
            if not water.empty:
                folium.GeoJson(
                    water.to_json(), name="OSM water",
                    style_function=lambda f: {"color": "#1976d2", "weight": 2, "fillColor": "#64b5f6", "fillOpacity": 0.35},
                ).add_to(water_fg)
            if not water_bodies_in_aoi.empty:
                folium.GeoJson(
                    water_bodies_in_aoi.to_json(), name="SAC water bodies",
                    style_function=lambda f: {"color": "#005f73", "weight": 1.5, "fillColor": "#00b4d8", "fillOpacity": 0.45},
                    tooltip=folium.GeoJsonTooltip(
                        fields=["wetname", "level_iii", "area_ha"], aliases=["Name", "Type", "Area (ha)"],
                        localize=True, sticky=False,
                    ),
                ).add_to(water_fg)
            water_fg.add_to(m)
    except Exception as error:
        print("Water layer could not be added:", error)

    # --- 10D-1. River floodplain proxy + village boundaries -------------
    try:
        if not river_polygons.empty:
            river_fg = folium.FeatureGroup(name="River polygons / floodplain proxy", show=False)
            folium.GeoJson(
                river_polygons.to_json(),
                style_function=lambda f: {"color": "#7f1d1d", "weight": 1.5, "fillColor": "#ef4444", "fillOpacity": 0.28},
                tooltip=folium.GeoJsonTooltip(fields=["rivname", "ripcode"], aliases=["River", "River code"], localize=True, sticky=False),
            ).add_to(river_fg)
            river_floodplain = site_summary.get("hydrology", {}).get("river_floodplain", {})
            folium.Marker(
                [lat, lon],
                tooltip=(
                    "Floodplain proxy: inside supplied river polygon"
                    if river_floodplain.get("inside_supplied_river_polygon")
                    else f"Nearest supplied river polygon: {river_floodplain['nearest_river_m']:.0f} m"
                    if river_floodplain.get("nearest_river_m") is not None else "No river polygons in query window"
                ),
            ).add_to(river_fg)
            river_fg.add_to(m)
    except Exception as error:
        print("River floodplain layer could not be added:", error)

    try:
        if not villages_in_aoi_gdf.empty:
            village_fg = folium.FeatureGroup(name="Village boundaries", show=False)
            # Village name plus each village's OWN guideline rate (not just
            # the host site's) -- district/tehsil are dropped here since
            # every village in a several-km AOI already shares the same
            # one, making them redundant on a per-village hover tooltip.
            tooltip_field_aliases = [
                ("village", "Village"),
                ("guideline_plot_residential_sqm", "Residential rate (Rs/sqm)"),
                ("guideline_agri_land_irrigated_per_ha", "Irrigated agri rate (Rs/ha)"),
                ("guideline_agri_land_unirrigated_per_ha", "Unirrigated agri rate (Rs/ha)"),
            ]
            tooltip_fields = [f for f, _ in tooltip_field_aliases if f in villages_in_aoi_gdf.columns]
            tooltip_aliases = [a for f, a in tooltip_field_aliases if f in villages_in_aoi_gdf.columns]
            folium.GeoJson(
                villages_in_aoi_gdf.to_json(),
                style_function=lambda f: {"color": "#6b21a8", "weight": 1, "fillColor": "#c084fc", "fillOpacity": 0.08},
                tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True, sticky=False) if tooltip_fields else None,
            ).add_to(village_fg)
            village_fg.add_to(m)
    except Exception as error:
        print("Village boundaries layer could not be added:", error)

    # Always-visible village NAME labels — a separate overlay from "Village
    # boundaries" above (which only shows names on hover). Deliberately uses
    # our own Survey-of-India village data rather than the Esri "Place names
    # (overlay)" layer added earlier: Esri's is a global reference dataset
    # with sparse/unreliable coverage of small rural villages in India,
    # while this is the same precise per-village dataset used for admin
    # lookups elsewhere in the pipeline.
    try:
        if not villages_in_aoi_gdf.empty and "village" in villages_in_aoi_gdf.columns:
            label_fg = folium.FeatureGroup(name="Village names (overlay)", show=False)
            for _, row in villages_in_aoi_gdf.iterrows():
                name = row.get("village")
                geom = row.geometry
                if not name or geom is None or geom.is_empty:
                    continue
                centroid = geom.centroid
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    icon=folium.DivIcon(html=(
                        '<div style="font-size:10px;font-weight:600;color:#4c1d95;'
                        'text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;'
                        f'white-space:nowrap;pointer-events:none;">{html_lib.escape(str(name))}</div>'
                    )),
                ).add_to(label_fg)
            label_fg.add_to(m)
    except Exception as error:
        print("Village name labels layer could not be added:", error)

    # --- 10E. Groundwater markers -----------------------------------------
    try:
        if not latest_wells.empty:
            vmin, vmax = latest_wells["latest_level"].min(), latest_wells["latest_level"].max()
            colormap = branca.colormap.LinearColormap(colors=["red", "yellow", "green"], vmin=vmin, vmax=vmax, caption="Latest groundwater level (m bgl)")
            gw_fg = folium.FeatureGroup(name="Groundwater", show=False)

            for _, row in latest_wells.iterrows():
                level = row["latest_level"]
                well_no = row["Well No"]
                seasonal = well_seasonal.get(well_no, {})
                seasonal_records = [
                    {"year": y, "pre_monsoon": seasonal.get(y, {}).get("pre_monsoon"), "monsoon": seasonal.get(y, {}).get("monsoon"),
                     "observed_months": seasonal.get(y, {}).get("observed_months", []),
                     "fluctuation_m": None}
                    for y in sorted(seasonal.keys())
                ]
                depth = row.get("Depth of Well")
                depth_text = f"{depth:.2f} m" if pd.notna(depth) else "N/A"
                popup = f"""
                <div style="font-size:13px;line-height:1.5;width:290px;">
                    <h4 style="margin-bottom:8px;">Groundwater Well: {well_no}</h4>
                    <b>Well type:</b> {row.get('Well Type', 'N/A')}<br>
                    <b>Well depth:</b> {depth_text}<br>
                    <b>Agency:</b> {row.get('Agency', 'N/A')}<br>
                    <hr>
                    <b>Latest observation:</b> {row['latest_date'].strftime('%d-%m-%Y')}<br>
                    <b>Latest water level:</b> <span style="font-weight:bold;">{level:.2f} m bgl</span>
                    <hr>
                    <b>5-year average:</b> {_fmt(row.get('five_year_avg'), '{:.2f}', ' m bgl')}<br>
                    <b>Observations:</b> {int(row['n_5yr']) if pd.notna(row.get('n_5yr')) else 'n/a'}<br>
                    <hr>
                    <b>Pre / post-monsoon levels</b><br>
                    {_seasonal_table_html(seasonal_records)}
                    <hr>
                    <b>5-year trend:</b> <span style="font-weight:bold;">{row.get('trend_symbol', '')} {row.get('trend', 'n/a')}</span><br>
                    <small>{row.get('trend_description', '')} ({_fmt(row.get('change_5yr'), '{:+.2f}', ' m')})</small>
                    <hr>
                    <b>Village:</b> {row.get('Village', 'N/A')}<br>
                    <b>District:</b> {row.get('District', 'N/A')}<br>
                    <hr>
                    <b>Latitude:</b> {row.geometry.y:.6f}<br>
                    <b>Longitude:</b> {row.geometry.x:.6f}
                </div>
                """
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], radius=6,
                    tooltip=f"Groundwater: {level:.1f} m bgl", popup=popup,
                    fill=True, fill_opacity=0.8, color=colormap(level),
                ).add_to(gw_fg)
            gw_fg.add_to(m)
    except Exception as error:
        print("Groundwater layer could not be added:", error)

    # --- Terrain tile layers ---------------------------------------------
    try:
        elevation_image, slope_image = terrain_obs["elevation_image"], terrain_obs["slope_image"]
        elevation_vis = {"min": 0, "max": 1000, "palette": ["006400", "7FFF00", "FFFF00", "FFA500", "A52A2A"]}
        slope_vis = {"min": 0, "max": 30, "palette": ["006400", "FFFF00", "FF8C00", "FF0000"]}
        folium.TileLayer(tiles=elevation_image.getMapId(elevation_vis)["tile_fetcher"].url_format, attr="SRTM", name="Terrain - Elevation", overlay=True, control=True, show=False).add_to(m)
        folium.TileLayer(tiles=slope_image.getMapId(slope_vis)["tile_fetcher"].url_format, attr="SRTM", name="Terrain - Slope", overlay=True, control=True, show=False).add_to(m)
    except Exception as error:
        print("Terrain tile layers could not be added:", error)

    # --- Earth Engine raster layers via geemap --------------------------
    import os
    os.environ["USE_FOLIUM"] = "1"
    import geemap.foliumap as geemap

    try:
        ndvi_vis = {"min": 0.0, "max": 0.9, "palette": ["440154", "414487", "2A788E", "22A884", "7AD151", "FDE725"]}
        geemap.ee_tile_layer(sentinel_obs["ndvi_image"], ndvi_vis, "NDVI", shown=False).add_to(m)
    except Exception as error:
        print("NDVI layer could not be added:", error)

    try:
        landcover_vis = {"min": 10, "max": 100, "palette": ["006400", "ffbb22", "ffff4c", "f096ff", "fa0000", "b4b4b4", "f0f0f0", "0064c8", "0096a0", "00cf75", "fae6a0"]}
        geemap.ee_tile_layer(land_cover_obs["worldcover_image"], landcover_vis, "Land Cover", shown=False).add_to(m)
    except Exception as error:
        print("Land Cover layer could not be added:", error)

    try:
        lst_vis = {"min": 20, "max": 50, "palette": ["313695", "74add1", "abd9e9", "fee090", "f46d43", "a50026"]}
        geemap.ee_tile_layer(lst_obs["lst_image"], lst_vis, "Heat / Land Surface Temperature", shown=False).add_to(m)
    except Exception as error:
        print("Heat layer could not be added:", error)

    # --- Nearby places markers -------------------------------------------
    try:
        if any(serp_places.values()):
            nearby_fg = folium.FeatureGroup(name="Nearby places (10 km)", show=False)
            for category, items in serp_places.items():
                style = PLACE_STYLES.get(category, {"icon": "info-sign", "color": "gray"})
                for place in items:
                    folium.Marker(
                        [place["latitude"], place["longitude"]],
                        icon=folium.Icon(icon=style["icon"], prefix="fa", color=style["color"]),
                        tooltip=f"{place['name']} - {category.replace('_', ' ')} - {place['distance_km']:.2f} km",
                        popup=folium.Popup(place_popup(place, category), max_width=320),
                    ).add_to(nearby_fg)
            nearby_fg.add_to(m)
    except Exception as error:
        print("Nearby places layer could not be added:", error)

    # --- Summary panel -----------------------------------------------
    try:
        panel_html = _build_panel_html(site_summary)
    except Exception as error:
        print("Summary panel could not be built:", error)
        panel_html = PANEL_CSS + (
            '<div class="li-summary" id="li-summary"><div class="li-summary__body">'
            f'<div class="li-foot">Summary panel unavailable: {_esc(error)}</div></div></div>'
        )
    m.get_root().html.add_child(folium.Element(panel_html))

    folium.LayerControl(collapsed=False, position="topright").add_to(m)

    html_str = m.get_root().render()
    return m, html_str


def _build_panel_html(s: dict) -> str:
    """Render the collapsible site-summary HTML panel injected into the map.

    Args:
        s: The assembled ``site_summary`` dict (see :mod:`assemble_summary`'s
            schema doc). Raises (caught by :func:`build_map`'s caller, which
            substitutes a short "unavailable" panel) if a section a
            failed upstream module was supposed to fill is accessed here.

    Returns:
        A self-contained HTML string (``<style>`` + the panel markup).
    """

    admin = s["administration"]
    land_use = admin.get("land_use") or {}
    census = admin.get("census") or {}
    land_cover = s["land_cover"]
    groundwater = s["groundwater"]
    rainfall = s["rainfall"]
    air_temp = s["air_temperature"]
    lst = s["land_surface_temperature"]
    terrain = s["terrain"]
    roads = s["roads"]
    rail = s["railways"]
    water = s["water_resources"]
    settlements = s["settlements"]
    nearby_places = s["nearby_places"]
    ndvi = s["ndvi"]
    land_price = s.get("land_price") or {}
    radius_km = s["site"]["aoi_radius_km"]
    aoi_area_km2 = s["site"]["aoi_area_km2"]
    lat, lon = s["site"]["latitude"], s["site"]["longitude"]

    place = " · ".join(p for p in [admin.get("village"), admin.get("tehsil"), admin.get("district"), admin.get("state")] if p) or "Location"

    chirps_means = rainfall["means"]["chirps"]
    blocks = [
        '<div class="li-hero"><div class="li-hero__value">'
        f'{_fmt(chirps_means.get("annual_mm"), "{:.0f}", " mm")}</div>'
        f'<div class="li-hero__label">mean annual rainfall (CHIRPS v3) · '
        f'groundwater {_fmt(groundwater.get("latest_mean_depth_m_bgl"), "{:.1f}", " m bgl")}</div></div>'
    ]

    blocks.append(_section("Location & administration", "".join([
        _row("Coordinates", f"{lat:.6f}, {lon:.6f}"),
        _row("Village", admin.get("village")),
        _row("Tehsil / sub-district", admin.get("tehsil")),
        _row("Block", admin.get("block")),
        _row("District", admin.get("district")),
        _row("State", admin.get("state")),
        _row("AOI", f"{radius_km} km radius", f"{aoi_area_km2:.1f} km²"),
        _row("Villages in AOI", admin["aoi_overlap"]["village_count"] or "n/a"),
    ])))

    blocks.append(_section("Govt guideline land rate (FY2026-27)", _guideline_rates_html(admin.get("govt_guideline_rates"))))

    classes = land_cover["classes"]
    top_classes, other_pct = classes[:5], sum(c["percentage"] for c in classes[5:])
    landcover_html = "".join(
        _meter(c["name"], c["percentage"], WORLDCOVER_COLORS.get(c["name"], "#898781"), f'{c["percentage"]:.1f}%  ({c["area_km2"]:.2f} km²)')
        for c in top_classes
    )
    if other_pct > 0:
        landcover_html += _meter("Other classes", other_pct, "#898781")
    blocks.append(_section("Land cover — ESA WorldCover v200 (10 m)", landcover_html))

    years = rainfall["years"]
    rain_rows = "".join(
        "<tr>"
        f"<td>{y['year']}</td>"
        f"<td>{_esc(_fmt((y['chirps'] or {}).get('annual_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt((y['imd'] or {}).get('annual_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt((y['chirps'] or {}).get('monsoon_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt((y['chirps'] or {}).get('rain_days'), '{:.0f}'))}</td>"
        "</tr>"
        for y in rainfall["yearly"]
    )
    imd_means = rainfall["means"]["imd"]
    rain_rows += (
        '<tr class="is-total"><td>Mean</td>'
        f"<td>{_esc(_fmt(chirps_means.get('annual_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt(imd_means.get('annual_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt(chirps_means.get('monsoon_mm'), '{:.0f}'))}</td>"
        f"<td>{_esc(_fmt(chirps_means.get('rain_days'), '{:.0f}'))}</td></tr>"
    )
    blocks.append(_section(
        f"Rainfall {min(years)}–{max(years)}",
        f'<table class="li-table"><tr><th>Year</th><th>CHIRPS<br>mm</th><th>IMD<br>mm</th><th>Monsoon<br>mm</th><th>Rain<br>days</th></tr>{rain_rows}</table>'
        + _row("Days >= 25 / 50 mm (mean)", f"{_fmt(chirps_means.get('days_ge_25mm'), '{:.1f}')} / {_fmt(chirps_means.get('days_ge_50mm'), '{:.1f}')}")
        + _row("Wettest day in period", _fmt(rainfall["wettest_day_mm"].get("chirps"), "{:.0f}", " mm"))
        + _row("Monsoon share of annual", _fmt(100 * chirps_means["monsoon_mm"] / chirps_means["annual_mm"], "{:.0f}", "%") if chirps_means.get("annual_mm") else "n/a"),
    ))

    if groundwater.get("wells"):
        groundwater_html = "".join([
            _row("Latest mean depth", _fmt(groundwater["latest_mean_depth_m_bgl"], "{:.2f}", " m bgl"), f"as of {groundwater['latest_observation']}"),
            _row("Latest depth range", f"{groundwater['latest_min_depth_m_bgl']:.2f} – {groundwater['latest_max_depth_m_bgl']:.2f} m bgl"),
            _row("5-year mean depth", _fmt(groundwater["five_year_mean_m_bgl"], "{:.2f}", " m bgl")),
            _seasonal_table_html(groundwater["seasonal"]),
            _row("5-year trend", groundwater.get("dominant_trend", "n/a"), f"mean change {_fmt(groundwater.get('mean_change_5yr_m'), '{:+.2f}', ' m')}"),
            _row("Monitoring wells", groundwater["wells"], f"nearest {_fmt_distance(groundwater['nearest_well_m'])}"),
        ])
    else:
        groundwater_html = _row("Monitoring wells in AOI", "none")
    blocks.append(_section("Groundwater — CGWB", groundwater_html))

    blocks.append(_section("Temperature", "".join([
        _row(f"Mean air temp {air_temp.get('period', '')}", _fmt(air_temp.get("annual_mean_c"), "{:.1f}", " °C"), "ERA5-Land 2 m"),
        _row("Warmest month", _fmt(air_temp.get("warmest_month_c"), "{:.1f}", " °C"), air_temp.get("warmest_month")),
        _row("Coldest month", _fmt(air_temp.get("coldest_month_c"), "{:.1f}", " °C"), air_temp.get("coldest_month")),
        _row("Land-surface temp (median)", _fmt(lst.get("mean_c"), "{:.1f}", " °C"), f"range {_fmt(lst.get('min_c'), '{:.0f}')}–{_fmt(lst.get('max_c'), '{:.0f}')} °C"),
    ])))

    slope_html = "".join([
        _row("Elevation", f"{_fmt(terrain['elevation_min_m'], '{:.0f}')}–{_fmt(terrain['elevation_max_m'], '{:.0f}', ' m')}", f"mean {_fmt(terrain['elevation_mean_m'], '{:.0f}', ' m')}"),
        _row("Mean / max slope", f"{_fmt(terrain['slope_mean_deg'], '{:.1f}')}° / {_fmt(terrain['slope_max_deg'], '{:.1f}')}°"),
    ]) + "".join(
        _meter(c["label"], c["share_percent"], SLOPE_COLORS.get(c["label"], "#2a78d6"))
        for c in terrain["slope_class_share"]
    )
    blocks.append(_section("Terrain — SRTM 30 m", slope_html))

    major_road_note = roads.get("nearest_major_road_class")
    if major_road_note and roads.get("nearest_major_road_beyond_aoi"):
        major_road_note += " (beyond AOI)"
    major_road_row = _row("Nearest major road", _fmt_distance(roads.get("nearest_major_road_m")), major_road_note or "none found nearby")

    if roads.get("features"):
        access_html = "".join([
            _row("Nearest road", _fmt_distance(roads["nearest_road_m"]), ", ".join(p for p in [roads.get("nearest_road_class"), roads.get("nearest_road_name")] if p)),
            major_road_row,
            _row("Road length in AOI", _fmt(roads["total_length_km"], "{:.1f}", " km"), f"{_fmt(roads['density_km_per_km2'], '{:.2f}')} km/km²"),
        ] + [
            _row(f"— {c['highway_class']}", _fmt(c["length_km"], "{:.1f}", " km"))
            for c in roads.get("length_km_by_class", [])[:4]
        ])
    else:
        access_html = _row("Roads in AOI", "none") + major_road_row
    access_html += (
        _row("Nearest railway", _fmt_distance(rail["nearest_rail_m"]), rail.get("nearest_rail_name") or "unnamed")
        if rail.get("features") else _row("Railways in AOI", "none")
    )
    blocks.append(_section("Roads & rail — OpenStreetMap", access_html))

    if water.get("features"):
        water_html = "".join([
            _row("Nearest water feature", _fmt_distance(water["nearest_water_m"]), ", ".join(p for p in [water.get("nearest_water_type"), water.get("nearest_water_name")] if p)),
            _row("Water bodies / waterways", f"{water['waterbodies']} / {water['waterways']}"),
            _row("Water-body area", _fmt(water["waterbody_area_km2"], "{:.2f}", " km²"), f"{_fmt(water.get('waterbody_share_of_aoi_percent'), '{:.1f}', '%')} of AOI"),
            _row("Waterway length", _fmt(water["waterway_length_km"], "{:.1f}", " km")),
        ])
    else:
        water_html = _row("Mapped water features", "none")
    water_html += _row("WorldCover permanent water", _fmt(water.get("worldcover_permanent_water_percent"), "{:.2f}", "%"))
    blocks.append(_section("Water resources", water_html))

    village_html = [
        _row("Population", _fmt(census.get("population_total"), "{:.0f}")),
        _row("Literacy", _fmt(census.get("literacy_percent"), "{:.1f}", "%")),
        _row("SC + ST", _fmt(sum(v for v in [census.get("sc_percent"), census.get("st_percent")] if v is not None), "{:.1f}", "%")),
        _row("Forest share", _fmt(land_use.get("forest_share_percent"), "{:.1f}", "%")),
        _row("Barren share", _fmt(land_use.get("barren_share_percent"), "{:.1f}", "%")),
        _row("Net sown share", _fmt(land_use.get("net_sown_share_percent"), "{:.1f}", "%")),
    ]
    blocks.append(_section("Village census & land use", "".join(village_html)))

    nearby_html = [
        _row("Settlements in 10 km", settlements["settlement_count_10km"]),
        _row("Nearest settlement", settlements.get("nearest_settlement_name") or "none",
             _fmt_distance((settlements.get("distance_to_nearest_settlement_km") or 0) * 1000) if settlements.get("distance_to_nearest_settlement_km") is not None else "n/a"),
    ]
    for cat in nearby_places["categories"]:
        nearest = cat.get("nearest") or {}
        rating_note = None
        if nearest.get("rating") is not None:
            try:
                rating_note = f"{float(nearest['rating']):g}/5 stars"
                if nearest.get("reviews") is not None:
                    rating_note += f" ({int(str(nearest['reviews']).replace(',', '')):,})"
            except (TypeError, ValueError):
                rating_note = None
        distance = nearest.get("distance_km")
        nearby_html.append(
            _row(f"Nearest {cat['category'].replace('_', ' ').title()}", nearest.get("name") or "none",
                 (_fmt_distance(distance * 1000) if distance is not None else "n/a") + (f" · {rating_note}" if rating_note else ""))
        )
    blocks.append(_section("Settlements & nearby places — 10 km", "".join(nearby_html)))

    ndvi_peak = ndvi.get("peak")
    blocks.append(_section("Vegetation — NDVI (Sentinel-2)", "".join([
        _row("Whole-year NDVI (AOI mean)", _fmt(ndvi["mean"], "{:.2f}"), f"range {_fmt(ndvi['min'], '{:.2f}')}–{_fmt(ndvi['max'], '{:.2f}')}"),
        _row("Area NDVI >= 0.30", _fmt(ndvi.get("share_ge_030_percent"), "{:.1f}", "%")),
        _row("Area NDVI >= 0.50", _fmt(ndvi.get("share_ge_050_percent"), "{:.1f}", "%")),
        _row("Whole-year period", ndvi["period"]),
        _row("Peak-season NDVI (AOI mean)", _fmt(ndvi_peak.get("mean"), "{:.2f}") if ndvi_peak else "n/a",
             "greenest period found in the year" if ndvi_peak else "no usable scenes in any period"),
        _row("Peak period", ndvi_peak["period"] if ndvi_peak else "n/a"),
    ])))

    if land_price.get("observations"):
        blocks.append(_section("Land price — observed asking prices", "".join([
            _row("Observations in AOI", land_price["observations"]),
            _row("Median", f"₹{land_price['median_price_per_acre']:,.0f} / acre"),
            _row("Range", f"₹{land_price['min_price_per_acre']:,.0f} – ₹{land_price['max_price_per_acre']:,.0f}"),
        ])))

    blocks.append(
        '<div class="li-foot">'
        f'Generated {_esc(s["generated_at"])} · CHIRPS v3 (AOI mean) &amp; IMD 0.25° (nearest grid cell) · '
        'ESA WorldCover v200 · Sentinel-2 · Landsat 8 C2 L2 · ERA5-Land · SRTM 30 m · OpenStreetMap · '
        'CGWB · Survey of India village boundaries.'
        f'<br>Lengths, areas and distances computed in {_esc(s["site"]["metric_crs"])}.'
        '</div>'
    )

    return (
        PANEL_CSS
        + '<div class="li-summary" id="li-summary"><div class="li-summary__head"><div>'
        '<div class="li-summary__title">Site summary</div>'
        f'<div class="li-summary__place">{_esc(place)}</div></div>'
        '<button class="li-summary__toggle" type="button" '
        "onclick=\"var p=document.getElementById('li-summary');p.classList.toggle('is-collapsed');"
        "this.textContent=p.classList.contains('is-collapsed')?'Show':'Hide';\">Hide</button></div>"
        f'<div class="li-summary__body">{"".join(blocks)}</div></div>'
    )
