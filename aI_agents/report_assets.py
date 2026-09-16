"""Static report assets — charts + maps rendered once during the analysis
run itself (not on-demand at "Generate Report" click time), saved to
``<run_dir>/report_assets/`` alongside ``site_summary.json``/``map.html``.

Two independent branches read the same run's raw ``results`` dict
(GeoDataFrames / ``ee.Image``s) that ``build_map.py`` also consumes —
neither is derived from the other::

    Pipeline
        |
     +--+--+
     v         v
  interactive   static
  web maps      maps
  (Folium)      (PNG)
     |             |
     v             v
    UI          Report

``generate_report_assets()`` is the entry point (bound as the pipeline's
``on_assembled`` callback — see ``orchestrator.run_analysis``). Every
generator is independently fault-isolated: one failing chart/map is recorded
in the manifest's ``warnings`` and simply omitted from ``assets``, it never
blocks the others. Every chart/map gets a "Source: ..." caption baked
directly into the image (not just in surrounding report text) — no title,
since each one already sits under an HTML section header that makes it
self-explanatory.

Basemap choice varies by map (per product steer): the "water things" +
overview map use a satellite + place-labels overlay; the nearby-places and
reference-location maps use a street basemap (already has labels baked in);
NDVI/land-cover are plain (a basemap would visually clash with a full-AOI
color raster).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import branca.colormap
import contextily as cx
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyproj
import requests
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

from data_analysis_pipeline.aoi import AOI, haversine_km
from data_analysis_pipeline.get_nearby_places import PLACE_STYLES

_TO_3857 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Same Esri tile servers already used for the satellite+labels view in
# location_picker.py — kept visually consistent with the rest of the app.
_SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_LABELS_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
_SATELLITE_SOURCE_NAME = "Esri World Imagery + Esri Reference labels"
# Esri's street tiles, not OSM's own Mapnik server — confirmed by direct
# testing that contextily's default OSM.Mapnik provider gets rejected by
# OSM's tile usage policy for this kind of bulk/scripted fetching (returns
# an HTTP-200 "403 Access Blocked" notice *as the tile image itself*, which
# silently renders as if it were a real map unless you look at it). Esri's
# ArcGIS Online tiles are already used elsewhere in this app (satellite +
# labels, above) without this restriction.
_STREET_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
_STREET_SOURCE = _STREET_TILES
_STREET_SOURCE_NAME = "Esri World Street Map"

# Same visualization params as build_map.py's live GEE tile layers, so the
# static PNGs match what the interactive map shows.
_NDVI_VIS = {"min": 0.0, "max": 0.9, "palette": ["440154", "414487", "2A788E", "22A884", "7AD151", "FDE725"]}
_WORLDCOVER_CLASSES = [
    # (value, name, hex) — same order as build_map.py's landcover_vis palette
    (10, "Tree cover", "#006400"),
    (20, "Shrubland", "#ffbb22"),
    (30, "Grassland", "#ffff4c"),
    (40, "Cropland", "#f096ff"),
    (50, "Built-up", "#fa0000"),
    (60, "Bare / sparse vegetation", "#b4b4b4"),
    (70, "Snow and ice", "#f0f0f0"),
    (80, "Permanent water bodies", "#0064c8"),
    (90, "Herbaceous wetland", "#0096a0"),
    (95, "Mangroves", "#00cf75"),
    (100, "Moss and lichen", "#fae6a0"),
]
_LANDCOVER_VIS = {
    "min": 10, "max": 100,
    "palette": [c[2].lstrip("#") for c in _WORLDCOVER_CLASSES],
}


# ---------------------------------------------------------------------
# Shared map-drawing helpers
# ---------------------------------------------------------------------


def _project_bounds(aoi: AOI, buffer_factor: float = 1.3) -> tuple[float, float, float, float]:
    """EPSG:3857 (west, south, east, north) bounds around the AOI center,
    padded by ``buffer_factor`` so the AOI boundary isn't flush with the
    image edge."""

    x0, y0 = _TO_3857.transform(aoi.lon, aoi.lat)
    buf = aoi.radius_km * 1000 * buffer_factor
    return x0 - buf, y0 - buf, x0 + buf, y0 + buf


def _fetch_satellite_labels_basemap(aoi: AOI):
    """Fetch the satellite + labels basemap tiles once for this AOI's
    extent — reused as the background for every "water things" + overview
    map so three maps don't each trigger their own tile fetch."""

    w, s, e, n = _project_bounds(aoi)
    sat_img, sat_ext = cx.bounds2img(w, s, e, n, source=_SATELLITE_TILES, ll=False)
    labels_img, labels_ext = cx.bounds2img(w, s, e, n, source=_LABELS_TILES, ll=False)
    return (sat_img, sat_ext), (labels_img, labels_ext)


def _fetch_street_basemap(aoi: AOI, buffer_factor: float = 2.2):
    """Fetch the street basemap once — reused for the site-overview and
    nearby-places maps (Esri street tiles already bake in place labels), and
    as a fallback basemap for the water/groundwater maps if the satellite
    fetch fails. Fetched at ``buffer_factor=2.2`` by default (wide enough to
    cover the nearby-places search radius, itself wider than the AOI) — a
    narrower map (e.g. the site overview, at the default 1.3x AOI framing)
    just crops into this same already-fetched image via a tighter
    ``set_xlim``/``set_ylim``, no separate fetch needed."""

    w, s, e, n = _project_bounds(aoi, buffer_factor=buffer_factor)
    return cx.bounds2img(w, s, e, n, source=_STREET_SOURCE, ll=False)


def _new_map_fig(figsize=(5.2, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _draw_water_basemap(ax, basemap) -> None:
    """Draw the base layer (+ optional labels overlay) for the groundwater/
    water-bodies maps — accepts either the satellite+labels pair from
    :func:`_fetch_satellite_labels_basemap`, or a plain street basemap
    passed as a same-shaped ``(base, None)`` fallback (used when the
    satellite fetch itself fails — a real street map beats no map)."""

    base, overlay = basemap
    ax.imshow(base[0], extent=base[1])
    if overlay is not None:
        ax.imshow(overlay[0], extent=overlay[1], alpha=0.85)


def _finish_map(fig, ax, source: str, out_path: Path) -> None:
    """No title baked into the image — every map/chart already sits under
    an HTML section header (e.g. "GROUNDWATER") that makes it self-
    explanatory, so a second, redundant title inside the image itself was
    just wasted vertical space. Only the "Source: ..." caption stays, since
    that information isn't otherwise stated anywhere nearby."""

    fig.text(0.5, 0.01, f"Source: {source}", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _draw_aoi_boundary(ax, aoi: AOI, color: str = "#facc15") -> None:
    coords = [_TO_3857.transform(lon, lat) for lon, lat in aoi.polygon_4326.exterior.coords]
    xs, ys = zip(*coords)
    ax.plot(xs, ys, color=color, linewidth=2, linestyle="--", zorder=4)


def _draw_site_marker(ax, aoi: AOI, label: str = "Site") -> None:
    x0, y0 = _TO_3857.transform(aoi.lon, aoi.lat)
    ax.scatter([x0], [y0], color="#ef4444", s=140, marker="*", edgecolor="white", linewidth=1, zorder=6)
    ax.annotate(label, (x0, y0), textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold", color="#ef4444")


def _set_aoi_extent(ax, aoi: AOI, buffer_factor: float = 1.3) -> None:
    w, s, e, n = _project_bounds(aoi, buffer_factor=buffer_factor)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)


# ---------------------------------------------------------------------
# Maps — satellite + labels ("water things" + overview)
# ---------------------------------------------------------------------


def _site_context_map(aoi: AOI, street_basemap, out_path: Path) -> None:
    img, ext = street_basemap
    fig, ax = _new_map_fig()
    ax.imshow(img, extent=ext)
    _draw_aoi_boundary(ax, aoi, color="#1f2937")
    _draw_site_marker(ax, aoi)
    _set_aoi_extent(ax, aoi, buffer_factor=2.0)  # zoomed out further so street/place names are actually legible
    _finish_map(fig, ax, _STREET_SOURCE_NAME, out_path)


def _well_2025_median(well_seasonal: dict, well_no) -> float | None:
    """Median of whatever 2025 pre-monsoon/monsoon readings exist for one
    well — the plain number requested in place of a depth-to-color
    encoding."""

    year_data = (well_seasonal or {}).get(well_no, {}).get(2025, {})
    levels = [
        (year_data.get(season) or {}).get("level_m_bgl")
        for season in ("pre_monsoon", "monsoon")
    ]
    levels = [v for v in levels if v is not None]
    if not levels:
        return None
    levels.sort()
    mid = len(levels) // 2
    return levels[mid] if len(levels) % 2 else (levels[mid - 1] + levels[mid]) / 2


def _groundwater_map(aoi: AOI, results: dict, basemap, out_path: Path) -> bool:
    gw_obs = results.get("groundwater", {}).get("observations", {})
    wells = gw_obs.get("latest_wells_gdf")
    if wells is None or wells.empty:
        return False
    well_seasonal = gw_obs.get("well_seasonal") or {}

    fig, ax = _new_map_fig()
    _draw_water_basemap(ax, basemap)
    _draw_aoi_boundary(ax, aoi)
    _draw_site_marker(ax, aoi)

    wells_4326 = wells.to_crs("EPSG:4326")
    for _, row in wells_4326.iterrows():
        if row.geometry is None:
            continue
        x, y = _TO_3857.transform(row.geometry.x, row.geometry.y)
        ax.scatter([x], [y], color="#2563eb", s=70, edgecolor="white", linewidth=0.8, zorder=5)
        well_no = row.get("Well No")
        median_2025 = _well_2025_median(well_seasonal, well_no)
        label_lines = [str(well_no)] if well_no is not None else []
        label_lines.append(f"{median_2025:.1f} m (2025)" if median_2025 is not None else "2025: n/a")
        ax.annotate(
            "\n".join(label_lines), (x, y), textcoords="offset points", xytext=(6, 4), fontsize=6.5,
            color="#1e3a8a",
        )

    _set_aoi_extent(ax, aoi)
    _finish_map(fig, ax, "CGWB manual monthly groundwater monitoring", out_path)
    return True


def _water_bodies_map(aoi: AOI, results: dict, basemap, out_path: Path) -> bool:
    hydro_obs = results.get("hydrology", {}).get("observations", {})
    osm_obs = results.get("osm", {}).get("observations", {})
    water_bodies = hydro_obs.get("water_bodies_gdf")
    rivers = hydro_obs.get("river_polygons_gdf")
    osm_water = (osm_obs.get("water") or {}).get("gdf")

    has_content = any(gdf is not None and not gdf.empty for gdf in (water_bodies, rivers, osm_water))
    if not has_content:
        return False

    fig, ax = _new_map_fig()
    _draw_water_basemap(ax, basemap)
    _draw_aoi_boundary(ax, aoi)
    _draw_site_marker(ax, aoi)

    def _plot_shapes(gdf, name_col, color, label_max=6):
        """Draw the actual feature geometry (like the interactive Folium
        map does — polygons/lines, not just centroid dots), and label only
        the features that actually have a name (skip "Unnamed" clutter)."""

        if gdf is None or gdf.empty:
            return
        g = gdf[gdf.geometry.notna()].to_crs("EPSG:3857")
        g.plot(ax=ax, color=color, alpha=0.55, edgecolor=color, linewidth=1.2, zorder=4)

        named = g[g[name_col].notna() & (g[name_col].astype(str).str.strip() != "")] if name_col in g.columns else g.iloc[0:0]
        if named.empty:
            return
        g_utm = gdf.loc[named.index].to_crs(aoi.utm_crs)
        centroids_utm = g_utm.geometry.centroid
        dists = centroids_utm.distance(aoi.center_utm)
        order = sorted(range(len(g_utm)), key=lambda i: dists.iloc[i])[:label_max]
        centroids_4326 = gpd.GeoSeries(centroids_utm, crs=aoi.utm_crs).to_crs("EPSG:4326")
        for i in order:
            pt = centroids_4326.iloc[i]
            x, y = _TO_3857.transform(pt.x, pt.y)
            name = named.iloc[i][name_col]
            ax.annotate(str(name), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=7.5, color=color, fontweight="bold")

    _plot_shapes(water_bodies, "wetname", "#0ea5e9")
    _plot_shapes(rivers, "rivname", "#1d4ed8")
    _plot_shapes(osm_water, "name", "#22d3ee")

    _set_aoi_extent(ax, aoi)
    _finish_map(fig, ax, "SAC water bodies + river floodplain + OpenStreetMap", out_path)
    return True


# ---------------------------------------------------------------------
# Maps — street basemap (nearby places, reference location)
# ---------------------------------------------------------------------


def _reference_location_map(aoi: AOI, site_summary: dict, out_path: Path) -> bool:
    ref = site_summary.get("reference_location")
    if not ref or ref.get("lat") is None:
        return False

    ref_lat, ref_lon = ref["lat"], ref["lon"]
    # Frame around the midpoint of site + reference so both fit.
    mid_lat, mid_lon = (aoi.lat + ref_lat) / 2, (aoi.lon + ref_lon) / 2
    span_km = max(aoi.radius_km, haversine_km(aoi.lat, aoi.lon, ref_lat, ref_lon) / 1.6)
    fake_aoi_center = _TO_3857.transform(mid_lon, mid_lat)
    buf = span_km * 1000 * 1.3
    w, s, e, n = fake_aoi_center[0] - buf, fake_aoi_center[1] - buf, fake_aoi_center[0] + buf, fake_aoi_center[1] + buf

    img, ext = cx.bounds2img(w, s, e, n, source=_STREET_SOURCE, ll=False)
    fig, ax = _new_map_fig()
    ax.imshow(img, extent=ext)

    x0, y0 = _TO_3857.transform(aoi.lon, aoi.lat)
    x1, y1 = _TO_3857.transform(ref_lon, ref_lat)
    ax.plot([x0, x1], [y0, y1], color="#7c3aed", linewidth=1.5, linestyle=":", zorder=4)
    ax.scatter([x0], [y0], color="#ef4444", s=140, marker="*", edgecolor="white", linewidth=1, zorder=6)
    ax.annotate("Site", (x0, y0), textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold", color="#ef4444")
    ax.scatter([x1], [y1], color="#7c3aed", s=100, marker="D", edgecolor="white", linewidth=1, zorder=6)
    ax.annotate(ref.get("label") or "Reference", (x1, y1), textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold", color="#7c3aed")

    dist_note = f"{ref.get('straight_line_km', 0):.1f} km straight-line"
    if ref.get("travel_distance_km") is not None:
        dist_note += f" · {ref.get('travel_distance_km'):.1f} km by road"
    ax.text(0.5, -0.06, dist_note, transform=ax.transAxes, ha="center", fontsize=8.5)

    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    _finish_map(fig, ax, _STREET_SOURCE_NAME, out_path)
    return True


def _nearby_places_map(aoi: AOI, results: dict, basemap, out_path: Path) -> bool:
    serp_places = results.get("nearby_places", {}).get("observations", {}).get("serp_places") or {}
    has_content = any(places for places in serp_places.values())
    if not has_content:
        return False

    img, ext = basemap
    fig, ax = _new_map_fig()
    ax.imshow(img, extent=ext)
    _draw_aoi_boundary(ax, aoi, color="#1f2937")
    _draw_site_marker(ax, aoi)

    used_categories = []
    for category, places in serp_places.items():
        style = PLACE_STYLES.get(category, {"color": "black"})
        nearest = sorted(places, key=lambda p: p.get("distance_km", 1e9))[:5]
        if not nearest:
            continue
        used_categories.append((category, style["color"]))
        for place in nearest:
            if place.get("latitude") is None:
                continue
            x, y = _TO_3857.transform(place["longitude"], place["latitude"])
            ax.scatter([x], [y], color=style["color"], s=45, edgecolor="white", linewidth=0.5, zorder=5)

    for category, color in used_categories:
        ax.scatter([], [], color=color, label=category.replace("_", " ").title())
    if used_categories:
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    w, s, e, n = _project_bounds(aoi, buffer_factor=2.2)  # nearby-places search radius (10km) is wider than the AOI
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    _finish_map(fig, ax, f"{_STREET_SOURCE_NAME} + SerpApi / Google Maps", out_path)
    return True


# ---------------------------------------------------------------------
# Maps — plain GEE raster thumbnails (NDVI, land cover)
# ---------------------------------------------------------------------


def _ee_region(aoi: AOI):
    import ee

    return ee.Geometry.Polygon([list(coord) for coord in aoi.polygon_4326.exterior.coords])


def _fetch_ee_thumbnail(image, vis: dict, region, dimensions: int = 480) -> PILImage.Image:
    url = image.visualize(**vis).getThumbURL({"region": region, "dimensions": dimensions, "format": "png"})
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return PILImage.open(io.BytesIO(response.content)).convert("RGBA")


def _draw_site_star_on_thumb(ax, thumb: PILImage.Image) -> None:
    """The AOI is a circle centered on the site, and the thumbnail's region
    is exactly that circle's bounding box, so the image's pixel center is
    the site location — no coordinate transform needed here."""

    cx, cy = thumb.width / 2, thumb.height / 2
    ax.scatter([cx], [cy], color="#ef4444", s=130, marker="*", edgecolor="white", linewidth=1, zorder=6)
    ax.annotate("Site", (cx, cy), textcoords="offset points", xytext=(7, 7), fontsize=8.5, fontweight="bold", color="#ef4444")


def _ndvi_map(aoi: AOI, results: dict, out_path: Path) -> bool:
    ndvi_image = results.get("sentinel", {}).get("observations", {}).get("ndvi_image")
    if ndvi_image is None:
        return False

    thumb = _fetch_ee_thumbnail(ndvi_image, _NDVI_VIS, _ee_region(aoi))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(thumb)
    _draw_site_star_on_thumb(ax, thumb)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cmap = LinearSegmentedColormap.from_list("ndvi", [f"#{c}" for c in _NDVI_VIS["palette"]])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=_NDVI_VIS["min"], vmax=_NDVI_VIS["max"]))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI", fontsize=8)

    _finish_map(fig, ax, "Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)", out_path)
    return True


def _land_cover_map(aoi: AOI, results: dict, site_summary: dict, out_path: Path) -> bool:
    worldcover_image = results.get("land_cover", {}).get("observations", {}).get("worldcover_image")
    if worldcover_image is None:
        return False

    thumb = _fetch_ee_thumbnail(worldcover_image, _LANDCOVER_VIS, _ee_region(aoi))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(thumb)
    _draw_site_star_on_thumb(ax, thumb)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    present_names = {c.get("name") for c in (site_summary.get("land_cover", {}).get("classes") or [])}
    for _value, name, hexcolor in _WORLDCOVER_CLASSES:
        if name in present_names:
            ax.scatter([], [], color=hexcolor, marker="s", s=50, label=name)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7, title="Land cover", title_fontsize=7.5)

    _finish_map(fig, ax, "ESA WorldCover v200", out_path)
    return True


# ---------------------------------------------------------------------
# Charts (plain matplotlib, no basemap — migrated from build_report.py,
# now writing to a PNG file instead of an in-memory base64 tag)
# ---------------------------------------------------------------------


def _groundwater_chart(site_summary: dict, out_path: Path) -> bool:
    seasonal = site_summary.get("groundwater", {}).get("seasonal") or []
    years, pre, mon = [], [], []
    for record in seasonal:
        pre_v = (record.get("pre_monsoon") or {}).get("level_m_bgl")
        mon_v = (record.get("monsoon") or {}).get("level_m_bgl")
        if pre_v is None and mon_v is None:
            continue
        years.append(str(record.get("year")))
        pre.append(pre_v)
        mon.append(mon_v)
    if not years:
        return False

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = range(len(years))
    width = 0.35
    ax.bar([i - width / 2 for i in x], [v if v is not None else 0 for v in pre], width, label="Pre-monsoon", color="#f59e0b")
    ax.bar([i + width / 2 for i in x], [v if v is not None else 0 for v in mon], width, label="Monsoon", color="#3b82f6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(years)
    ax.set_ylabel("Depth (m below ground)")
    ax.invert_yaxis()  # deeper (worse) reads visually "lower"
    ax.legend(fontsize=8)
    fig.text(0.5, 0.01, "Source: CGWB manual monthly groundwater monitoring", ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def _slope_class_chart(site_summary: dict, out_path: Path) -> bool:
    shares = site_summary.get("terrain", {}).get("slope_class_share") or []
    if not shares:
        return False

    labels = [s.get("label") for s in shares]
    values = [s.get("share_percent") or 0 for s in shares]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(labels, values, color="#65a30d")
    ax.set_ylabel("Share of AOI (%)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.text(0.5, 0.01, "Source: USGS SRTM 1 arc-second (SRTMGL1_003)", ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------
# Small utility + entrypoint
# ---------------------------------------------------------------------


def generate_report_assets(aoi: AOI, results: dict, site_summary: dict, assets_dir: Path) -> dict:
    """Generate every static report chart/map for this run and write
    ``assets_dir / "manifest.json"``. Bound as the pipeline orchestrator's
    ``on_assembled`` callback (see ``run_worker.py``), so this runs once,
    during the analysis run itself — the report generator later just reads
    the manifest, never guessing what exists or regenerating anything live.

    Each generator is independently try/except-isolated: one failing
    chart/map is recorded in ``manifest["warnings"]`` and simply omitted
    from ``manifest["assets"]``, never blocking the others.

    Returns:
        The manifest dict (also written to disk).
    """

    assets_dir.mkdir(parents=True, exist_ok=True)
    run_dir = assets_dir.parent
    assets: dict[str, str] = {}
    warnings: list[str] = []

    def _rel(filename: str) -> str:
        return str((assets_dir / filename).relative_to(run_dir))

    def _try(key: str, filename: str, fn, *args) -> None:
        path = assets_dir / filename
        try:
            ok = fn(*args, path)
        except Exception as error:  # noqa: BLE001 - one bad asset must never block the rest
            warnings.append(f"{key}: {error}")
            return
        if ok is False:
            warnings.append(f"{key}: no data available for this run")
            return
        assets[key] = _rel(filename)

    # Fetch each basemap once, reused across the maps that share it.
    street_basemap = None
    try:
        street_basemap = _fetch_street_basemap(aoi)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"street basemap fetch failed, overview/nearby-places maps skipped: {error}")

    sat_labels_basemap = None
    try:
        sat_labels_basemap = _fetch_satellite_labels_basemap(aoi)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"satellite basemap fetch failed, falling back to street for water maps: {error}")

    # Water/groundwater maps prefer satellite+labels, but a real street map
    # beats no map at all if the satellite fetch specifically fails.
    water_basemap = sat_labels_basemap if sat_labels_basemap is not None else (
        (street_basemap, None) if street_basemap is not None else None
    )

    if street_basemap is not None:
        _try("site_context_map", "site_context_map.png", lambda p: _site_context_map(aoi, street_basemap, p))
        _try("nearby_places_map", "nearby_places_map.png", lambda p: _nearby_places_map(aoi, results, street_basemap, p))

    if water_basemap is not None:
        _try("groundwater_map", "groundwater_map.png", lambda p: _groundwater_map(aoi, results, water_basemap, p))
        _try("water_bodies_map", "water_bodies_map.png", lambda p: _water_bodies_map(aoi, results, water_basemap, p))

    _try("reference_location_map", "reference_location_map.png", lambda p: _reference_location_map(aoi, site_summary, p))
    _try("ndvi_map", "ndvi_map.png", lambda p: _ndvi_map(aoi, results, p))
    _try("land_cover_map", "land_cover_map.png", lambda p: _land_cover_map(aoi, results, site_summary, p))

    _try("groundwater_chart", "groundwater_chart.png", lambda p: _groundwater_chart(site_summary, p))
    _try("slope_class_chart", "slope_class_chart.png", lambda p: _slope_class_chart(site_summary, p))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
        "warnings": warnings,
    }
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
