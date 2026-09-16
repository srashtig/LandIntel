"""Chapter-wise land intelligence report — deterministic-first (per product
steer: ~75% of the report's content is computed directly from
``site_summary.json`` + the pre-rendered assets in ``report_assets/``; the
remaining ~25% is exactly ONE consolidated Groq call producing a short
interpretation per chapter plus the dedicated strengths/concerns/trade-offs
chapter — see :func:`aI_agents.qa.generate_report_interpretations`).

Every chart/map is generated once, during the analysis run itself (see
``aI_agents.report_assets``), not on-demand here — this module just assembles
HTML from ``site_summary.json`` + ``report_assets/manifest.json``, so
"Generate Report" is fast and never makes a live GEE/OSM/basemap call.

The same HTML serves both the on-screen popup and the PDF export
(:func:`report_to_pdf_bytes`, via ``xhtml2pdf``) — no iframes/live maps (they
don't survive HTML->PDF conversion; that's exactly why every map here is a
pre-rendered PNG, not a Folium embed), no CSS custom properties/fixed
positioning (``xhtml2pdf`` doesn't support them).

Chapters:
    Cover — Site / Location / Purpose / Area-price / overview map.
    1. Location & Administration
    2. Water & Hydrology
    3. Climate & Land Use
    4. Human / Locality Context
    5. Overall Summary & Suitability for <purpose>  (fully AI)
    6. Data Sources & Limitations  (fully deterministic, kept short)

Chapters 1-4 each end with a short "AI Interpretation" box; chapter 5 IS the
interpretation (no separate box needed there).
"""

from __future__ import annotations

import base64
import functools
import html as html_lib
import io
import json
from pathlib import Path

from .qa import ChatUnavailable, _settlement_context, generate_report_interpretations

_ICON_PATH = Path(__file__).resolve().parents[1] / "docs" / "icon_title.png"

# ---------------------------------------------------------------------
# Look & feel — plain, portable CSS (no custom properties, no fixed
# positioning, no iframes) so the exact same HTML renders correctly both as
# an on-screen popup (st.components.v1.html) and through xhtml2pdf for the
# PDF download.
# ---------------------------------------------------------------------

_REPORT_CSS = """
<style>
@page {
    size: A4;
    margin: 2cm 1.5cm 2cm 1.5cm;
}
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2320;
       background: #f7f6f3; margin: 0; padding: 0; }
.li-report { max-width: 880px; margin: 0 auto; padding: 20px 20px 32px; }
.li-hero { background: #1e3a2f; color: #ffffff; border-radius: 12px; padding: 16px 22px; margin-bottom: 14px; }
.li-hero p { margin: 0; color: #cfe3d7; font-size: 13px; line-height: 1.7; }
.li-hero .title-line { font-size: 20px; font-weight: 700; color: #ffffff; }
/* xhtml2pdf supports neither `float` nor flexbox/grid for the logo-on-the-
   right layout (confirmed directly — `float` was silently ignored, logo
   rendered inline instead) — a table is the one layout mechanism it
   reliably supports for this. The table needs its OWN background color,
   not just the wrapping .li-hero div's: confirmed directly that xhtml2pdf
   does not paint a parent div's background behind a child table, leaving
   the hero looking blank/backgroundless without this. */
table.li-hero-table { width: 100%; background: #1e3a2f; }
table.li-hero-table td { vertical-align: middle; padding: 0; background: #1e3a2f; }
td.li-hero-logo-cell { width: 150px; text-align: right; }
img.li-logo { height: 56px; }
.li-card { background: #ffffff; border: 1px solid #e5e3dd; border-radius: 12px; padding: 14px 18px;
           margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); page-break-before: always; }
.li-card h2 { margin: 0 0 10px; font-size: 16px; color: #1e3a2f; border-bottom: 2px solid #2f855a; padding-bottom: 6px; }
.li-card h3 { margin: 10px 0 4px; font-size: 12.5px; color: #52514e; text-transform: uppercase;
              letter-spacing: 0.04em; }
.li-fact-table { width: 100%; border-spacing: 0; margin: 0; }
.li-fact-table td { padding: 3px 6px; border-bottom: 1px solid #f0efec; font-size: 13.5px; }
.li-fact-table td.label { color: #52514e; width: 55%; }
.li-fact-table td.value { font-weight: 600; text-align: right; }
.li-fact-table .note { color: #898781; font-weight: 400; font-size: 11.5px; display: block; }
.li-ai-box { background: #f0f7f2; border-left: 3px solid #2f855a; border-radius: 6px;
             padding: 8px 14px; margin-top: 8px; font-size: 13px; line-height: 1.5; }
.li-ai-box .tag { font-weight: 700; color: #276749; font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.04em; display: block; margin-bottom: 3px; }
.li-table { width: 100%; border-collapse: collapse; border-spacing: 0; margin-top: 4px; font-size: 13px; }
.li-table th { text-align: left; color: #898781; font-weight: 500; font-size: 11px;
               border-bottom: 1px solid #e1e0d9; padding: 3px 6px; }
.li-table td { padding: 3px 6px; border-bottom: 1px solid #f0efec; }
.li-bullet-table { width: 100%; border-spacing: 0; margin: 4px 0; }
.li-bullet-table td { padding: 2px 0 2px 14px; font-size: 13px; }
.li-overall { background: #f0f7f2; border: 1px solid #bfe3cc; border-radius: 10px; padding: 12px 16px;
              font-size: 14px; line-height: 1.5; }
img.li-chart { width: 480px; margin: 4px auto; display: block; border-radius: 8px; }
.li-note { color: #898781; font-size: 11px; margin-top: 6px; }
.li-footer { text-align: center; color: #898781; font-size: 11px; margin-top: 24px; }
</style>
"""


_TOFU_GLYPH_FIXES = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen — both confirmed via direct PDF
    # inspection to render as a solid tofu box in xhtml2pdf's base font,
    # unlike en/em dashes, curly quotes, or the degree sign, which render
    # fine — occasionally emitted by the AI-generated report text.
}


def _esc(value) -> str:
    text = "n/a" if value is None else str(value)
    for bad, good in _TOFU_GLYPH_FIXES.items():
        text = text.replace(bad, good)
    return html_lib.escape(text)


def _fmt_num(value, decimals: int = 1, unit: str = "") -> str | None:
    """Round a float to a sane number of decimals — fixes numbers like
    ``65.51724137931035`` showing up verbatim."""

    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}{unit}"
    return f"{value}{unit}"


def _card_open(title: str) -> str:
    return f'<div class="li-card"><h2>{_esc(title)}</h2>'


_CARD_CLOSE = "</div>"


def _fact(label: str, value, note: str | None = None) -> str:
    """A single label/value row, as its own tiny 2-column table — xhtml2pdf
    doesn't support flexbox (confirmed by direct inspection of a rendered
    PDF: a flex-based row silently stacks label/value into two separate
    full-width blocks instead of one aligned row), but it handles real
    ``<table>`` elements correctly, so every fact row is one."""

    note_html = f'<span class="note">{_esc(note)}</span>' if note else ""
    return (
        '<table class="li-fact-table"><tr><td class="label">' + _esc(label) + "</td>"
        '<td class="value">' + _esc("n/a" if value is None else value) + note_html + "</td></tr></table>"
    )


def _table(headers: list[str], rows: list[tuple]) -> str | None:
    if not rows:
        return None
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table class="li-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _bullets(items: list[str], marker: str = "•") -> str:
    """A bulleted list as a one-column table (not ``<ul><li>``, which shows
    the same broken-box artifacts as the old flex-based fact rows once
    rendered through xhtml2pdf) — ``marker`` is a plain-text character, not
    an emoji, since xhtml2pdf's base font can't render color emoji glyphs
    (they show as solid tofu boxes)."""

    if not items:
        return ""
    rows = "".join(f"<tr><td>{marker} {_esc(item)}</td></tr>" for item in items)
    return f'<table class="li-bullet-table">{rows}</table>'


def _ai_box(text: str | None) -> str:
    if not text:
        return (
            '<div class="li-ai-box"><span class="tag">AI interpretation</span>'
            "Not available for this chapter (Q&A unavailable or nothing returned).</div>"
        )
    return f'<div class="li-ai-box"><span class="tag">AI interpretation</span>{_esc(text)}</div>'


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _month_name(yyyy_mm: str | None) -> str | None:
    """"2024-05" -> "May" — spelled out, per product steer, instead of the
    raw "YYYY-MM" the underlying ERA5/Landsat data uses."""

    if not yyyy_mm or "-" not in str(yyyy_mm):
        return yyyy_mm
    try:
        month_num = int(str(yyyy_mm).split("-")[1])
        return _MONTH_NAMES[month_num - 1]
    except (ValueError, IndexError):
        return yyyy_mm


def _img_tag(manifest: dict, run_dir: Path, key: str, alt: str) -> str | None:
    """Embed a pre-rendered PNG (from ``report_assets/``, indexed by the
    manifest) as a base64 ``<img>`` tag — reads bytes off disk instead of
    generating a live figure, and keeps the PDF export fully self-contained
    (no external file references)."""

    rel_path = (manifest.get("assets") or {}).get(key)
    if not rel_path:
        return None
    file_path = run_dir / rel_path
    if not file_path.exists():
        return None
    b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f'<img class="li-chart" src="data:image/png;base64,{b64}" alt="{_esc(alt)}">'


_HERO_BACKGROUND_RGB = (0x1E, 0x3A, 0x2F)  # matches .li-hero's background: #1e3a2f


@functools.lru_cache(maxsize=1)
def _logo_img_tag() -> str:
    """Embed the app's static ``docs/icon_title.png`` logo (icon + "LANDINTEL"
    wordmark lockup) as a base64 ``<img>`` tag — unlike :func:`_img_tag`,
    this isn't a per-run generated asset, so it's read once and cached for
    the life of the process rather than looked up per report. Returns ``""``
    if the file is missing, so a caller can just concatenate it into the
    HTML unconditionally.

    ``docs/icon_title.png`` itself is a fully opaque RGB image (a white
    card behind the logo, not a transparent background) — the flatten step
    below is a no-op for it today, but is kept so a future transparent
    variant of this asset would still render correctly against the hero's
    dark green (``_HERO_BACKGROUND_RGB``) instead of showing a white box:
    confirmed directly that xhtml2pdf does not composite PNG alpha against
    the page background on its own.
    """

    if not _ICON_PATH.exists():
        return ""
    from PIL import Image

    icon = Image.open(_ICON_PATH).convert("RGBA")
    background = Image.new("RGBA", icon.size, (*_HERO_BACKGROUND_RGB, 255))
    flattened = Image.alpha_composite(background, icon).convert("RGB")
    buffer = io.BytesIO()
    flattened.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<img class="li-logo" src="data:image/png;base64,{b64}" alt="LandIntel logo">'


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "report_assets" / "manifest.json"
    if not manifest_path.exists():
        return {"assets": {}, "warnings": ["report_assets/manifest.json not found for this run — re-run analysis to generate maps/charts."]}
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"assets": {}, "warnings": ["report_assets/manifest.json could not be read."]}


# ---------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------


def _cover(site_summary: dict, manifest: dict, run_dir: Path) -> str:
    site = site_summary.get("site", {})
    admin = site_summary.get("administration", {})
    purpose = site_summary.get("buying_purpose")
    preferences = site_summary.get("buying_preferences")
    land_price = site_summary.get("land_price")

    # Everything below is ONE <p> with <br> line breaks, not several separate
    # <p> tags — xhtml2pdf renders each block-level element boundary inside
    # a colored background as its own banded strip with a thin divider line
    # (confirmed by direct inspection: title+tagline showed this exact
    # artifact as two separate <p>s, and disappeared once merged into one
    # with a <br>), so keeping this whole hero as a single block is what
    # makes the PDF match the on-screen HTML's clean, unbroken look.
    lines = [
        '<span class="title-line">Site Report</span>',
        f"{_esc(admin.get('village'))}, {_esc(admin.get('tehsil'))}, {_esc(admin.get('district'))}, "
        f"{_esc(admin.get('state'))}",
        f"{_fmt_num(site.get('latitude'), 5)}, {_fmt_num(site.get('longitude'), 5)} &middot; "
        f"{_fmt_num(site.get('aoi_radius_km'))} km radius",
        f"Purpose: {_esc((purpose or {}).get('label') or 'Not set')}"
        + (f" &middot; Preferences: {_esc(preferences)}" if preferences else ""),
    ]
    if land_price and land_price.get("source") == "User-supplied single-plot price entry":
        lines.append(
            f"Plot: {_fmt_num(land_price.get('area_sqft'), 0)} sqft &middot; "
            f"Rs. {_fmt_num(land_price.get('total_price'), 0)} total "
            f"(Rs. {_fmt_num(land_price.get('price_per_sqft'), 2)}/sqft)"
        )
    text_cell = "<p>" + "<br>".join(lines) + "</p>"
    hero = (
        '<div class="li-hero"><table class="li-hero-table" cellpadding="0" cellspacing="0" border="0"><tr>'
        f"<td>{text_cell}</td>"
        f'<td class="li-hero-logo-cell">{_logo_img_tag()}</td>'
        "</tr></table></div>"
    )

    img = _img_tag(manifest, run_dir, "site_context_map", "Site overview map")
    footer = f'<p class="li-footer">Generated at {_esc(site_summary.get("generated_at"))}</p>'
    return hero + (img or '<p class="li-note">Overview map unavailable for this run.</p>') + footer


# ---------------------------------------------------------------------
# Chapter 1 — Location & Administration
# ---------------------------------------------------------------------


def _guideline_rates_html(rates: dict | None) -> str:
    if not rates:
        return '<p class="li-note">No government guideline rate match for this location.</p>'

    if rates.get("row_kind") == "rural_village":
        rows = [
            ("Residential (per sqm)", _fmt_num(rates.get("plot_residential_sqm"), 0, " Rs.") or "n/a"),
            ("Commercial (per sqm)", _fmt_num(rates.get("plot_commercial_sqm"), 0, " Rs.") or "n/a"),
            ("Industrial (per sqm)", _fmt_num(rates.get("plot_industrial_sqm"), 0, " Rs.") or "n/a"),
            ("Irrigated agri land (per ha)", _fmt_num(rates.get("agri_land_irrigated_per_ha"), 0, " Rs.") or "n/a"),
            ("Unirrigated agri land (per ha)", _fmt_num(rates.get("agri_land_unirrigated_per_ha"), 0, " Rs.") or "n/a"),
        ]
        return _table(["Type", "Rate"], rows) or ""

    if rates.get("row_kind") == "urban_range":
        def _range(min_key, max_key):
            lo, hi = _fmt_num(rates.get(min_key), 0), _fmt_num(rates.get(max_key), 0)
            return f"Rs. {lo} – {hi}" if lo and hi else "n/a"

        rows = [
            ("Residential (per sqm)", _range("plot_residential_sqm_min", "plot_residential_sqm_max")),
            ("Commercial (per sqm)", _range("plot_commercial_sqm_min", "plot_commercial_sqm_max")),
            ("Industrial (per sqm)", _range("plot_industrial_sqm_min", "plot_industrial_sqm_max")),
        ]
        return _table(["Type", "Range"], rows) or ""

    return '<p class="li-note">Guideline rate data unavailable in an expected shape.</p>'


def _chapter_location_admin(site_summary: dict, manifest: dict, run_dir: Path, interp: dict) -> str:
    admin = site_summary.get("administration", {})
    roads = site_summary.get("roads", {})
    railways = site_summary.get("railways", {})
    reference = site_summary.get("reference_location")

    parts = [_card_open("1. Location & Administration")]
    parts.append(_fact("Village / Tehsil / District", f"{_esc(admin.get('village'))} / {_esc(admin.get('tehsil'))} / {_esc(admin.get('district'))}"))
    parts.append(_fact("State", admin.get("state")))

    parts.append("<h3>Connectivity</h3>")
    parts.append(_fact("Nearest road", _fmt_num(roads.get("nearest_road_m"), 0, " m"), note=roads.get("nearest_road_name")))
    if roads.get("nearest_major_road_m") is not None:
        note = roads.get("nearest_major_road_name")
        if roads.get("nearest_major_road_beyond_aoi"):
            note = f"{note or ''} (beyond the {_fmt_num(site_summary.get('site', {}).get('aoi_radius_km'))} km AOI)".strip()
        parts.append(_fact("Nearest major road", _fmt_num(roads.get("nearest_major_road_m"), 0, " m"), note=note))
    parts.append(_fact("Nearest railway", _fmt_num(railways.get("nearest_rail_m"), 0, " m"), note=railways.get("nearest_rail_name")))

    if reference and reference.get("lat") is not None:
        parts.append("<h3>Distance from reference location</h3>")
        parts.append(_fact("Reference", reference.get("label") or "Reference location"))
        parts.append(_fact("Straight-line distance", _fmt_num(reference.get("straight_line_km"), 1, " km")))
        if reference.get("travel_distance_km") is not None:
            note = f"~{_fmt_num(reference.get('travel_time_min'), 0)} min" if reference.get("travel_time_min") is not None else None
            parts.append(_fact("By road", _fmt_num(reference.get("travel_distance_km"), 1, " km"), note=note))
        transit = reference.get("transit") or {}
        if transit.get("bus_duration_min") or transit.get("train_duration_min"):
            legs = []
            if transit.get("bus_duration_min") is not None:
                legs.append(f"{transit['bus_duration_min']:.0f} min bus")
            if transit.get("train_duration_min") is not None:
                legs.append(f"{transit['train_duration_min']:.0f} min train")
            parts.append(_fact("Public transit", " + ".join(legs)))
        img = _img_tag(manifest, run_dir, "reference_location_map", "Distance to reference location map")
        if img:
            parts.append(img)

    parts.append("<h3>Government guideline land rate</h3>")
    parts.append(_guideline_rates_html(admin.get("govt_guideline_rates")))

    parts.append(_ai_box(interp.get("location_admin")))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


# ---------------------------------------------------------------------
# Chapter 2 — Water & Hydrology
# ---------------------------------------------------------------------


def _named_water_features_table(hydrology: dict) -> str | None:
    bodies = (hydrology.get("sac_water_bodies") or {}).get("water_bodies") or []
    rivers = (hydrology.get("river_floodplain") or {}).get("rivers_in_query_window") or []

    rows = []
    for entry in sorted(bodies, key=lambda e: e.get("distance_m") if e.get("distance_m") is not None else 1e18)[:5]:
        rows.append((
            entry.get("name") or "Unnamed water body",
            entry.get("type") or "n/a",
            _fmt_num(entry.get("distance_m"), 0, " m") or "n/a",
        ))
    for entry in sorted(rivers, key=lambda e: e.get("distance_m") if e.get("distance_m") is not None else 1e18)[:5]:
        rows.append((
            entry.get("name") or "Unnamed river",
            "River",
            _fmt_num(entry.get("distance_m"), 0, " m") or "n/a",
        ))
    return _table(["Name", "Type", "Distance"], rows)


def _seasonal_table(seasonal: list) -> str | None:
    rows = []
    for record in seasonal or []:
        pre = (record.get("pre_monsoon") or {}).get("level_m_bgl")
        mon = (record.get("monsoon") or {}).get("level_m_bgl")
        fluct = record.get("fluctuation_m")
        rows.append((
            record.get("year"),
            _fmt_num(pre, 1, " m") or "n/a",
            _fmt_num(mon, 1, " m") or "n/a",
            _fmt_num(fluct, 1, " m") or "n/a",
        ))
    return _table(["Year", "Pre-monsoon depth", "Monsoon depth", "Fluctuation"], rows)


def _rainfall_table(yearly: list) -> str | None:
    rows = []
    for record in yearly or []:
        chirps = record.get("chirps") or {}
        imd = record.get("imd") or {}
        rows.append((
            record.get("year"),
            _fmt_num(chirps.get("annual_mm"), 0, " mm") or "n/a",
            _fmt_num(imd.get("annual_mm"), 0, " mm") or "n/a",
            _fmt_num(chirps.get("monsoon_mm"), 0, " mm") or "n/a",
            _fmt_num(chirps.get("rain_days"), 0) or "n/a",
        ))
    return _table(["Year", "CHIRPS annual", "IMD annual", "CHIRPS monsoon", "Rain days"], rows)


def _chapter_water(site_summary: dict, manifest: dict, run_dir: Path, interp: dict) -> str:
    groundwater = site_summary.get("groundwater", {})
    rainfall = site_summary.get("rainfall", {})
    water_resources = site_summary.get("water_resources", {})
    hydrology = site_summary.get("hydrology", {})

    parts = [_card_open("2. Water & Hydrology")]

    parts.append("<h3>Groundwater</h3>")
    parts.append(_fact("Latest mean depth", _fmt_num(groundwater.get("latest_mean_depth_m_bgl"), 1, " m bgl")))
    parts.append(_fact("5-year mean depth", _fmt_num(groundwater.get("five_year_mean_m_bgl"), 1, " m bgl")))
    parts.append(_fact("Trend", groundwater.get("dominant_trend"), note=_fmt_num(groundwater.get("mean_change_5yr_m"), 2, " m/5yr change")))
    parts.append(_fact("Nearest well", _fmt_num(groundwater.get("nearest_well_m"), 0, " m")))
    img = _img_tag(manifest, run_dir, "groundwater_map", "Groundwater monitoring sites map")
    if img:
        parts.append(img)
    table = _seasonal_table(groundwater.get("seasonal"))
    if table:
        parts.append("<h3>Depth by season, year-wise</h3>")
        parts.append(table)
    chart = _img_tag(manifest, run_dir, "groundwater_chart", "Groundwater seasonal depth chart")
    if chart:
        parts.append(chart)

    parts.append("<h3>Rainfall (2023–2025)</h3>")
    rain_table = _rainfall_table(rainfall.get("yearly"))
    if rain_table:
        parts.append(rain_table)
    parts.append('<p class="li-note">Annual/monsoon totals only — monthly rainfall isn\'t currently tracked by this pipeline.</p>')

    parts.append("<h3>Surface water</h3>")
    parts.append(_fact("Nearest surface water", _fmt_num(water_resources.get("nearest_water_m"), 0, " m"), note=water_resources.get("nearest_water_name")))
    img = _img_tag(manifest, run_dir, "water_bodies_map", "Named water bodies map")
    if img:
        parts.append(img)
    table = _named_water_features_table(hydrology)
    if table:
        parts.append(table)

    river = hydrology.get("river_floodplain") or {}
    parts.append("<h3>Floodplain</h3>")
    if river.get("inside_supplied_river_polygon"):
        parts.append('<p class="li-note">This site falls inside a mapped river floodplain polygon.</p>')
    parts.append(_fact("Nearest river", _fmt_num(river.get("nearest_river_m"), 0, " m"), note=river.get("nearest_river_name")))

    parts.append(_ai_box(interp.get("water_hydrology")))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


# ---------------------------------------------------------------------
# Chapter 3 — Climate & Land Use
# ---------------------------------------------------------------------


def _land_cover_table(classes: list) -> str | None:
    rows = [
        (c.get("name"), _fmt_num(c.get("percentage"), 1, "%") or "n/a", _fmt_num(c.get("area_km2"), 2, " km²") or "n/a")
        for c in sorted(classes or [], key=lambda c: c.get("percentage") or 0, reverse=True)
    ]
    return _table(["Class", "Share", "Area"], rows)


def _chapter_climate_land(site_summary: dict, manifest: dict, run_dir: Path, interp: dict) -> str:
    land_cover = site_summary.get("land_cover", {})
    ndvi = site_summary.get("ndvi", {})
    air_temp = site_summary.get("air_temperature", {})
    lst = site_summary.get("land_surface_temperature", {})
    terrain = site_summary.get("terrain", {})

    parts = [_card_open("3. Climate & Land Use")]

    parts.append("<h3>Temperature</h3>")
    parts.append(_fact("Annual mean air temperature", _fmt_num(air_temp.get("annual_mean_c"), 1, " °C")))
    parts.append(_fact(
        "Warmest / coldest month",
        f"{_month_name(air_temp.get('warmest_month'))} ({_fmt_num(air_temp.get('warmest_month_c'), 1, ' °C')}) / "
        f"{_month_name(air_temp.get('coldest_month'))} ({_fmt_num(air_temp.get('coldest_month_c'), 1, ' °C')})"
        if air_temp.get("warmest_month") else None,
    ))
    parts.append(_fact(
        "Land surface temperature (annual)",
        _fmt_num(lst.get("mean_c"), 1, " °C"),
        note=f"range {_fmt_num(lst.get('min_c'), 1)}–{_fmt_num(lst.get('max_c'), 1)} °C" if lst.get("min_c") is not None else None,
    ))

    parts.append("<h3>Vegetation (NDVI)</h3>")
    parts.append(_fact("Whole-year mean NDVI", _fmt_num(ndvi.get("mean"), 2), note=f"range {_fmt_num(ndvi.get('min'), 2)}–{_fmt_num(ndvi.get('max'), 2)}" if ndvi.get("min") is not None else None))
    parts.append(_fact("Area with NDVI ≥ 0.30 / ≥ 0.50", f"{_fmt_num(ndvi.get('share_ge_030_percent'), 0, '%')} / {_fmt_num(ndvi.get('share_ge_050_percent'), 0, '%')}"))
    img = _img_tag(manifest, run_dir, "ndvi_map", "NDVI map")
    if img:
        parts.append(img)

    parts.append("<h3>Land cover</h3>")
    chart = _img_tag(manifest, run_dir, "land_cover_chart", "Land cover pie chart")
    if chart:
        parts.append(chart)
    table = _land_cover_table(land_cover.get("classes"))
    if table:
        parts.append(table)
    img = _img_tag(manifest, run_dir, "land_cover_map", "Land cover map")
    if img:
        parts.append(img)

    parts.append("<h3>Terrain</h3>")
    parts.append(_fact("Elevation", f"{_fmt_num(terrain.get('elevation_min_m'), 0)}–{_fmt_num(terrain.get('elevation_max_m'), 0)} m", note=f"mean {_fmt_num(terrain.get('elevation_mean_m'), 0)} m"))
    range_90 = terrain.get("slope_range_90pct_deg") or [None, None]
    parts.append(_fact(
        "Slope", f"mean {_fmt_num(terrain.get('slope_mean_deg'), 1)}°, median {_fmt_num(terrain.get('slope_median_deg'), 1)}°",
        note=f"90% of AOI between {_fmt_num(range_90[0], 1)}°–{_fmt_num(range_90[1], 1)}°; right at the site (100 m): {_fmt_num(terrain.get('slope_median_deg_100m'), 1)}°",
    ))
    chart = _img_tag(manifest, run_dir, "slope_class_chart", "Slope class distribution chart")
    if chart:
        parts.append(chart)

    parts.append(_ai_box(interp.get("climate_land")))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


# ---------------------------------------------------------------------
# Chapter 4 — Human / Locality Context
# ---------------------------------------------------------------------


def _nearby_categories_table(categories: list) -> str | None:
    rows = [
        (
            (c.get("category") or "").replace("_", " ").title(),
            c.get("count", 0),
            (c.get("nearest") or {}).get("name") or "n/a",
            _fmt_num((c.get("nearest") or {}).get("distance_km"), 1, " km") or "n/a",
        )
        for c in categories or []
    ]
    return _table(["Category", "Count", "Nearest", "Distance"], rows)


def _chapter_locality(site_summary: dict, manifest: dict, run_dir: Path, interp: dict) -> str:
    admin = site_summary.get("administration", {})
    census = admin.get("census") or {}
    land_use = admin.get("land_use") or {}
    settlements = site_summary.get("settlements", {})
    nearby = site_summary.get("nearby_places", {})
    land_cover = site_summary.get("land_cover", {})

    built_up = next(
        (c.get("percentage") for c in (land_cover.get("classes") or []) if c.get("name") == "Built-up"), None
    )

    parts = [_card_open("4. Human / Locality Context")]

    parts.append("<h3>Population &amp; census</h3>")
    parts.append(_fact("Village population", _fmt_num(census.get("population_total") or land_use.get("population"), 0)))
    parts.append(_fact("Literacy", _fmt_num(census.get("literacy_percent"), 1, "%")))
    parts.append(_fact("SC / ST share", f"{_fmt_num(census.get('sc_percent'), 1, '%')} / {_fmt_num(census.get('st_percent'), 1, '%')}"))
    parts.append(_fact("Built-up land (WorldCover)", _fmt_num(built_up, 1, "%")))

    parts.append("<h3>Settlements</h3>")
    parts.append(_fact(
        "Nearest settlement (any type)", settlements.get("nearest_settlement_name"),
        note=f"{settlements.get('nearest_settlement_type')}, {_fmt_num(settlements.get('distance_to_nearest_settlement_km'), 1, ' km')}"
        if settlements.get("nearest_settlement_name") else None,
    ))
    developed = _settlement_context(settlements)
    if developed:
        parts.append(_fact("Nearest developed city/town", developed.get("name"), note=f"{_fmt_num(developed.get('distance_km'), 1, ' km')}"))
    parts.append(_fact("Settlements within 10 km", settlements.get("settlement_count_10km")))

    parts.append("<h3>Nearby places (10 km)</h3>")
    table = _nearby_categories_table(nearby.get("categories"))
    if table:
        parts.append(table)
    img = _img_tag(manifest, run_dir, "nearby_places_map", "Nearby places map")
    if img:
        parts.append(img)
    else:
        parts.append('<p class="li-note">Nearby-places map unavailable for this run (no places data, or SerpApi was disabled).</p>')

    parts.append(_ai_box(interp.get("locality_context")))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


# ---------------------------------------------------------------------
# Chapter 5 — Overall Summary & Suitability (fully AI)
# ---------------------------------------------------------------------


def _chapter_summary(site_summary: dict, interp: dict) -> str:
    purpose = site_summary.get("buying_purpose")
    purpose_label = (purpose or {}).get("label") or "the stated purpose"
    parts = [_card_open(f"5. Overall Summary & Suitability for {_esc(purpose_label)}")]
    if not purpose:
        parts.append('<p class="li-note">Set a buying purpose in the app to generate this section.</p>')
        parts.append(_CARD_CLOSE)
        return "".join(parts)

    strengths = interp.get("strengths") or []
    concerns = interp.get("concerns") or []
    trade_offs = interp.get("trade_offs") or []
    if strengths:
        parts.append("<h3>Strengths</h3>")
        parts.append(_bullets(strengths, marker="+"))
    if concerns:
        parts.append("<h3>Concerns</h3>")
        parts.append(_bullets(concerns, marker="!"))
    if trade_offs:
        parts.append("<h3>Trade-offs</h3>")
        parts.append(_bullets(trade_offs, marker="•"))
    if not (strengths or concerns or trade_offs):
        parts.append('<p class="li-note">AI summary unavailable (Q&A error or nothing returned).</p>')

    parts.append(_sources_note_html(site_summary))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


# ---------------------------------------------------------------------
# Chapter 6 — Data Sources & Limitations (short, deterministic, no AI)
# ---------------------------------------------------------------------

_SECTION_SOURCES = [
    ("Administration", "administration", "Survey of India village boundaries + MP govt guideline rates + census"),
    ("Land cover", "land_cover", "ESA WorldCover v200"),
    ("Groundwater", "groundwater", "CGWB manual monthly groundwater monitoring"),
    ("Rainfall", "rainfall", "CHIRPS v3 + IMD 0.25° gridded rainfall"),
    ("Air temperature", "air_temperature", "ERA5-Land monthly aggregates"),
    ("Land surface temperature", "land_surface_temperature", "Landsat 8 Collection 2 Level 2"),
    ("Terrain", "terrain", "USGS SRTM 1 arc-second"),
    ("Roads / railways / water", "roads", "OpenStreetMap via OSMnx"),
    ("Hydrology", "hydrology", "River floodplain + SAC water body reference layers"),
    ("Settlements & nearby places", "nearby_places", "OpenStreetMap + SerpApi / Google Maps"),
    ("Vegetation (NDVI)", "ndvi", "Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)"),
]


def _sources_note_html(site_summary: dict) -> str:
    """No chapter number/heading of its own — per product steer, this reads
    as a closing note continuing chapter 5's card, not a separate numbered
    chapter (so no page-break-before, no card wrapper)."""

    unavailable = [
        label for label, key, _source in _SECTION_SOURCES
        if (site_summary.get(key) or {}).get("available") is False
    ]
    unavailable_note = (
        f" For this run, {', '.join(unavailable)} could not be fetched and is shown as unavailable above."
        if unavailable else ""
    )

    return (
        '<p style="font-size:12px;line-height:1.6;color:#52514e;margin-top:10px;'
        'border-top:1px solid #f0efec;padding-top:10px;">'
        "<strong>Note:</strong> This report draws on Sentinel-2 (vegetation), ESA WorldCover (land cover), "
        "CGWB (groundwater), CHIRPS/IMD (rainfall), ERA5-Land (air temperature), Landsat (surface "
        "temperature), USGS SRTM (terrain), OpenStreetMap (roads/rail/water), Survey of India village "
        "boundaries with MP govt guideline rates and census, and SerpApi/Google Maps (nearby places)."
        f"{unavailable_note} "
        "It contains no price-trend or appreciation data — any land price shown is a single point-in-time "
        "snapshot, not a forecast. It assesses physical suitability (terrain, water, climate, access) from "
        "real data; legal suitability (title, zoning, permits) is not covered. OSM, SerpApi, and Overpass "
        "coverage varies by area, and NDVI/rainfall/temperature reflect the data window used at analysis "
        "time, not a guarantee of future conditions. Electrical power grid, legal documents, and market "
        "price datasets are not yet part of this system and will be added soon.</p>"
    )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def generate_report(site_summary: dict, summary_path: Path) -> tuple[str, dict]:
    """Assemble the full report HTML for one run — see module docstring.
    Purely an assembler: every chart/map is read from ``report_assets/``
    (pre-rendered during the analysis run itself, see
    ``aI_agents.report_assets.generate_report_assets``), and the only network call
    is the one consolidated Groq interpretation call.

    Returns:
        ``(html, site_summary)`` — ``site_summary`` is returned unchanged
        (kept in the signature for compatibility with the caller, which
        previously received a possibly-mutated copy).
    """

    run_dir = summary_path.parent
    manifest = _load_manifest(run_dir)

    purpose = site_summary.get("buying_purpose")
    interp: dict = {}
    if purpose:
        purpose_label = purpose.get("label") or purpose.get("key")
        try:
            interp = generate_report_interpretations(
                site_summary, purpose_label, preferences=site_summary.get("buying_preferences")
            )
        except ChatUnavailable:
            interp = {}

    footer = f'<p class="li-footer">Generated at {_esc(site_summary.get("generated_at"))}</p>'
    chapters = [
        _cover(site_summary, manifest, run_dir),
        _chapter_location_admin(site_summary, manifest, run_dir, interp),
        _chapter_water(site_summary, manifest, run_dir, interp),
        _chapter_climate_land(site_summary, manifest, run_dir, interp),
        _chapter_locality(site_summary, manifest, run_dir, interp),
        _chapter_summary(site_summary, interp),
        footer,
    ]

    html = _REPORT_CSS + '<div class="li-report">' + "".join(chapters) + "</div>"
    return html, site_summary


def report_to_pdf_bytes(html: str) -> bytes | None:
    """Convert the report HTML to a PDF via ``xhtml2pdf`` (pure Python, no
    system dependencies) — the report's CSS is deliberately kept portable
    (no custom properties, no fixed positioning, no iframes) so the same
    HTML that renders on-screen converts cleanly. Returns ``None`` if the
    conversion fails rather than raising, so a PDF hiccup never blocks the
    on-screen report."""

    try:
        from xhtml2pdf import pisa
    except ImportError:
        return None

    buffer = io.BytesIO()
    result = pisa.CreatePDF(io.StringIO(html), dest=buffer)
    if result.err:
        return None
    return buffer.getvalue()
