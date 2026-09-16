"""Combine every ``get_*`` module's envelope into one flat ``site_summary``
dict — the single, stable JSON-serializable record of everything computed
for a site. Downstream consumers (Stage B's Streamlit results view, Stage
C's decision-engine notebook) read *this* shape, not any individual
module's envelope.

============================================================================
SCHEMA — site_summary (schema_version "1.0")
============================================================================

This is a documented restructuring of the original notebook's
``land_intelligence_site_summary.json``, not a byte-for-byte mirror of it.
Two duplicate blocks the notebook produced by accident are dropped
entirely (``village_boundary`` was an exact copy of ``village_land_use``;
``census_features`` was an exact copy of ``village_census`` — both are
folded into ``administration`` below, see "Renamed/restructured" notes).
A few naming/unit inconsistencies are cleaned up and several values the
notebook computed but never carried past its own printed output or map
popups are now included (see "Newly captured" below). Every GEE-derived
numeric value is otherwise reproduced exactly (same source collections,
same reducers, same scale) — only shape, naming and unit conventions
change.

Top-level keys:

  schema_version   str  — this document's version, e.g. "1.0".
  generated_at     str  — "YYYY-MM-DD HH:MM", pipeline run time.

  site             dict — {latitude, longitude, aoi_radius_km,
                           aoi_area_km2, metric_crs}. ``metric_crs`` is the
                           local UTM zone (EPSG code) used for every
                           length/area/distance computation below.

  administration   dict — host-village identity and everything keyed off
                          "the village containing/nearest the site":
      village, tehsil, block, district, state    str | None
      boundary_village_code                       str | None
      point_inside_village_polygon                bool | None
      source                                       str
      govt_guideline_rates  dict | None — MP Govt district guideline
                                  (circle) rate for the host village: a
                                  specific village's rate (row_kind
                                  "rural_village": village, vlcode, tehsil,
                                  frontage_type, match_score,
                                  plot_residential/commercial/industrial_sqm,
                                  agri_land_irrigated/unirrigated_per_ha) if
                                  it matched one in the source PDF, else a
                                  (district, tehsil) rate range (row_kind
                                  "urban_range": tehsil, n_wards, n_rows,
                                  same rate fields with _min/_max) if the
                                  host village is itself urban (no
                                  ward-boundary shapefile exists to resolve
                                  any finer than that); None if no district
                                  table or matching row was found. See
                                  scripts/build_district_guideline_tables.py
                                  for how these per-district tables were
                                  built (fuzzy name-matched from the MP
                                  Govt's FY2026-27 guideline-rate PDFs).
      land_use            dict — village-level land-use areas (hectares)
                                  and shares (percent, 0-100) from the
                                  Survey-of-India boundary attributes.
      census              dict — village-level Census features (population,
                                  households, literacy/SC/ST percent, worker
                                  shares) joined by normalized village name.
      census_join         dict — {method, boundary_villages,
                                   matched_villages, unmatched_villages,
                                   source} — join diagnostics.
      aoi_overlap         dict — {village_count, villages: [name, ...],
                                   tehsil_count, tehsils: [...],
                                   district_count, districts: [...]} for
                                   every village/tehsil/district
                                   intersecting the AOI (not just the host).

  land_cover       dict — {classes: [{name, area_km2, percentage}, ...]
                           (percentage is 0-100, sums to ~100 across
                           classes), aoi_area_km2, dominant_class, source}.
                           ESA WorldCover v200.

  groundwater      dict — CGWB manual monitoring, AOI + per-well:
      wells, latest_observation, latest_mean/min/max_depth_m_bgl
      five_year_mean_m_bgl          — mean of each well's own 5-year avg
      five_year_mean_m_bgl_pooled   — mean over every raw observation from
                                       every well within 5 years of the
                                       AOI-wide latest date (previously
                                       printed by the notebook, never
                                       stored — see "Newly captured")
      five_year_observation_count   — size of that pooled window
      mean_change_5yr_m, dominant_trend, trend_counts
      nearest_well_m, nearest_well_depth_m_bgl, well_depth_range_m
      seasonal_windows    dict — {pre_monsoon_months, monsoon_months}
      seasonal            list — one entry per year: {year, pre_monsoon,
                                  monsoon, observed_months, fluctuation_m},
                                  AOI average (replaces the original's
                                  year-string-keyed dict with a list).
      wells_detail        list — one entry per monitoring well (previously
                                  only rendered into map marker popups —
                                  see "Newly captured"): {well_no,
                                  latitude, longitude, village, district,
                                  well_type, agency, depth_of_well_m,
                                  latest_date, latest_level_m_bgl,
                                  five_year_avg_m_bgl, change_5yr_m, trend,
                                  trend_description, n_5yr_observations,
                                  seasonal: [...]}.
      source

  rainfall         dict — CHIRPS v3 (AOI mean) + IMD 0.25° (nearest grid
                          cell), kept side by side (never blended):
      years, monsoon_months, thresholds_mm
      yearly       list — one entry per year: {year, chirps: {...},
                           imd: {...} | None, diff_percent: {...}}
                           (replaces the original's two parallel
                           year-string-keyed dicts).
      means        dict — {chirps: {...}, imd: {...}} — multi-year means.
      wettest_day_mm dict — {chirps, imd} max single-day rainfall across
                             the whole period (previously printed only —
                             see "Newly captured").
      imd_grid     list — [{year, latitude, longitude}, ...] IMD grid-cell
                           location used per year.
      notes, source

  air_temperature  dict — ERA5-Land 2 m air temperature: {annual_mean_c,
                          coldest_month, coldest_month_c, warmest_month,
                          warmest_month_c, months, period, source}.

  land_surface_temperature  dict — Landsat 8 C2 L2 median composite:
                          {mean_c, min_c, max_c, scenes, source}.

  terrain          dict — SRTM 30 m: {elevation_mean/min/max_m,
                          slope_mean/min/max_deg,
                          slope_class_share: [{class, label,
                          share_percent}, ...] (percent 0-100, was a 0-1
                          fraction under symbol-laden dict keys in the
                          original), source}.

  roads            dict — OSM: {features, total_length_km,
                          length_km_by_class: [{highway_class, length_km},
                          ...] (was a dict in the original), density_km_per_km2,
                          nearest_road_m, nearest_road_class,
                          nearest_road_name, nearest_major_road_m,
                          nearest_major_road_class, nearest_major_road_name,
                          nearest_major_road_beyond_aoi (bool — True if no
                          motorway/trunk/primary/secondary road existed
                          within the AOI itself and an expanding-radius
                          search outside it found the nearest one instead;
                          see get_osm_features._find_nearest_major_road_beyond_aoi),
                          source}.

  railways         dict — OSM: {features, nearest_rail_m,
                          nearest_rail_name, source}.

  water_resources  dict — OSM mapped water: {features, waterbody_area_km2,
                          waterway_length_km, waterbodies, waterways,
                          waterway_types: [{type, count}, ...] (was a dict),
                          nearest_water_m, nearest_water_type,
                          nearest_water_name,
                          waterbody_share_of_aoi_percent (renamed from
                          "waterbody_share_of_aoi" and converted from a 0-1
                          fraction to 0-100 percent, for consistency with
                          every other *_percent field),
                          worldcover_permanent_water_percent (cross-checked
                          against the independent land_cover classification;
                          renamed from "worldcover_permanent_water_pct" for
                          naming consistency), source}.

  hydrology        dict — supplied vector datasets:
      river_floodplain  dict — {inside_supplied_river_polygon,
                                 nearest_river_m, nearest_river_name,
                                 features_in_query_window,
                                 rivers_in_query_window: [{name, ripcode,
                                 distance_m}, ...] (previously map-only —
                                 see "Newly captured"), source}.
      sac_water_bodies  dict — {waterbodies_in_aoi, nearest_water_body_m,
                                 nearest_water_body_name,
                                 nearest_water_body_type,
                                 water_bodies: [{name, type, area_ha,
                                 distance_m}, ...] (previously map-only —
                                 see "Newly captured"), source}.
      (Note: the original's stray "villages_in_aoi"/"village_boundary_source"
      keys under "hydrology" are administrative, not hydrological, and now
      live under administration.aoi_overlap / administration.source.)

  settlements      dict — OSM settlements within 10 km: {settlement_count_10km,
                          distance_to_nearest_settlement_km,
                          nearest_settlement_name, nearest_settlement_type,
                          search_radius_km,
                          settlements: [{name, place_type, distance_km}, ...]
                          nearest-first (previously map-only — see "Newly
                          captured"), source}.

  nearby_places    dict — SerpApi category searches within 10 km:
                          {search_radius_km, serpapi_enabled,
                          categories: [{category, count, nearest,
                          places: [...]} , ...] (was a dict keyed by
                          category name; "places" is the full fetched list,
                          previously map-only — see "Newly captured";
                          "nearest" is kept as a convenience duplicate of
                          places[0]), source}.

  ndvi             dict — Sentinel-2 median composite, TWO numbers: top-level
                          fields are the trailing-365-day whole-year
                          composite — {mean, min, max, share_ge_030_percent,
                          share_ge_050_percent (renamed + converted from 0-1
                          fractions, for consistency), period, source,
                          peak: {mean, period, scene_count, source} | None
                          — the single greenest 3-month period FOUND WITHIN
                          that year (not a guessed season like "monsoon" —
                          found by scanning every quarter, since e.g.
                          irrigated winter-wheat cropland actually peaks in
                          Rabi season, not during the monsoon). peak is
                          deliberately lighter-weight than the whole-year
                          block (mean only, no min/max/share) — it reuses
                          the scan's own coarse-resolution value rather than
                          re-compositing the period a second time at full
                          resolution just for a fuller breakdown}. Two
                          numbers because a single whole-year median
                          understates peak-season vegetation vigor on
                          seasonal cropland (dragged down by fallow/dry
                          periods), while a single assumed-season snapshot
                          can miss the real peak entirely.

  land_price       dict, OPTIONAL, absent by default — {observations,
                          median/min/max_price_per_acre, price_per_sqft,
                          total_price, area_sqft, source}. NOT
                          produced by this module (there is no automated
                          land-price data source): added post-hoc whenever
                          the user enters a single plot's price/area via
                          data_analysis_pipeline.custom_facts.merge_custom_facts
                          (source becomes "User-supplied single-plot price
                          entry", observations=1). When present, this is a
                          real observed/expected price for the exact plot
                          being evaluated, so the original notebook's scoring engine
                          treats it as strictly more trustworthy than the
                          govt guideline rate above and scores from it
                          instead whenever both are available; see
                          reference_location below for the sibling optional
                          key added the same way.

  reference_location  dict, OPTIONAL, absent by default — {lat, lon, label,
                          straight_line_km, travel_time_min | None,
                          travel_distance_km | None, transit | None,
                          source: "user_supplied"}. NOT produced by this
                          module or by run_analysis at all: added after the
                          fact, by the user (Stage B's "Custom facts"
                          expander) or by chat (Stage C), via
                          data_analysis_pipeline.custom_facts.merge_custom_facts,
                          which writes this key directly into an existing
                          run's site_summary.json. travel_time_min /
                          travel_distance_km come from a best-effort public
                          OSRM routing lookup and are None if that lookup
                          failed — the original notebook's scoring engine's two
                          factors reading this section (distance_to_
                          reference_km, travel_time_to_reference_min) both
                          degrade to "missing, excluded from scoring" (not
                          a bad score) whenever this section or a given
                          sub-field is absent, exactly like land_price when
                          absent or reporting zero observations.
                          transit (see
                          data_analysis_pipeline.custom_facts.fetch_transit_time)
                          is {bus_duration_min, train_duration_min, walk_km,
                          total_duration_min, departs_at, arrives_at} | None
                          — ONE specific scheduled Google Maps transit
                          itinerary found at the moment custom facts were
                          saved, not a stable daily average (a different
                          query time could find a different bus/train
                          entirely). Display-only, not a scoring input.

----------------------------------------------------------------------------
Dropped duplicates (present twice, byte-identical, in the original JSON):
  - "village_boundary" (== "village_land_use")  -> administration.land_use
  - "census_features" (== "village_census")     -> administration.census

Renamed/restructured (information preserved, shape/units changed):
  - Top-level "villages_in_aoi"/"tehsils_in_aoi"/"districts_in_aoi" (the
    original scattered these as siblings of "administration", while
    "administration.villages_in_aoi" was *also* a same-named but
    different-typed field — an int count vs. this list of names) all move
    to administration.aoi_overlap.{villages,tehsils,districts} (+ *_count).
  - groundwater.seasonal_levels / seasonal_fluctuation_m (dict keyed by
    stringified year) -> groundwater.seasonal (list of per-year records).
  - rainfall.chirps_v3 / imd_025 / comparison (three parallel dicts keyed
    by stringified year) -> rainfall.yearly (one list of per-year records).
    rainfall.means_2023_2025 -> rainfall.means. rainfall.location /
    rainfall.aoi_radius_km dropped as exact duplicates of site.latitude/
    longitude/aoi_radius_km.
  - terrain.slope_class_share (dict keyed by a display string containing
    "°"/"<"/">=") -> list of {class (stable id), label (display string),
    share_percent}. Values converted from 0-1 fractions to 0-100 percent.
  - roads.length_km_by_class, water_resources.waterway_types (dicts) ->
    lists of {..._class|type, length_km|count}.
  - ndvi.share_ge_030 / share_ge_050 and water_resources.
    waterbody_share_of_aoi (0-1 fractions) -> *_percent fields (0-100), for
    consistency with land_cover.percentage and every other share field.
  - nearby_places.serpapi_categories (dict keyed by category name) ->
    nearby_places.categories (list of {category, ...}).

Newly captured (computed by the notebook but dropped on the floor before
reaching the original JSON — printed to console and/or used only for map
layers/popups):
  - groundwater.wells_detail (per-well breakdown table)
  - groundwater.five_year_mean_m_bgl_pooled / five_year_observation_count
  - rainfall.wettest_day_mm
  - hydrology.river_floodplain.rivers_in_query_window
  - hydrology.sac_water_bodies.water_bodies
  - settlements.settlements (full nearest-first list, not just the single
    nearest one)
  - nearby_places.categories[i].places (full fetched list per category,
    not just the nearest one)

----------------------------------------------------------------------------
Failure handling — distinct from "missing vs. zero" (e.g. a nearby_places
category correctly reporting count=0):

  run_success      bool — True only if every get_* module completed without
                          raising; False if one or more genuinely failed
                          (GEE outage, file read error, network timeout) and
                          was replaced by a fallback envelope.
  generated_at     str  — also serves as this document's top-level
                          timestamp field (pipeline run completion time).

  If a given module failed, its section below is replaced with
  {"available": false, "source": <module source>, "warnings": [...],
   "limitations": ["failed: <exception message>"]} instead of the normal
  shape documented above for that key -- check section.get("available",
  True) before reading a section's normal fields. The short failure reason
  lives in that fallback's "limitations"; the full exception + traceback is
  written to that run's log file (<out-dir>/run.log for a CLI run, set up
  by orchestrator.py), not to site_summary itself.
============================================================================
"""

from __future__ import annotations

from datetime import datetime

SCHEMA_VERSION = "1.0"


def _dict_to_list(d: dict, key_name: str, value_name: str) -> list:
    """Convert a {key: value} dict into a list of {key_name: key, value_name:
    value} records — used to flatten a few dict-keyed sections (e.g. roads'
    length-by-class) into the more report-generator-friendly list shape."""

    return [{key_name: k, value_name: v} for k, v in d.items()]


def _unavailable(envelope: dict) -> dict:
    """Build the fallback shape used in place of a section whose get_*
    module failed (envelope["observations"] == {}) — see the "Failure
    handling" note in this module's docstring.

    Args:
        envelope: The (failed) module envelope to pull source/warnings/
            limitations from.

    Returns:
        {"available": False, "source", "warnings", "limitations"}.
    """

    return {
        "available": False,
        "source": envelope.get("source"),
        "warnings": envelope.get("warnings", []),
        "limitations": envelope.get("limitations", []),
    }


def _section_summary(results: dict, module_key: str):
    """Get the JSON-safe "summary" block from one module's envelope.

    Args:
        results: The full envelope dict from :func:`orchestrator.run_analysis`.
        module_key: Key into ``results``, e.g. "groundwater".

    Returns:
        The module's ``observations["summary"]`` dict, or None if that
        module failed (empty observations) or is missing entirely.
    """

    envelope = results.get(module_key, {})
    return envelope.get("observations", {}).get("summary")


def assemble_summary(aoi, results: dict, run_success: bool = True) -> dict:
    """Build the final ``site_summary`` dict — the single, stable,
    JSON-serializable record of everything computed for a site — from
    every ``get_*`` module's envelope. See this module's docstring for the
    full schema.

    Never raises on a failed module: a section whose ``get_*`` call failed
    (envelope["observations"] == {}) is replaced with the
    ``{"available": False, ...}`` fallback shape documented at the top of
    this module, so ``site_summary`` always has every top-level key with
    *something* well-formed at it.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        results: Dict of every ``get_*`` module's envelope, keyed by module
            name (see :mod:`orchestrator` for the exact keys used).
        run_success: Whether every module completed without falling back to
            a failure envelope (normally computed by :mod:`orchestrator`);
            carried through unchanged as the top-level "run_success" field.

    Returns:
        The assembled ``site_summary`` dict (schema documented at the top
        of this module), ready to ``json.dump``.
    """

    osm = results.get("osm", {})
    sentinel = _section_summary(results, "sentinel")
    land_cover = _section_summary(results, "land_cover")
    lst = _section_summary(results, "lst")
    air_temp = _section_summary(results, "air_temp")
    terrain = _section_summary(results, "terrain")
    groundwater = _section_summary(results, "groundwater")
    hydrology = _section_summary(results, "hydrology")
    rainfall = _section_summary(results, "rainfall")
    nearby = _section_summary(results, "nearby_places")
    admin = _section_summary(results, "admin")

    # --- administration ---------------------------------------------------
    if admin is not None:
        try:
            administration = dict(admin["administration"])
            administration["land_use"] = admin["land_use"]
            administration["census"] = admin["census"]
            administration["census_join"] = admin["census_join"]
            administration["aoi_overlap"] = {
                "village_count": len(admin["villages_in_aoi"]),
                "villages": admin["villages_in_aoi"],
                "tehsil_count": len(admin["tehsils_in_aoi"]),
                "tehsils": admin["tehsils_in_aoi"],
                "district_count": len(admin["districts_in_aoi"]),
                "districts": admin["districts_in_aoi"],
            }
        except Exception as error:  # malformed but non-empty observations
            administration = _unavailable(results.get("admin", {}))
            administration["limitations"] = administration["limitations"] + [f"assembly failed: {error}"]
    else:
        administration = _unavailable(results.get("admin", {}))

    # --- roads / railways / water_resources -------------------------------
    osm_obs = osm.get("observations", {})
    if "roads" in osm_obs:
        road_summary = dict(osm_obs["roads"]["summary"])
        if "length_km_by_class" in road_summary:
            road_summary["length_km_by_class"] = _dict_to_list(
                road_summary["length_km_by_class"], "highway_class", "length_km"
            )
    else:
        road_summary = _unavailable(osm)

    rail_summary = dict(osm_obs["railways"]["summary"]) if "railways" in osm_obs else _unavailable(osm)

    if "water" in osm_obs:
        water_summary = dict(osm_obs["water"]["summary"])
        if "waterway_types" in water_summary:
            water_summary["waterway_types"] = _dict_to_list(water_summary["waterway_types"], "type", "count")
        if "waterbody_share_of_aoi" in water_summary:
            water_summary["waterbody_share_of_aoi_percent"] = water_summary.pop("waterbody_share_of_aoi") * 100
        # Cross-reference against the independent WorldCover classification
        # (None if land_cover itself failed — a cross-check, not a hard dependency).
        permanent_water_class = next(
            (c for c in (land_cover or {}).get("classes", []) if c["name"] == "Permanent water"), None
        )
        water_summary["worldcover_permanent_water_percent"] = (
            permanent_water_class["percentage"] if permanent_water_class else None
        )
    else:
        water_summary = _unavailable(osm)

    site_summary = {
        "schema_version": SCHEMA_VERSION,
        "run_success": bool(run_success),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "site": {
            "latitude": float(aoi.lat),
            "longitude": float(aoi.lon),
            "aoi_radius_km": aoi.radius_km,
            "aoi_area_km2": aoi.area_km2,
            "metric_crs": str(aoi.utm_crs),
        },
        "administration": administration,
        "land_cover": land_cover if land_cover is not None else _unavailable(results.get("land_cover", {})),
        "groundwater": groundwater if groundwater is not None else _unavailable(results.get("groundwater", {})),
        "rainfall": rainfall if rainfall is not None else _unavailable(results.get("rainfall", {})),
        "air_temperature": air_temp if air_temp is not None else _unavailable(results.get("air_temp", {})),
        "land_surface_temperature": lst if lst is not None else _unavailable(results.get("lst", {})),
        "terrain": terrain if terrain is not None else _unavailable(results.get("terrain", {})),
        "roads": road_summary,
        "railways": rail_summary,
        "water_resources": water_summary,
        "hydrology": hydrology if hydrology is not None else _unavailable(results.get("hydrology", {})),
        "settlements": (
            nearby["settlements"] if nearby is not None else _unavailable(results.get("nearby_places", {}))
        ),
        "nearby_places": (
            nearby["nearby_places"] if nearby is not None else _unavailable(results.get("nearby_places", {}))
        ),
        "ndvi": sentinel if sentinel is not None else _unavailable(results.get("sentinel", {})),
    }

    return site_summary
