"""River floodplain proxy + SAC water bodies.

Ports the river-polygon and SAC-waterbody portion of notebook cell 27
(large local GeoJSON files, read with a padded bbox window).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config
from .aoi import AOI, read_local_vector

RIVER_COLUMNS = ["id", "rivname", "ripcode"]
SAC_COLUMNS = ["id", "wetcode", "wetname", "level_i", "level_ii", "level_iii", "area_ha"]


def get_hydrology(aoi: AOI) -> dict:
    """Compute the river-floodplain proxy and nearby SAC water-body context
    for the AOI, from two large local GeoJSON datasets.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"river_polygons_gdf":
        GeoDataFrame, "water_bodies_gdf": GeoDataFrame (SAC water bodies
        intersecting the AOI), "summary": {
        "river_floodplain": {inside_supplied_river_polygon (bool),
        nearest_river_m, nearest_river_name, features_in_query_window,
        rivers_in_query_window (list, nearest-first), source},
        "sac_water_bodies": {waterbodies_in_aoi (count), nearest_water_body_m,
        nearest_water_body_name, nearest_water_body_type, water_bodies
        (list, nearest-first), source}}}. Distances in meters, areas in
        hectares (water_bodies[i].area_ha).
    """

    warnings: list[str] = []
    site_point_metric = aoi.center_utm

    # --- River polygons (floodplain proxy) ------------------------------
    river_polygons = read_local_vector(config.RIVER_POLYGON_FILE, RIVER_COLUMNS, aoi)
    if not config.RIVER_POLYGON_FILE.exists():
        warnings.append(f"River polygon file not found: {config.RIVER_POLYGON_FILE}")

    river_metric = river_polygons.to_crs(aoi.utm_crs)
    river_metric = river_metric[river_metric.geometry.notna()].copy()

    if river_metric.empty:
        river_floodplain, nearest_river_m, nearest_river_row = False, None, None
    else:
        river_floodplain = bool(river_metric.geometry.covers(site_point_metric).any())
        river_distances = river_metric.geometry.distance(site_point_metric)
        idx = river_distances.idxmin()
        nearest_river_m = float(river_distances.loc[idx])
        nearest_river_row = river_polygons.loc[idx]

    # Every river polygon in the (padded) query window, nearest-first.
    # Previously only rendered as a map layer; not carried into the JSON.
    rivers_in_query_window = []
    if not river_metric.empty:
        river_dist_all = river_metric.geometry.distance(site_point_metric)
        for idx in river_dist_all.sort_values().index:
            row = river_polygons.loc[idx]
            rivers_in_query_window.append({
                "name": str(row.get("rivname")) if pd.notna(row.get("rivname")) else None,
                "ripcode": str(row.get("ripcode")) if pd.notna(row.get("ripcode")) else None,
                "distance_m": float(river_dist_all.loc[idx]),
            })

    river_summary = {
        "inside_supplied_river_polygon": river_floodplain,
        "nearest_river_m": nearest_river_m,
        "nearest_river_name": (
            str(nearest_river_row.get("rivname"))
            if nearest_river_row is not None and pd.notna(nearest_river_row.get("rivname"))
            else None
        ),
        "features_in_query_window": int(len(river_metric)),
        "rivers_in_query_window": rivers_in_query_window,
        "source": "river_polygon.GeoJSON; polygon coverage used as floodplain proxy",
    }

    # --- SAC water bodies -------------------------------------------
    water_bodies = read_local_vector(config.SAC_WATERBODY_FILE, SAC_COLUMNS, aoi)
    if not config.SAC_WATERBODY_FILE.exists():
        warnings.append(f"SAC water body file not found: {config.SAC_WATERBODY_FILE}")

    water_bodies_metric = water_bodies.to_crs(aoi.utm_crs)
    water_bodies_in_aoi = water_bodies.loc[water_bodies_metric.geometry.intersects(aoi.polygon_utm)].copy()
    water_bodies_in_aoi_metric = water_bodies_in_aoi.to_crs(aoi.utm_crs)

    if water_bodies_in_aoi_metric.empty:
        nearest_sac_water_m, nearest_sac_water_row = None, None
    else:
        sac_distances = water_bodies_in_aoi_metric.geometry.distance(site_point_metric)
        idx = sac_distances.idxmin()
        nearest_sac_water_m = float(sac_distances.loc[idx])
        nearest_sac_water_row = water_bodies_in_aoi.loc[idx]

    # Every SAC water body intersecting the AOI, nearest-first. Previously
    # only rendered as a map layer; not carried into the JSON.
    water_bodies_list = []
    if not water_bodies_in_aoi_metric.empty:
        wb_dist_all = water_bodies_in_aoi_metric.geometry.distance(site_point_metric)
        for idx in wb_dist_all.sort_values().index:
            row = water_bodies_in_aoi.loc[idx]
            name = str(row.get("wetname")) if pd.notna(row.get("wetname")) and str(row.get("wetname")).strip() else None
            water_bodies_list.append({
                "name": name,
                "type": str(row.get("level_iii")) if pd.notna(row.get("level_iii")) else None,
                "area_ha": None if pd.isna(row.get("area_ha")) else float(row.get("area_ha")),
                "distance_m": float(wb_dist_all.loc[idx]),
            })

    sac_water_summary = {
        "waterbodies_in_aoi": int(len(water_bodies_in_aoi)),
        "nearest_water_body_m": nearest_sac_water_m,
        "nearest_water_body_name": (
            str(nearest_sac_water_row.get("wetname"))
            if nearest_sac_water_row is not None and pd.notna(nearest_sac_water_row.get("wetname"))
            and str(nearest_sac_water_row.get("wetname")).strip()
            else None
        ),
        "nearest_water_body_type": (
            str(nearest_sac_water_row.get("level_iii"))
            if nearest_sac_water_row is not None and pd.notna(nearest_sac_water_row.get("level_iii"))
            else None
        ),
        "water_bodies": water_bodies_list,
        "source": "SAC water bodies (wb_sac_mp.GeoJSON)",
    }

    return {
        "dataset": "hydrology",
        "source": "river_polygon.GeoJSON + wb_sac_mp.GeoJSON",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
        "observations": {
            "river_polygons_gdf": river_polygons,
            "water_bodies_gdf": water_bodies_in_aoi,
            "summary": {"river_floodplain": river_summary, "sac_water_bodies": sac_water_summary},
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "medium",
        "warnings": warnings,
        "limitations": [
            "River-floodplain flag is a coverage proxy from a single supplied polygon layer, "
            "not an official flood-hazard delineation.",
            "MP-only source.",
        ],
    }
