"""CGWB manual monthly groundwater levels.

Ports notebook cells 16-18 (load + spatial filter), 20-23 (seasonal/5-year
per-well summaries) and the groundwater portion of cell 61 (AOI summary,
nearest well).
"""

from __future__ import annotations

from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import config
from .aoi import AOI

REQUIRED_COLUMNS = {"Latitude", "Longitude", "Water Level"}

GW_SEASON_YEARS = [2023, 2024, 2025]
PRE_MONSOON_MONTHS = [3, 4, 5, 6]
MONSOON_MONTHS = [7, 8, 9]
TREND_THRESHOLD_M = 0.5


def _clean_opt(value):
    """Return a stripped string, or None for null/empty values."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _season_level(df: pd.DataFrame, year: int, months: list[int]):
    """Mean water level for one year/season window.

    Args:
        df: Groundwater observations with "date" and "Water Level" columns.
        year: Calendar year to filter to.
        months: Month numbers (1-12) making up the season window.

    Returns:
        {"level_m_bgl", "observations" (count), "dates" (DD-MM-YYYY,
        sorted), "months" (abbreviations, sorted)}, or None if no
        observation falls in that year/season.
    """

    selected = df[(df["date"].dt.year == year) & (df["date"].dt.month.isin(months))]
    if selected.empty:
        return None
    return {
        "level_m_bgl": float(selected["Water Level"].mean()),
        "observations": int(len(selected)),
        "dates": sorted(selected["date"].dt.strftime("%d-%m-%Y").unique().tolist()),
        "months": sorted(selected["date"].dt.strftime("%b").unique().tolist()),
    }


def _observed_months(df: pd.DataFrame, year: int) -> list[str]:
    """Month abbreviations that actually have an observation in ``year``,
    in chronological order (a season can be missing simply because CGWB
    never measured that month)."""

    selected = df[df["date"].dt.year == year].sort_values("date")
    return list(dict.fromkeys(selected["date"].dt.strftime("%b")))


def _seasonal_levels(df: pd.DataFrame, years=None) -> dict:
    """Pre-monsoon/monsoon mean levels and observed months, per year.

    Args:
        df: Groundwater observations with "date" and "Water Level" columns.
        years: Calendar years to compute for; defaults to GW_SEASON_YEARS.

    Returns:
        {year: {"pre_monsoon": <see _season_level>, "monsoon": <...>,
        "observed_months": [...]}}.
    """

    years = years or GW_SEASON_YEARS
    return {
        year: {
            "pre_monsoon": _season_level(df, year, PRE_MONSOON_MONTHS),
            "monsoon": _season_level(df, year, MONSOON_MONTHS),
            "observed_months": _observed_months(df, year),
        }
        for year in years
    }


def get_groundwater(aoi: AOI) -> dict:
    """Compute CGWB manual-monitoring groundwater-level statistics for the
    AOI: AOI-wide summary, seasonal (pre-monsoon/monsoon) levels, and a
    per-well breakdown with 5-year trend.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"latest_wells_gdf":
        GeoDataFrame (one row per well, most recent observation + trend
        columns), "well_seasonal": {well_no: <see _seasonal_levels>},
        "aoi_seasonal": <see _seasonal_levels>, "summary": {wells (count),
        latest_observation (date), latest_mean/min/max_depth_m_bgl,
        five_year_mean_m_bgl, five_year_mean_m_bgl_pooled,
        five_year_observation_count, mean_change_5yr_m, dominant_trend,
        trend_counts, nearest_well_m, nearest_well_depth_m_bgl,
        well_depth_range_m, seasonal_windows, seasonal (list, one entry per
        year), wells_detail (list, one entry per well), source}}. Depths in
        meters below ground level (m bgl).
    """

    warnings: list[str] = []
    limitations = ["CGWB manual monitoring network is sparse; density varies by district."]

    path = config.GROUNDWATER_FILE
    if path.exists():
        gw_raw = pd.read_csv(path)
    else:
        warnings.append(f"Groundwater file not found: {path}")
        gw_raw = pd.DataFrame(columns=["latitude", "longitude", "water_level_m_bgl", "date", "station_name"])

    if not gw_raw.empty:
        missing = REQUIRED_COLUMNS - set(gw_raw.columns)
        if missing:
            raise ValueError(f"CGWB file is missing columns: {missing}")

        gw_gdf = gpd.GeoDataFrame(
            gw_raw.copy(),
            geometry=gpd.points_from_xy(gw_raw.Longitude, gw_raw.Latitude),
            crs="EPSG:4326",
        )
        gw_gdf_m = gw_gdf.to_crs("EPSG:3857")
        center_m = gpd.GeoSeries([Point(aoi.lon, aoi.lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        gw_gdf = gw_gdf.loc[gw_gdf_m.geometry.distance(center_m) <= aoi.radius_km * 1000].copy()
    else:
        gw_gdf = gpd.GeoDataFrame(
            columns=["Latitude", "Longitude", "Water Level", "geometry"], geometry="geometry", crs="EPSG:4326"
        )

    if gw_gdf.empty:
        summary = {"wells": 0, "source": "CGWB manual monthly groundwater monitoring"}
        return {
            "dataset": "groundwater",
            "source": "CGWB manual monthly groundwater monitoring",
            "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
            "observations": {"latest_wells_gdf": gw_gdf, "well_seasonal": {}, "aoi_seasonal": {}, "summary": summary},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": None,
            "confidence": "low",
            "warnings": warnings + ["No monitoring wells found within the AOI."],
            "limitations": limitations,
        }

    gw = gw_gdf.copy()
    gw["Water Level"] = pd.to_numeric(gw["Water Level"], errors="coerce")
    gw["Depth of Well"] = pd.to_numeric(gw.get("Depth of Well"), errors="coerce")
    # CGWB dates are DD-MM-YYYY (confirmed against the source file: values
    # like "31-05-2013" have no valid month-first reading, and the
    # quarterly Jan/May/Aug/Nov monitoring pattern only lines up under a
    # day-first reading). Without dayfirst=True, pandas defaults to
    # MM-DD-YYYY: every day-of-month <= 12 gets silently misparsed (wrong
    # date, no error), and every day-of-month > 12 becomes unparseable and
    # is silently dropped by errors="coerce" below — together that was
    # corrupting the large majority of this dataset's dates.
    gw["date"] = pd.to_datetime(gw["date"], dayfirst=True, errors="coerce")
    gw = gw.dropna(subset=["Water Level", "date"]).copy()

    latest_date = gw["date"].max()
    five_year_start = latest_date - pd.DateOffset(years=5)
    gw_5yr = gw[gw["date"] >= five_year_start].copy()

    aoi_seasonal = _seasonal_levels(gw)

    # --- Per-well summary (5-year trend + seasonal) --------------------
    well_summaries = []
    well_seasonal: dict = {}

    for well_no, well_data in gw.groupby("Well No"):
        well_data = well_data.sort_values("date")
        latest = well_data.iloc[-1]
        latest_well_date, latest_level = latest["date"], latest["Water Level"]

        well_five_year_start = latest_well_date - pd.DateOffset(years=5)
        recent = well_data[
            (well_data["date"] >= well_five_year_start) & (well_data["date"] <= latest_well_date)
        ].copy()

        well_seasonal[well_no] = _seasonal_levels(well_data)

        earliest = recent.iloc[0]
        change = latest_level - earliest["Water Level"]

        if change > TREND_THRESHOLD_M:
            trend, trend_symbol, trend_description = "Declining", "↑", "Water level has become deeper"
        elif change < -TREND_THRESHOLD_M:
            trend, trend_symbol, trend_description = "Improving", "↓", "Water level has become shallower"
        else:
            trend, trend_symbol, trend_description = "Stable", "→", "Little change in water level"

        well_summaries.append({
            "Well No": well_no,
            "latest_date": latest_well_date,
            "latest_level": latest_level,
            "five_year_start": well_five_year_start,
            "five_year_avg": recent["Water Level"].mean(),
            "change_5yr": change,
            "trend": trend,
            "trend_symbol": trend_symbol,
            "trend_description": trend_description,
            "n_5yr": len(recent),
        })

    well_summary_df = pd.DataFrame(well_summaries)

    latest_wells = gw.sort_values("date").drop_duplicates("Well No", keep="last").copy()
    latest_wells = latest_wells.merge(well_summary_df, on="Well No", how="left")

    # --- AOI-level summary ------------------------------------------
    nearest_well_m, nearest_well = None, None
    valid = latest_wells[latest_wells.geometry.notna()]
    if not valid.empty:
        distances = valid.to_crs(aoi.utm_crs).geometry.distance(aoi.center_utm)
        idx = distances.idxmin()
        nearest_well_m = float(distances.loc[idx])
        nearest_well = valid.loc[idx]

    trend_counts = latest_wells["trend"].value_counts().to_dict() if "trend" in latest_wells.columns else {}

    # AOI-pooled 5-year stats (notebook cell 20): every raw observation from
    # every well within 5 years of the AOI-wide latest observation, pooled
    # together. This differs from "five_year_mean_m_bgl" below (the mean of
    # each *well's own* 5-year average) — the notebook printed this pooled
    # figure to console but never stored it, so it is captured here.
    five_year_mean_m_bgl_pooled = float(gw_5yr["Water Level"].mean()) if not gw_5yr.empty else None
    five_year_observation_count = int(len(gw_5yr))

    # Per-well breakdown (notebook cells 21/23) — previously only rendered
    # into the map's marker popups, never carried into the JSON summary.
    wells_detail = []
    for _, row in latest_wells.iterrows():
        well_no = row["Well No"]
        geometry = row.geometry
        wells_detail.append({
            "well_no": well_no,
            "latitude": float(geometry.y) if geometry is not None else None,
            "longitude": float(geometry.x) if geometry is not None else None,
            "village": _clean_opt(row.get("Village")),
            "district": _clean_opt(row.get("District")),
            "well_type": _clean_opt(row.get("Well Type")),
            "agency": _clean_opt(row.get("Agency")),
            "depth_of_well_m": None if pd.isna(row.get("Depth of Well")) else float(row.get("Depth of Well")),
            "latest_date": row["latest_date"].strftime("%Y-%m-%d") if pd.notna(row.get("latest_date")) else None,
            "latest_level_m_bgl": None if pd.isna(row.get("latest_level")) else float(row.get("latest_level")),
            "five_year_avg_m_bgl": None if pd.isna(row.get("five_year_avg")) else float(row.get("five_year_avg")),
            "change_5yr_m": None if pd.isna(row.get("change_5yr")) else float(row.get("change_5yr")),
            "trend": row.get("trend"),
            "trend_description": row.get("trend_description"),
            "n_5yr_observations": None if pd.isna(row.get("n_5yr")) else int(row.get("n_5yr")),
            "seasonal": [
                {
                    "year": year,
                    "pre_monsoon": well_seasonal.get(well_no, {}).get(year, {}).get("pre_monsoon"),
                    "monsoon": well_seasonal.get(well_no, {}).get(year, {}).get("monsoon"),
                    "observed_months": well_seasonal.get(well_no, {}).get(year, {}).get("observed_months", []),
                }
                for year in GW_SEASON_YEARS
            ],
        })

    seasonal_records = []
    for year in GW_SEASON_YEARS:
        entry = aoi_seasonal[year]
        pre, monsoon = entry["pre_monsoon"], entry["monsoon"]
        fluctuation = (pre["level_m_bgl"] - monsoon["level_m_bgl"]) if (pre and monsoon) else None
        seasonal_records.append({
            "year": year,
            "pre_monsoon": pre,
            "monsoon": monsoon,
            "observed_months": entry["observed_months"],
            "fluctuation_m": fluctuation,
        })

    summary = {
        "wells": int(len(latest_wells)),
        "latest_observation": str(latest_wells["latest_date"].max().date()),
        "latest_mean_depth_m_bgl": float(latest_wells["latest_level"].mean()),
        "latest_min_depth_m_bgl": float(latest_wells["latest_level"].min()),
        "latest_max_depth_m_bgl": float(latest_wells["latest_level"].max()),
        "five_year_mean_m_bgl": float(latest_wells["five_year_avg"].mean()),
        "five_year_mean_m_bgl_pooled": five_year_mean_m_bgl_pooled,
        "five_year_observation_count": five_year_observation_count,
        "seasonal_windows": {"pre_monsoon_months": PRE_MONSOON_MONTHS, "monsoon_months": MONSOON_MONTHS},
        "seasonal": seasonal_records,
        "wells_detail": wells_detail,
        "trend_counts": {k: int(v) for k, v in trend_counts.items()},
        "dominant_trend": max(trend_counts, key=trend_counts.get) if trend_counts else None,
        "mean_change_5yr_m": float(latest_wells["change_5yr"].mean()) if "change_5yr" in latest_wells.columns else None,
        "nearest_well_m": nearest_well_m,
        "nearest_well_depth_m_bgl": float(nearest_well["latest_level"]) if nearest_well is not None else None,
        "well_depth_range_m": (
            [
                None if pd.isna(latest_wells["Depth of Well"].min()) else float(latest_wells["Depth of Well"].min()),
                None if pd.isna(latest_wells["Depth of Well"].max()) else float(latest_wells["Depth of Well"].max()),
            ]
            if "Depth of Well" in latest_wells.columns else None
        ),
        "source": "CGWB manual monthly groundwater monitoring",
    }

    return {
        "dataset": "groundwater",
        "source": "CGWB manual monthly groundwater monitoring",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
        "observations": {
            "latest_wells_gdf": latest_wells,
            "well_seasonal": well_seasonal,
            "aoi_seasonal": aoi_seasonal,
            "summary": summary,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "high" if summary["wells"] >= 2 else "medium",
        "warnings": warnings,
        "limitations": limitations,
    }
