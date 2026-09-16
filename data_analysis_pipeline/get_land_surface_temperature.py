"""Landsat land-surface temperature + ERA5-Land air temperature.

Ports notebook cell 15 (LST composite), the LST portion of cell 60, and the
ERA5-Land air-temperature block also in cell 60 (same notebook cell as LST —
kept as two functions here per the module list, sharing this file).
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timedelta, timezone

from . import config
from .aoi import AOI, to_ee_geometry

CLOUD_COVER_MAX = 40
DEFAULT_LST_WINDOW_DAYS = 365
DEFAULT_AIR_TEMP_YEAR_SPAN = 3


def _default_lst_date_range() -> tuple[str, str]:
    """Trailing ``DEFAULT_LST_WINDOW_DAYS``-day window ending today (UTC),
    as "YYYY-MM-DD" strings — computed fresh on every call. A previous
    version hardcoded an absolute "2025-03-01"..."2026-03-31" window, which
    would have silently gone stale the further real time moved past it."""

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DEFAULT_LST_WINDOW_DAYS)
    return start.isoformat(), end.isoformat()


def _default_air_temp_years() -> list[int]:
    """The most recent ``DEFAULT_AIR_TEMP_YEAR_SPAN`` FULLY COMPLETED
    calendar years (excludes the current, possibly-partial year), computed
    from today's date rather than a fixed list — e.g. [2023, 2024, 2025]
    was previously hardcoded, which would have silently stopped being
    "recent" once real time moved past it."""

    current_year = datetime.now(timezone.utc).year
    return list(range(current_year - DEFAULT_AIR_TEMP_YEAR_SPAN, current_year))


def get_land_surface_temperature(aoi: AOI, start_date: str | None = None, end_date: str | None = None) -> dict:
    """Compute Landsat 8 C2 L2 land-surface temperature (median composite)
    for the AOI.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        start_date: Composite window start, "YYYY-MM-DD" (inclusive).
            Defaults to one year before ``end_date`` (see
            :func:`_default_lst_date_range`), recomputed relative to today
            on every call.
        end_date: Composite window end, "YYYY-MM-DD" (exclusive). Defaults
            to today (UTC).

    Returns:
        A standard envelope dict. ``observations`` = {"lst_image": ee.Image,
        "summary": {mean_c, min_c, max_c (degrees Celsius), scenes (Landsat
        scene count), source}}.
    """

    if start_date is None or end_date is None:
        default_start, default_end = _default_lst_date_range()
        start_date = start_date or default_start
        end_date = end_date or default_end

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)

    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(ee_aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", CLOUD_COVER_MAX))
    )

    def scale_lst(image):
        lst_k = image.select("ST_B10").multiply(0.00341802).add(149.0)
        return image.addBands(lst_k.subtract(273.15).rename("LST_C"))

    lst = landsat.map(scale_lst).select("LST_C").median().clip(ee_aoi)
    scene_count = landsat.size().getInfo()

    mean_min_max = ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
    lst_stats = lst.reduceRegion(reducer=mean_min_max, geometry=ee_aoi, scale=30, maxPixels=1e13).getInfo()

    summary = {
        "mean_c": lst_stats.get("LST_C_mean"),
        "min_c": lst_stats.get("LST_C_min"),
        "max_c": lst_stats.get("LST_C_max"),
        "scenes": scene_count,
        "source": "Landsat 8 C2 L2 ST_B10 (median composite)",
    }

    return {
        "dataset": "land_surface_temperature",
        "source": "LANDSAT/LC08/C02/T1_L2",
        "params": {
            "lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km,
            "start_date": start_date, "end_date": end_date, "cloud_cover_max": CLOUD_COVER_MAX,
        },
        "observations": {"lst_image": lst, "summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "30 m",
        "confidence": "high" if scene_count > 0 else "low",
        "warnings": [] if scene_count > 0 else ["No Landsat scenes matched the date/cloud filter."],
        "limitations": [],
    }


def get_air_temperature(aoi: AOI, years: list[int] | None = None) -> dict:
    """Compute ERA5-Land monthly 2 m air temperature statistics for the AOI,
    averaged over ``years``.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        years: Calendar years to average over, e.g. [2023, 2024, 2025].
            Defaults to the most recent :data:`DEFAULT_AIR_TEMP_YEAR_SPAN`
            fully completed years (see :func:`_default_air_temp_years`),
            recomputed relative to today on every call.

    Returns:
        A standard envelope dict. ``observations`` = {"summary":
        {annual_mean_c, coldest_month ("YYYY-MM"), coldest_month_c,
        warmest_month, warmest_month_c, months (count averaged), period,
        source}} (no map layer for this one, same as the notebook — ERA5-Land
        is reported as a statistic only). Temperatures in degrees Celsius.
    """

    if years is None:
        years = _default_air_temp_years()

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)

    era5_monthly = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(f"{min(years)}-01-01", f"{max(years) + 1}-01-01")
        .select("temperature_2m")
        .map(lambda img: img.subtract(273.15).set(
            "month_label", ee.Date(img.get("system:time_start")).format("YYYY-MM")
        ))
    )

    era5_features = era5_monthly.map(
        lambda img: ee.Feature(None, {
            "month": img.get("month_label"),
            "temp_c": img.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_aoi, scale=11132).get("temperature_2m"),
        })
    ).getInfo()["features"]

    era5_months = [
        (f["properties"]["month"], f["properties"]["temp_c"])
        for f in era5_features
        if f["properties"].get("temp_c") is not None
    ]

    warnings: list[str] = []
    if era5_months:
        coldest = min(era5_months, key=lambda item: item[1])
        warmest = max(era5_months, key=lambda item: item[1])
        summary = {
            "annual_mean_c": float(np.mean([t for _, t in era5_months])),
            "coldest_month": coldest[0],
            "coldest_month_c": coldest[1],
            "warmest_month": warmest[0],
            "warmest_month_c": warmest[1],
            "months": len(era5_months),
            "period": f"{min(years)}–{max(years)}",
            "source": "ERA5-Land monthly aggregated, 2 m air temperature",
        }
    else:
        warnings.append("ERA5-Land returned no monthly values for the requested period.")
        summary = {"source": "ERA5-Land unavailable"}

    return {
        "dataset": "air_temperature",
        "source": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km, "years": years},
        "observations": {"summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "~11 km",
        "confidence": "high" if era5_months else "low",
        "warnings": warnings,
        "limitations": [],
    }
