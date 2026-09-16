"""Rainfall — CHIRPS v3 daily (AOI mean, Earth Engine) + IMD 0.25° gridded
daily (nearest grid cell, local NetCDF).

Ports notebook cells 41-45. CHIRPS and IMD are kept side by side rather than
blended — their agreement is itself the confidence signal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .aoi import AOI, to_ee_geometry

MONSOON_MONTHS = [6, 7, 8, 9]

RAIN_DAY_THRESHOLD = 1.0
HEAVY_25_THRESHOLD = 25.0
HEAVY_50_THRESHOLD = 50.0
HEAVY_100_THRESHOLD = 100.0

RAIN_METRIC_COLS = [
    "annual_mm", "monsoon_mm", "rain_days",
    "days_ge_25mm", "days_ge_50mm", "days_ge_100mm", "max_1day_mm",
]

# Depth-in-mm fields that get an inch equivalent added alongside them (see
# _add_inches below) — rain_days/days_ge_*mm are day COUNTS, not depths, so
# they're left alone; the "mm" in their name is just the threshold's name.
DEPTH_METRIC_COLS = ["annual_mm", "monsoon_mm", "max_1day_mm"]
MM_PER_INCH = 25.4


def _default_rain_years() -> list[int]:
    """Years to compute rainfall metrics for by default: every year that
    has a locally downloaded IMD NetCDF file (see
    ``config.IMD_RAINFALL_FILES``), currently [2023, 2024, 2025].

    Deliberately NOT a rolling "most recent N years" window the way
    :func:`~data_analysis_pipeline.get_land_surface_temperature._default_air_temp_years`
    is: IMD here is a fixed set of manually-downloaded local files, not a
    live-updating cloud dataset like CHIRPS or ERA5-Land — a future year
    only becomes usable once someone downloads and registers its file in
    ``config.IMD_RAINFALL_FILES``, so the default should track that
    dict's actual keys, not the calendar. Deriving it from the dict (rather
    than a separate hardcoded list) means adding a new year's file to
    config.py is the only change needed to extend this later.
    """

    return sorted(config.IMD_RAINFALL_FILES.keys())


def _add_inches(metrics: dict | None) -> dict | None:
    """Add an "_in" (inches) sibling field for each mm depth metric in
    ``metrics`` (see :data:`DEPTH_METRIC_COLS`) — day-count fields
    (rain_days, days_ge_*mm) are left untouched, since they're counts, not
    depths. mm stays the authoritative/primary unit (existing consumers,
    e.g. the decision engine, read the ``_mm`` fields); ``_in`` is added
    for report/chat display, since many readers find inches more familiar
    than millimeters.

    Args:
        metrics: A per-year or mean rainfall metrics dict (this module's
            CHIRPS or IMD shape), or None.

    Returns:
        A new dict with ``_in`` siblings added, or None unchanged.
    """

    if metrics is None:
        return None
    enriched = dict(metrics)
    for key in DEPTH_METRIC_COLS:
        mm_value = metrics.get(key)
        enriched[key.replace("_mm", "_in")] = None if mm_value is None else mm_value / MM_PER_INCH
    return enriched


def _find_coord(ds, candidates):
    """Find the first coordinate/dimension name in ``ds`` matching one of
    ``candidates`` (exact match first, then substring), or None."""

    names = list(ds.coords) + list(ds.dims)
    for candidate in candidates:
        for name in names:
            if name.lower() == candidate.lower():
                return name
    for candidate in candidates:
        for name in names:
            if candidate.lower() in name.lower():
                return name
    return None


def _find_rainfall_variable(ds):
    """Find the data variable in ``ds`` holding rainfall values, by name
    match or by being the sole data variable. Returns None if ambiguous."""

    candidates = ["rainfall", "rain", "rf", "precipitation", "precip", "pr"]
    variables = list(ds.data_vars)
    for candidate in candidates:
        for var in variables:
            if var.lower() == candidate.lower():
                return var
    for candidate in candidates:
        for var in variables:
            if candidate.lower() in var.lower():
                return var
    if len(variables) == 1:
        return variables[0]
    return None


def _imd_structure(path):
    """Detect the (lat, lon, time, rainfall) coordinate/variable names in
    an IMD NetCDF file. Raises ValueError if any cannot be identified."""

    with xr.open_dataset(path) as ds:
        structure = (
            _find_coord(ds, ["lat", "latitude", "y"]),
            _find_coord(ds, ["lon", "longitude", "x"]),
            _find_coord(ds, ["time", "date", "datetime"]),
            _find_rainfall_variable(ds),
        )
    if not all(structure):
        raise ValueError(f"Could not identify IMD coordinates/variable in {path}: lat/lon/time/rain = {structure}")
    return structure


def _load_imd_year(lat, lon, year, path):
    """Load daily rainfall for ``year`` at the IMD grid cell nearest (lat, lon).

    Returns:
        ``(df, grid_lat, grid_lon)`` — ``df`` has "date", "year", "month",
        "rainfall_mm" columns (negative fill values converted to NaN and
        dropped); ``grid_lat``/``grid_lon`` are the actual grid-cell center.
    """

    lat_name, lon_name, time_name, rain_name = _imd_structure(path)

    with xr.open_dataset(path) as ds:
        cell = ds[rain_name].sel({lat_name: lat, lon_name: lon}, method="nearest")
        grid_lat = float(cell[lat_name].values)
        grid_lon = float(cell[lon_name].values)
        df = cell.to_dataframe(name="rainfall_mm").reset_index()

    df["date"] = pd.to_datetime(df[time_name])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
    df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = np.nan
    df = df[df["year"] == year].dropna(subset=["rainfall_mm"]).copy()
    return df, grid_lat, grid_lon


def _imd_year_metrics(df: pd.DataFrame, year: int) -> dict:
    """Compute one year's IMD rainfall metrics (annual/monsoon totals, rain-day
    counts at each threshold, max 1-day) from that year's daily ``df``."""

    if df.empty:
        return {"year": year, **{col: np.nan for col in RAIN_METRIC_COLS}}
    rain = df["rainfall_mm"]
    return {
        "year": year,
        "annual_mm": rain.sum(),
        "monsoon_mm": df.loc[df["month"].isin(MONSOON_MONTHS), "rainfall_mm"].sum(),
        "rain_days": int((rain >= RAIN_DAY_THRESHOLD).sum()),
        "days_ge_25mm": int((rain >= HEAVY_25_THRESHOLD).sum()),
        "days_ge_50mm": int((rain >= HEAVY_50_THRESHOLD).sum()),
        "days_ge_100mm": int((rain >= HEAVY_100_THRESHOLD).sum()),
        "max_1day_mm": rain.max(),
    }


def _metrics_by_year(df: pd.DataFrame) -> dict:
    """Convert a per-year metrics DataFrame (one row per year, RAIN_METRIC_COLS
    columns) into {year (int): {metric: value | None}}."""

    return {
        int(row["year"]): {
            col: (float(row[col]) if pd.notna(row[col]) else None) for col in RAIN_METRIC_COLS
        }
        for _, row in df.iterrows()
    }


def _difference_pct(chirps_value, imd_value):
    """Percent difference of CHIRPS vs. IMD for one metric/year: 100 * (chirps
    - imd) / |imd|, or None if either value is missing/zero-denominator."""

    if chirps_value is None or imd_value in (None, 0) or (imd_value is not None and pd.isna(imd_value)):
        return None
    return 100.0 * (chirps_value - imd_value) / abs(imd_value)


def _mean_of(source: dict, metric: str, years):
    """Mean of one metric across ``years`` in a ``_metrics_by_year``-shaped
    dict, ignoring years where that metric is missing. None if all missing."""

    values = [source[year][metric] for year in years if source.get(year, {}).get(metric) is not None]
    return float(np.mean(values)) if values else None


def get_rainfall(aoi: AOI, years: list[int] | None = None) -> dict:
    """Compute CHIRPS v3 (AOI mean, Earth Engine) and IMD 0.25° (nearest grid
    cell, local NetCDF) rainfall metrics for the AOI, per year. The two
    datasets are kept side by side rather than blended.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        years: Calendar years to compute metrics for, e.g. [2023, 2024, 2025].
            Defaults to the most recent :data:`DEFAULT_RAIN_YEAR_SPAN` fully
            completed years (see :func:`_default_rain_years`), recomputed
            relative to today on every call. IMD files must exist locally
            for each year (see ``config.IMD_RAINFALL_FILES``) or that
            year's IMD metrics are None.

    Returns:
        A standard envelope dict. ``observations`` = {"summary": {years,
        monsoon_months, thresholds_mm, thresholds_in, yearly: [{year,
        chirps: {annual_mm, annual_in, monsoon_mm, monsoon_in, rain_days,
        days_ge_25/50/100mm, max_1day_mm, max_1day_in}, imd: {...} | None,
        diff_percent: {...}}, ...], means: {chirps: {...}, imd: {...}},
        wettest_day_mm: {chirps, imd}, wettest_day_in: {chirps, imd},
        imd_grid: [{year, latitude, longitude}, ...], notes, source}} —
        JSON-safe only (no map layer objects; the notebook's rainfall
        raster layers were commented out). Depths given in both millimeters
        (authoritative/primary — matches the source datasets' native unit)
        and inches (``_in`` siblings, for display) — day-count fields
        (rain_days, days_ge_*mm) have no inch equivalent, they're counts.
    """

    if years is None:
        years = _default_rain_years()

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)
    warnings: list[str] = []

    chirps_daily = (
        ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_RNL")
        .filterBounds(ee_aoi)
        .filterDate(f"{min(years)}-01-01", f"{max(years) + 1}-01-01")
        .select("precipitation")
    )

    def chirps_year_metrics(year):
        start = ee.Date.fromYMD(year, 1, 1)
        daily = chirps_daily.filterDate(start, start.advance(1, "year"))

        def days_above(threshold):
            return daily.map(lambda img: img.gte(threshold)).sum()

        metrics = ee.Image.cat([
            daily.sum().rename("annual_mm"),
            daily.filter(
                ee.Filter.calendarRange(min(MONSOON_MONTHS), max(MONSOON_MONTHS), "month")
            ).sum().rename("monsoon_mm"),
            days_above(RAIN_DAY_THRESHOLD).rename("rain_days"),
            days_above(HEAVY_25_THRESHOLD).rename("days_ge_25mm"),
            days_above(HEAVY_50_THRESHOLD).rename("days_ge_50mm"),
            days_above(HEAVY_100_THRESHOLD).rename("days_ge_100mm"),
            daily.max().rename("max_1day_mm"),
        ])
        values = metrics.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_aoi, scale=5566, maxPixels=1e13)
        return ee.Feature(None, values.set("year", year))

    chirps_fc = ee.FeatureCollection([chirps_year_metrics(y) for y in years])
    chirps_metrics = pd.DataFrame([f["properties"] for f in chirps_fc.getInfo()["features"]])
    chirps_metrics = chirps_metrics[["year"] + RAIN_METRIC_COLS].sort_values("year").reset_index(drop=True)
    chirps_by_year = _metrics_by_year(chirps_metrics)

    imd_daily, imd_grid_locations = {}, {}
    for year in years:
        path = config.IMD_RAINFALL_FILES.get(year)
        if path is None or not path.exists():
            warnings.append(f"IMD file missing for {year}: {path}")
            imd_daily[year] = pd.DataFrame(columns=["date", "year", "month", "rainfall_mm"])
            imd_grid_locations[year] = (None, None)
            continue
        df, grid_lat, grid_lon = _load_imd_year(aoi.lat, aoi.lon, year, path)
        imd_daily[year] = df
        imd_grid_locations[year] = (grid_lat, grid_lon)

    imd_metrics = pd.DataFrame([_imd_year_metrics(imd_daily[y], y) for y in years])
    imd_by_year = _metrics_by_year(imd_metrics)

    yearly = [
        {
            "year": year,
            "chirps": _add_inches(chirps_by_year.get(year)),
            "imd": _add_inches(imd_by_year.get(year)),
            "diff_percent": {
                metric: _difference_pct(chirps_by_year.get(year, {}).get(metric), imd_by_year.get(year, {}).get(metric))
                for metric in RAIN_METRIC_COLS
            },
        }
        for year in years
    ]

    # Wettest single day across the whole period — printed by the notebook
    # (cell 45) but never stored in the summary dict.
    chirps_max_days = [chirps_by_year[y]["max_1day_mm"] for y in years if chirps_by_year.get(y, {}).get("max_1day_mm") is not None]
    imd_max_days = [imd_by_year[y]["max_1day_mm"] for y in years if imd_by_year.get(y, {}).get("max_1day_mm") is not None]

    summary = {
        "years": years,
        "monsoon_months": MONSOON_MONTHS,
        "thresholds_mm": {
            "rain_day": RAIN_DAY_THRESHOLD, "heavy_25": HEAVY_25_THRESHOLD,
            "heavy_50": HEAVY_50_THRESHOLD, "heavy_100": HEAVY_100_THRESHOLD,
        },
        "thresholds_in": {
            "rain_day": RAIN_DAY_THRESHOLD / MM_PER_INCH, "heavy_25": HEAVY_25_THRESHOLD / MM_PER_INCH,
            "heavy_50": HEAVY_50_THRESHOLD / MM_PER_INCH, "heavy_100": HEAVY_100_THRESHOLD / MM_PER_INCH,
        },
        "yearly": yearly,
        "means": {
            "chirps": _add_inches({metric: _mean_of(chirps_by_year, metric, years) for metric in RAIN_METRIC_COLS}),
            "imd": _add_inches({metric: _mean_of(imd_by_year, metric, years) for metric in RAIN_METRIC_COLS}),
        },
        "wettest_day_mm": {
            "chirps": max(chirps_max_days) if chirps_max_days else None,
            "imd": max(imd_max_days) if imd_max_days else None,
        },
        "wettest_day_in": {
            "chirps": max(chirps_max_days) / MM_PER_INCH if chirps_max_days else None,
            "imd": max(imd_max_days) / MM_PER_INCH if imd_max_days else None,
        },
        "imd_grid": [
            {"year": year, "latitude": imd_grid_locations[year][0], "longitude": imd_grid_locations[year][1]}
            for year in years
        ],
        "notes": [
            f"CHIRPS v3 daily (UCSB-CHC/CHIRPS/V3/DAILY_RNL), AOI mean over {aoi.radius_km} km, native ~5.5 km grid.",
            "IMD 0.25° gridded daily rainfall at the grid cell nearest the site.",
            "diff_percent = 100 x (CHIRPS - IMD) / |IMD| — a diagnostic, not a claim that either dataset is ground truth.",
        ],
        "source": "CHIRPS v3 daily + IMD 0.25° gridded daily",
    }

    return {
        "dataset": "rainfall",
        "source": "UCSB-CHC/CHIRPS/V3/DAILY_RNL + IMD 0.25° gridded daily",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km, "years": years},
        "observations": {"summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "CHIRPS ~5.5 km; IMD 0.25°",
        "confidence": "high" if not warnings else "medium",
        "warnings": warnings,
        "limitations": [],
    }
