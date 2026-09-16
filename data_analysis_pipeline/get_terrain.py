"""SRTM elevation and slope. Ports notebook cells 39-40 and the terrain
portion of cell 60 (AOI reduceRegion statistics)."""

from __future__ import annotations

from datetime import datetime, timezone

from . import config
from .aoi import AOI, to_ee_geometry

SLOPE_CLASS_BOUNDS = [
    ("lt_3", "< 3° (nearly level)", 0, 3),
    ("3_8", "3–8° (gentle)", 3, 8),
    ("8_15", "8–15° (moderate)", 8, 15),
    ("ge_15", ">= 15° (steep)", 15, None),
]


def get_terrain(aoi: AOI) -> dict:
    """Compute SRTM-derived elevation and slope statistics for the AOI.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"elevation_image":
        ee.Image, "slope_image": ee.Image, "summary": {elevation_mean_m,
        elevation_min_m, elevation_max_m, slope_mean_deg, slope_median_deg
        (the AOI-wide 50th percentile), slope_min_deg, slope_max_deg,
        slope_range_90pct_deg ([p5, p95] — the middle-90%-of-pixels spread;
        a whole-AOI mean/median alone can look deceptively gentle while
        masking a much steeper sub-area), slope_median_deg_100m (median
        slope in just a 100 m radius around the exact site point — ~35-40
        SRTM pixels at 30 m resolution, a real but coarse local sample, and
        a "right here" figure distinct from the whole-AOI stats),
        slope_class_share: [{class, label,
        share_percent (0-100)}, ...], source}} — the images are for map
        tile layers, ``summary`` is the JSON-safe stats block (elevation in
        meters, slope in degrees).
    """

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)

    dem = ee.Image("USGS/SRTMGL1_003").clip(ee_aoi)
    elevation = dem.select("elevation")
    slope = ee.Terrain.slope(elevation)

    # mean/min/max plus the 5th/50th(median)/95th percentiles, all in one
    # reduceRegion call (one server round trip) — the AOI-wide mean alone
    # can look deceptively gentle while masking a much steeper sub-area (or
    # vice versa); the 5-95th range gives a sense of how much the slope
    # actually varies across the AOI, and the median is a more typical-point
    # figure than the mean when that spread is skewed.
    combined_reducer = (
        ee.Reducer.mean()
        .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
        .combine(reducer2=ee.Reducer.percentile([5, 50, 95]), sharedInputs=True)
    )

    terrain_values = ee.Image.cat([
        elevation.rename("elevation"),
        slope.rename("slope"),
    ]).reduceRegion(reducer=combined_reducer, geometry=ee_aoi, scale=30, maxPixels=1e13).getInfo()

    # Median slope in just a 100 m radius around the exact site point —
    # distinct from the whole-AOI stats above, which can be several km
    # across and average over terrain far from where the plot actually is.
    # At SRTM's native 30 m resolution a 100 m-radius circle covers ~35-40
    # pixels — a real local sample, but still a coarse "right here" reading
    # at 30 m resolution, not a precise local survey.
    local_point = ee.Geometry.Point([aoi.lon, aoi.lat]).buffer(100)
    local_slope_stats = slope.rename("slope").reduceRegion(
        reducer=ee.Reducer.percentile([50]), geometry=local_point, scale=30, maxPixels=1e13
    ).getInfo()

    slope_class_images = ee.Image.cat([
        (slope.gte(lo) if hi is None else slope.gte(lo).And(slope.lt(hi))).rename(key)
        for key, _label, lo, hi in SLOPE_CLASS_BOUNDS
    ])
    slope_classes = slope_class_images.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=ee_aoi, scale=30, maxPixels=1e13
    ).getInfo()

    summary = {
        "elevation_mean_m": terrain_values.get("elevation_mean"),
        "elevation_min_m": terrain_values.get("elevation_min"),
        "elevation_max_m": terrain_values.get("elevation_max"),
        "slope_mean_deg": terrain_values.get("slope_mean"),
        "slope_median_deg": terrain_values.get("slope_p50"),
        "slope_min_deg": terrain_values.get("slope_min"),
        "slope_max_deg": terrain_values.get("slope_max"),
        "slope_range_90pct_deg": [terrain_values.get("slope_p5"), terrain_values.get("slope_p95")],
        # A single-value percentile() reducer doesn't get a "_p50" suffix
        # (EE only disambiguates with a suffix when several percentiles are
        # requested together, as in combined_reducer above) — confirmed
        # directly, it's keyed by the plain band name instead.
        "slope_median_deg_100m": local_slope_stats.get("slope"),
        "slope_class_share": [
            {
                "class": key,
                "label": label,
                # EE returns a 0-1 fraction; stored as a 0-100 percent for
                # consistency with every other "share"/"percentage" field.
                "share_percent": (slope_classes.get(key) or 0) * 100,
            }
            for key, label, _lo, _hi in SLOPE_CLASS_BOUNDS
        ],
        "source": "SRTM 1 arc-second (USGS/SRTMGL1_003)",
    }

    return {
        "dataset": "terrain",
        "source": "USGS/SRTMGL1_003",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km, "scale_m": 30},
        "observations": {
            "elevation_image": elevation,
            "slope_image": slope,
            "summary": summary,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "30 m",
        "confidence": "high",
        "warnings": [],
        "limitations": [],
    }
