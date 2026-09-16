"""NDVI from Sentinel-2 SR Harmonized. Ports notebook cell 11 (composite)
and the NDVI portion of cell 60 (AOI reduceRegion statistics)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from . import config
from .aoi import AOI, to_ee_geometry


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _month_range_label(start_iso: str, span_months: int) -> str:
    """A human-readable "Mon-Mon" label for a ``span_months``-month period
    starting in ``start_iso``'s calendar month — e.g. ``("2025-12-10", 3)``
    -> "Dec-Feb". Only the month is used (day-of-month is irrelevant to a
    month-name label); wraps around year-end correctly.

    Args:
        start_iso: Period start, "YYYY-MM-DD" (only the month is read).
        span_months: Number of calendar months the period covers.

    Returns:
        "Mon" if ``span_months <= 1``, else "Mon-Mon".
    """

    _year, month, _day = (int(part) for part in start_iso.split("-"))
    start_idx = month - 1
    if span_months <= 1:
        return _MONTH_ABBR[start_idx]
    end_idx = (start_idx + span_months - 1) % 12
    return f"{_MONTH_ABBR[start_idx]}-{_MONTH_ABBR[end_idx]}"

CLOUD_PCT_MAX = 30
DEFAULT_WINDOW_DAYS = 365
PEAK_SCAN_SCALE_M = 30  # coarser than the 10 m final-stats scale — this is only a relative bucket-vs-bucket comparison to find which one is highest, not the final reported number
PEAK_SCAN_BUCKET_MONTHS = 3  # quarterly buckets, not monthly — cuts the peak search's server-side work ~3x (4 buckets/year instead of 12) while still reliably identifying which season is greenest


def _default_date_range() -> tuple[str, str]:
    """Trailing ``DEFAULT_WINDOW_DAYS``-day window ending today (UTC), as
    "YYYY-MM-DD" strings — computed fresh on every call, not a fixed
    constant, so the default composite period always reflects a full year
    of the most recent available imagery. A previous version hardcoded an
    absolute "2025-11-01".."2026-03-31" window, which would have silently
    gone stale (querying an ever-more-outdated 5-month slice) the further
    real time moved past it.

    Returns:
        ``(start_date, end_date)`` in "YYYY-MM-DD" form.
    """

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=DEFAULT_WINDOW_DAYS)
    return start.isoformat(), end.isoformat()


def _find_peak_period(ee, ee_aoi, start_date: str, end_date: str) -> Optional[dict]:
    """Find the ``PEAK_SCAN_BUCKET_MONTHS``-month period within
    ``[start_date, end_date)`` with the highest AOI-mean NDVI, via ONE
    batched Earth Engine call across every bucket in the window (mirrors
    the pattern
    :func:`~data_analysis_pipeline.get_land_surface_temperature.get_air_temperature`
    already uses for its monthly ERA5 series — map a per-bucket reduction
    over an ``ee.List`` of bucket-starts into an ``ee.FeatureCollection``,
    then a single ``.getInfo()`` for all buckets at once, rather than N
    separate round trips).

    This exists so "peak" isn't a guessed season (monsoon, winter, etc.)
    that may not actually be this AOI's greenest period (e.g. irrigated
    winter-wheat cropland peaks in Rabi season, not during the monsoon) —
    it's found directly from the data. Quarterly (not monthly) buckets keep
    this search's server-side cost down — see :data:`PEAK_SCAN_BUCKET_MONTHS`.

    Args:
        ee: The imported ``earthengine-api`` module.
        ee_aoi: AOI as an ``ee.Geometry``.
        start_date: Scan window start, "YYYY-MM-DD" (inclusive).
        end_date: Scan window end, "YYYY-MM-DD" (exclusive).

    Returns:
        ``{"period_start": "YYYY-MM-DD", "mean_ndvi": float, "scene_count": int}``
        for the highest-mean bucket, or None if no bucket had any usable
        scenes at all.
    """

    start = ee.Date(start_date)
    end = ee.Date(end_date)
    n_buckets = ee.Number(end.difference(start, "month")).divide(PEAK_SCAN_BUCKET_MONTHS).ceil().max(1)
    bucket_starts = ee.List.sequence(0, n_buckets.subtract(1)).map(
        lambda i: start.advance(ee.Number(i).multiply(PEAK_SCAN_BUCKET_MONTHS), "month")
    )

    def bucket_feature(bucket_start):
        bucket_start = ee.Date(bucket_start)
        bucket_end = bucket_start.advance(PEAK_SCAN_BUCKET_MONTHS, "month")
        coll = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ee_aoi)
            .filterDate(bucket_start, bucket_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_MAX))
        )
        scene_count = coll.size()

        def compute_mean():
            # Only ever evaluated server-side when scene_count > 0 (see the
            # ee.Algorithms.If below) — median() of an EMPTY collection has
            # no bands at all, so .get("NDVI") on its reduceRegion() result
            # would otherwise raise "Dictionary does not contain key: NDVI"
            # and abort the WHOLE batched .getInfo() call below, not just
            # this one bucket.
            ndvi_img = (
                coll.map(lambda img: img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI")))
                .select("NDVI")
                .median()
            )
            return ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ee_aoi, scale=PEAK_SCAN_SCALE_M, maxPixels=1e13
            ).get("NDVI")

        mean_val = ee.Algorithms.If(scene_count.gt(0), compute_mean(), None)
        return ee.Feature(None, {"period_start": bucket_start.format("YYYY-MM-dd"), "mean_ndvi": mean_val, "scene_count": scene_count})

    features = ee.FeatureCollection(bucket_starts.map(bucket_feature)).getInfo()["features"]
    buckets = [f["properties"] for f in features]
    valid = [b for b in buckets if b.get("mean_ndvi") is not None]
    if not valid:
        return None
    return max(valid, key=lambda b: b["mean_ndvi"])


def _ndvi_composite_stats(ee, ee_aoi, start_date: str, end_date: str) -> tuple["ee.Image", int, dict]:
    """Build a median-NDVI composite for one date window and reduce it to
    AOI-wide summary statistics. Shared by the whole-year and monsoon-season
    composites below so both go through identical logic.

    Args:
        ee: The imported ``earthengine-api`` module (avoids re-importing
            per call).
        ee_aoi: AOI as an ``ee.Geometry`` (see
            :func:`data_analysis_pipeline.aoi.to_ee_geometry`).
        start_date: Window start, "YYYY-MM-DD" (inclusive).
        end_date: Window end, "YYYY-MM-DD" (exclusive).

    Returns:
        ``(ndvi_image, scene_count, stats)`` — ``stats`` is
        ``{mean, min, max, share_ge_030_percent, share_ge_050_percent}``.
    """

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ee_aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_PCT_MAX))
    )

    def add_ndvi(img):
        return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

    ndvi = s2.map(add_ndvi).select("NDVI").median().clip(ee_aoi)
    scene_count = s2.size().getInfo()

    # Mean/min/max of NDVI itself, plus the two vegetation-threshold
    # fractions, all as bands of ONE image reduced in ONE reduceRegion call
    # — previously these were two separate reduceRegion().getInfo() calls
    # against the same year-long 10 m composite, which very likely made
    # Earth Engine evaluate that expensive median composite twice (once per
    # independent .getInfo() round trip) for no benefit. min/max of the two
    # 0/1 threshold bands are computed too (harmless/cheap) but unused
    # below — the point is collapsing this to a single server round trip.
    mean_min_max = ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
    combined = ee.Image.cat([
        ndvi.rename("NDVI"),
        ndvi.gte(0.3).rename("frac_ndvi_ge_030"),
        ndvi.gte(0.5).rename("frac_ndvi_ge_050"),
    ])
    combined_stats = combined.reduceRegion(
        reducer=mean_min_max, geometry=ee_aoi, scale=10, maxPixels=1e13
    ).getInfo()

    stats = {
        "mean": combined_stats.get("NDVI_mean"),
        "min": combined_stats.get("NDVI_min"),
        "max": combined_stats.get("NDVI_max"),
        # 0-1 fractions from EE, stored as 0-100 percent for consistency
        # with every other share/percentage field in the summary.
        "share_ge_030_percent": (combined_stats.get("frac_ndvi_ge_030_mean") or 0) * 100,
        "share_ge_050_percent": (combined_stats.get("frac_ndvi_ge_050_mean") or 0) * 100,
    }
    return ndvi, scene_count, stats


def get_sentinel_features(aoi: AOI, start_date: str | None = None, end_date: str | None = None) -> dict:
    """Compute NDVI (median composite) statistics for the AOI from
    Sentinel-2 — TWO numbers, not one: a whole-year average and the peak
    (greenest) period found within that year. A single whole-year median
    understates peak vegetation vigor (it's dragged down by fallow/dry-season
    bareness on seasonal cropland); a single guessed "season" (e.g. monsoon)
    can easily miss the actual peak — irrigated winter-wheat cropland, for
    instance, peaks in the Rabi season, not during the June-Sept monsoon.
    So "peak" here is found directly from the data (see
    :func:`_find_peak_period`, scanned in
    :data:`PEAK_SCAN_BUCKET_MONTHS`-month buckets) rather than assumed from
    a calendar convention, and reported alongside the whole-year average.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.
        start_date: Whole-year window start, "YYYY-MM-DD" (inclusive).
            Defaults to one year before ``end_date`` (see
            :func:`_default_date_range`), recomputed relative to today on
            every call rather than a fixed date. The peak search covers
            this same window.
        end_date: Whole-year window end, "YYYY-MM-DD" (exclusive).
            Defaults to today (UTC).

    Returns:
        A standard envelope dict. ``observations`` = {"ndvi_image": ee.Image
        (the whole-year composite — used for the map layer), "scene_count":
        int (whole-year), "summary": {mean, min, max (NDVI, unitless,
        -1..1), share_ge_030_percent, share_ge_050_percent (0-100), period,
        source, peak: {mean, period (the identified peak
        ``PEAK_SCAN_BUCKET_MONTHS``-month period), scene_count, source} |
        None (None only if every bucket in the window had zero usable
        scenes) — deliberately just {mean, period, scene_count}, no
        min/max/share breakdown: those would need a second full-resolution
        composite of the peak period purely to compute, which was the
        actual cost driver, for a modest amount of extra detail}}.
    """

    if start_date is None or end_date is None:
        default_start, default_end = _default_date_range()
        start_date = start_date or default_start
        end_date = end_date or default_end

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)

    ndvi, scene_count, stats = _ndvi_composite_stats(ee, ee_aoi, start_date, end_date)

    warnings = []
    if scene_count == 0:
        warnings.append("No Sentinel-2 scenes matched the whole-year date/cloud filter.")

    peak_period = _find_peak_period(ee, ee_aoi, start_date, end_date)
    if peak_period is None:
        warnings.append("No period in the whole-year window had any usable Sentinel-2 scenes for a peak search.")
        peak_summary = None
    else:
        # Reuse the scan's own mean directly rather than re-compositing this
        # period a second time at full (10 m) resolution for min/max/share —
        # that second pass was the actual bottleneck (~11s to recomposite
        # ~13 scenes) for very little extra information; peak's job is to
        # answer "which period, and roughly how green," not to duplicate the
        # whole-year summary's full breakdown at higher precision.
        peak_start = peak_period["period_start"]
        peak_summary = {
            "mean": peak_period["mean_ndvi"],
            "period": _month_range_label(peak_start, PEAK_SCAN_BUCKET_MONTHS),
            "scene_count": peak_period["scene_count"],
            "source": (
                f"Sentinel-2 SR Harmonized (median NDVI, peak {PEAK_SCAN_BUCKET_MONTHS}-month period found "
                f"within the year; mean from the {PEAK_SCAN_SCALE_M} m scan, no separate min/max/share breakdown)"
            ),
        }

    summary = {
        **stats,
        "period": f"{start_date} → {end_date}",
        "source": "Sentinel-2 SR Harmonized (median NDVI, whole year)",
        "peak": peak_summary,
    }

    return {
        "dataset": "sentinel_ndvi",
        "source": "COPERNICUS/S2_SR_HARMONIZED",
        "params": {
            "lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km,
            "start_date": start_date, "end_date": end_date, "cloud_pct_max": CLOUD_PCT_MAX,
        },
        "observations": {"ndvi_image": ndvi, "scene_count": scene_count, "summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "10 m",
        "confidence": "high" if scene_count > 0 and peak_summary is not None else "low",
        "warnings": warnings,
        "limitations": [],
    }
