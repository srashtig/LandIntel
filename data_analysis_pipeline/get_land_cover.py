"""ESA WorldCover class areas. Ports notebook cells 12-14 (class area
reduction) and the land-cover portion of cell 61 (dominant class, AOI area)."""

from __future__ import annotations

from datetime import datetime, timezone

from . import config
from .aoi import AOI, to_ee_geometry

CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow / ice",
    80: "Permanent water", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss / lichen",
}


def get_land_cover(aoi: AOI) -> dict:
    """Compute the ESA WorldCover v200 land-cover class-area breakdown for
    the AOI (e.g. cropland/built-up/tree-cover shares).

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"worldcover_image":
        ee.Image, "summary": {classes: [{name, area_km2, percentage
        (0-100, sums to ~100 across classes)}, ...], aoi_area_km2,
        dominant_class, source}}.
    """

    config.init_earth_engine()
    import ee

    ee_aoi = to_ee_geometry(aoi)

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(ee_aoi)
    class_names_dict = ee.Dictionary({str(k): v for k, v in CLASS_NAMES.items()})

    pixel_area = ee.Image.pixelArea()
    area_image = pixel_area.addBands(worldcover)

    class_areas = area_image.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
        geometry=ee_aoi, scale=10, maxPixels=1e13,
    )

    groups = ee.List(class_areas.get("groups"))
    total_classified_area = groups.map(lambda x: ee.Dictionary(x).get("sum")).reduce(ee.Reducer.sum())

    def format_class(item):
        item = ee.Dictionary(item)
        class_id = ee.Number(item.get("class"))
        area_m2 = ee.Number(item.get("sum"))
        return ee.Dictionary({
            "class_id": class_id,
            "class_name": class_names_dict.get(class_id.format()),
            "area_km2": area_m2.divide(1e6),
            "percentage": area_m2.divide(total_classified_area).multiply(100),
        })

    class_summary = groups.map(format_class).getInfo()
    class_summary = sorted(class_summary, key=lambda x: x["percentage"], reverse=True)

    summary = {
        "classes": [
            {"name": item["class_name"], "area_km2": item["area_km2"], "percentage": item["percentage"]}
            for item in class_summary
        ],
        "aoi_area_km2": aoi.area_km2,
        "dominant_class": class_summary[0]["class_name"] if class_summary else None,
        "source": "ESA WorldCover v200 (10 m)",
    }

    return {
        "dataset": "land_cover",
        "source": "ESA/WorldCover/v200",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
        "observations": {"worldcover_image": worldcover, "summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": "10 m",
        "confidence": "high",
        "warnings": [],
        "limitations": [],
    }
