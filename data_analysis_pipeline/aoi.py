"""Area-of-interest construction.

Ports the notebook's cell 2-3 logic (buffer in EPSG:3857, back to EPSG:4326)
plus the UTM-CRS / padded-native-CRS-bbox logic used by every large local
vector reader (notebook cell 27 / 55).
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import Point, Polygon

# Large local vector datasets (vb_soi_mp, river_polygon, wb_sac_mp) are
# stored in EPSG:7755. Padding the query window beyond the AOI radius avoids
# edge effects when computing "nearest feature" distances.
VECTOR_SOURCE_CRS = "EPSG:7755"
NATIVE_QUERY_PAD_M = 2000.0


@dataclass
class AOI:
    lat: float
    lon: float
    radius_km: float

    center_4326: Point  # Point(lon, lat), EPSG:4326
    polygon_4326: Polygon  # buffered AOI polygon, EPSG:4326
    aoi_gdf: gpd.GeoDataFrame  # one-row GeoDataFrame of polygon_4326, EPSG:4326

    utm_crs: object  # pyproj CRS, via aoi_gdf.estimate_utm_crs()
    center_utm: Point
    polygon_utm: Polygon
    area_km2: float

    # Padded bbox (minx, miny, maxx, maxy) in VECTOR_SOURCE_CRS, for
    # bbox-filtered reads of the large local GeoJSON datasets.
    native_bbox: tuple


def build_aoi(lat: float, lon: float, radius_km: float = 5.0) -> AOI:
    """Build the circular area-of-interest every ``get_*`` module takes as
    input. This is the required first call of any analysis.

    Reproduces the notebook's AOI geometry exactly: (1) buffer a point in
    EPSG:3857 and reproject back to EPSG:4326 (notebook cell 2-3); (2) a
    local UTM CRS via ``estimate_utm_crs()`` for metric calculations
    (notebook cell 27/55, ``UTM_CRS``); (3) a padded bbox in EPSG:7755 for
    bbox-filtered reads of the large village/river/water-body GeoJSON files
    (notebook cell 27).

    Args:
        lat: Site latitude in decimal degrees (WGS84 / EPSG:4326).
        lon: Site longitude in decimal degrees (WGS84 / EPSG:4326).
        radius_km: Radius of the circular area of interest, in kilometers,
            centered on (lat, lon). Defaults to 5.0.

    Returns:
        An :class:`AOI` with the circle in both EPSG:4326 and the local UTM
        CRS, the computed area in km², and a padded EPSG:7755 bbox for
        reading the large local vector files.
    """

    center_4326 = Point(lon, lat)

    point = gpd.GeoSeries([center_4326], crs="EPSG:4326")
    point_m = point.to_crs("EPSG:3857")
    aoi_m = point_m.iloc[0].buffer(radius_km * 1000)
    polygon_4326 = gpd.GeoSeries([aoi_m], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]

    aoi_gdf = gpd.GeoDataFrame({"name": ["AOI"]}, geometry=[polygon_4326], crs="EPSG:4326")

    utm_crs = aoi_gdf.estimate_utm_crs()
    center_utm = gpd.GeoSeries([center_4326], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
    polygon_utm = aoi_gdf.to_crs(utm_crs).geometry.iloc[0]
    area_km2 = polygon_utm.area / 1e6

    native_point = gpd.GeoSeries([center_4326], crs="EPSG:4326").to_crs(VECTOR_SOURCE_CRS).iloc[0]
    native_geom = native_point.buffer(radius_km * 1000 + NATIVE_QUERY_PAD_M)
    native_bbox = native_geom.bounds

    return AOI(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        center_4326=center_4326,
        polygon_4326=polygon_4326,
        aoi_gdf=aoi_gdf,
        utm_crs=utm_crs,
        center_utm=center_utm,
        polygon_utm=polygon_utm,
        area_km2=area_km2,
        native_bbox=native_bbox,
    )


def to_ee_geometry(aoi: AOI):
    """Convert an AOI's polygon to an Earth Engine geometry for use in GEE
    ``reduceRegion``/``filterBounds`` calls (notebook cell 10). Initializes
    Earth Engine on first use if not already done.

    Args:
        aoi: The AOI to convert, from :func:`build_aoi`.

    Returns:
        An ``ee.Geometry.Polygon`` matching ``aoi.polygon_4326``.
    """

    import ee

    from . import config

    config.init_earth_engine()
    return ee.Geometry.Polygon(list(aoi.polygon_4326.exterior.coords))


def read_local_vector(path, columns, aoi: AOI, source_crs: str = VECTOR_SOURCE_CRS) -> gpd.GeoDataFrame:
    """Read only the rows of a large local GeoJSON file that fall within the
    AOI's padded bounding box, without loading the whole (often 100s of MB)
    file into memory.

    Mirrors the notebook's ``read_local_vector`` helper (cell 27): rows are
    filtered server-side by ``aoi.native_bbox`` (pyogrio's bbox push-down),
    then the result is reprojected to EPSG:4326.

    Args:
        path: Path to the GeoJSON file to read.
        columns: Attribute column names to read (geometry is always included).
        aoi: The AOI whose ``native_bbox`` (in ``source_crs``) bounds the read.
        source_crs: The file's native CRS, matching how ``aoi.native_bbox``
            was computed. Defaults to ``VECTOR_SOURCE_CRS`` (EPSG:7755),
            the CRS of every large dataset this pipeline reads this way.

    Returns:
        A GeoDataFrame in EPSG:4326 with the requested columns, empty (but
        correctly typed) if ``path`` does not exist.
    """

    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.read_file(
        path,
        bbox=aoi.native_bbox,
        engine="pyogrio",
        columns=list(columns),
    ).to_crs("EPSG:4326")
