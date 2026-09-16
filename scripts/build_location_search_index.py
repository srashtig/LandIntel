"""One-time offline precompute: a flat name/level/centroid table for
village, subdistrict and district search, from ``vb_soi_mp.GeoJSON``.

Villages are indexed individually (one centroid each); subdistricts and
districts are dissolved first so e.g. "Dewas district" resolves to a single
point rather than one row per village in that district.

Writes ``data_analysis_pipeline/mp_location_index.parquet`` with columns
``name, level, centroid_lat, centroid_lon, district, subdistric``.

Not imported by the app — run manually:

    conda activate land_intel
    python scripts/build_location_search_index.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_pipeline import config  # noqa: E402

VILLAGE_BOUNDARY_FILE = config.VILLAGE_BOUNDARY_FILE
OUTPUT_FILE = config.LOCATION_SEARCH_INDEX

COLUMNS = ["village", "subdistric", "district"]


def _centroid_rows(gdf: gpd.GeoDataFrame, name_col: str, level: str) -> pd.DataFrame:
    """Build search-index rows (one per feature) with each feature's
    centroid reprojected to lat/lon.

    Args:
        gdf: Features to index (villages, or dissolved subdistricts/districts).
        name_col: Column in ``gdf`` holding the display name for each row.
        level: The "level" value to stamp on every row ("village" |
            "subdistrict" | "district").

    Returns:
        A DataFrame with columns name, level, centroid_lat, centroid_lon,
        district, subdistric. Centroid computed in the source's native
        projected (metric) CRS for accuracy, then the centroid points
        themselves reprojected to EPSG:4326 for lat/lon output.
    """

    centroids_native = gdf.geometry.centroid
    centroids = gpd.GeoSeries(centroids_native, crs=gdf.crs).to_crs("EPSG:4326")
    return pd.DataFrame({
        "name": gdf[name_col].astype(str).str.strip(),
        "level": level,
        "centroid_lat": centroids.y.values,
        "centroid_lon": centroids.x.values,
        "district": gdf["district"].astype(str).str.strip() if "district" in gdf.columns else None,
        "subdistric": gdf["subdistric"].astype(str).str.strip() if "subdistric" in gdf.columns else None,
    })


def main() -> None:
    """Build ``data_analysis_pipeline/mp_location_index.parquet``: one centroid row per
    village, plus one per dissolved subdistrict and one per dissolved
    district, for :func:`data_analysis_pipeline.location_search.search_places`.
    Run once (or whenever the source file changes); not imported by the app."""

    if not VILLAGE_BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Village boundary file not found: {VILLAGE_BOUNDARY_FILE}")

    print(f"Reading {VILLAGE_BOUNDARY_FILE} ...")
    start = time.time()
    gdf = gpd.read_file(VILLAGE_BOUNDARY_FILE, engine="pyogrio", columns=COLUMNS)
    print(f"  {len(gdf):,} village polygons read in {time.time() - start:.1f}s")

    print("Computing village centroids ...")
    village_rows = _centroid_rows(gdf, "village", "village")

    # Dissolve by (district, subdistric) TOGETHER, not subdistric alone: 5
    # subdistrict names in this dataset (Deori, Huzur, Shahpura, Sohagpur,
    # Tendukheda) each occur in TWO different districts (e.g. "Deori" exists
    # in both Raisen and Sagar) — dissolving by subdistric alone would
    # silently merge those two genuinely different places into one row with
    # a bogus blended centroid. District names have no such collisions (all
    # 52 are unique), which is why this bug only showed up at the
    # subdistrict level.
    print("Dissolving by (district, subdistrict) ...")
    subdistrict_gdf = gdf.dissolve(by=["district", "subdistric"]).reset_index()
    subdistrict_rows = _centroid_rows(subdistrict_gdf, "subdistric", "subdistrict")

    print("Dissolving by district ...")
    district_gdf = gdf.dissolve(by="district").reset_index()
    district_rows = _centroid_rows(district_gdf, "district", "district")
    district_rows["district"] = district_rows["name"]

    index = pd.concat([village_rows, subdistrict_rows, district_rows], ignore_index=True)
    index = index.dropna(subset=["name"])
    index = index[index["name"].str.strip() != ""]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(OUTPUT_FILE, index=False)

    print(f"Wrote {OUTPUT_FILE} ({len(index):,} rows, {OUTPUT_FILE.stat().st_size / 1e6:.2f} MB)")
    print(index["level"].value_counts())


if __name__ == "__main__":
    main()
