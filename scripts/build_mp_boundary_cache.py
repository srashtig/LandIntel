"""One-time offline precompute: dissolve + simplify the 230 MB
``vb_soi_mp.GeoJSON`` village-boundary file into a single coarse MP outline,
written to ``data_analysis_pipeline/mp_boundary.geojson``.

Not imported by the app — run manually:

    conda activate land_intel
    python scripts/build_mp_boundary_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_pipeline import config  # noqa: E402

VILLAGE_BOUNDARY_FILE = config.VILLAGE_BOUNDARY_FILE
OUTPUT_FILE = config.MP_BOUNDARY_CACHE

SIMPLIFY_TOLERANCE_DEG = 0.005  # ~500 m at MP's latitude


def main() -> None:
    """Build ``data_analysis_pipeline/mp_boundary.geojson`` from the full village-boundary
    file: dissolve all villages into one outline, reproject to EPSG:4326,
    simplify to ~500 m tolerance, and write it out. Run once (or whenever
    the source file changes); not imported by the app."""

    if not VILLAGE_BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Village boundary file not found: {VILLAGE_BOUNDARY_FILE}")

    print(f"Reading {VILLAGE_BOUNDARY_FILE} ...")
    start = time.time()
    gdf = gpd.read_file(VILLAGE_BOUNDARY_FILE, engine="pyogrio", columns=["state"])
    print(f"  {len(gdf):,} village polygons read in {time.time() - start:.1f}s (native CRS: {gdf.crs})")

    print("Dissolving into one outline ...")
    start = time.time()
    dissolved = gdf.dissolve()
    print(f"  dissolved in {time.time() - start:.1f}s")

    # The source file's native CRS (EPSG:7755) is a *projected*, metre-based
    # CRS, not degrees — despite what a naive reading of "tolerance=0.005"
    # might suggest. Reproject to EPSG:4326 first so the 0.005 tolerance
    # below is applied in degrees (~500 m at this latitude) as intended, and
    # so the cached boundary is directly usable for lat/lon `within()`
    # checks in `mp_boundary.is_in_mp` without any CRS conversion at lookup
    # time.
    dissolved = dissolved.to_crs("EPSG:4326")

    print(f"Simplifying (tolerance={SIMPLIFY_TOLERANCE_DEG} deg, EPSG:4326) ...")
    dissolved["geometry"] = dissolved.geometry.simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dissolved[["geometry"]].to_file(OUTPUT_FILE, driver="GeoJSON")

    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"Wrote {OUTPUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
