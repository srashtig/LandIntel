"""CLI entrypoint.

    python -m data_analysis_pipeline.main --lat 23.0401972 --lon 76.2086806 \\
        --radius 5 --out-dir runs/manual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .orchestrator import run_analysis


def _json_safe(value):
    """``json.dump(..., default=_json_safe)`` handler for numpy/pandas
    scalar types that ``site_summary`` may still contain.

    Args:
        value: A non-JSON-native value encountered while serializing.

    Returns:
        A JSON-serializable equivalent (int/float/bool/ISO datetime string),
        or ``str(value)`` as a last resort.
    """

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def main() -> None:
    """CLI entrypoint: parse ``--lat/--lon/--radius/--out-dir``, run
    :func:`data_analysis_pipeline.orchestrator.run_analysis`, and write
    ``site_summary.json``, ``map.html`` and ``run.log`` to ``--out-dir``."""

    parser = argparse.ArgumentParser(description="Run the Land Intelligence pipeline for one site.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--radius", type=float, default=5.0, help="AOI radius in km (default 5.0)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for site_summary.json / map.html")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(message: str) -> None:
        print(f"[{args.lat},{args.lon}] {message}")

    log_path = out_dir / "run.log"
    site_summary, map_html = run_analysis(args.lat, args.lon, args.radius, progress_cb=progress, log_path=log_path)
    print(f"Log:    {log_path.resolve()}")

    summary_path = out_dir / "site_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(site_summary, handle, indent=2, default=_json_safe)
    print(f"Saved: {summary_path.resolve()}")

    map_path = out_dir / "map.html"
    map_path.write_text(map_html)
    print(f"Saved: {map_path.resolve()}")


if __name__ == "__main__":
    main()
