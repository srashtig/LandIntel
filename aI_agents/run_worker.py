"""The app's background analysis-run worker — wraps
:func:`data_analysis_pipeline.orchestrator.run_analysis` with the
``on_assembled`` hook that static report-asset generation
(:mod:`aI_agents.report_assets`) needs, and puts results into the
queue/event/result-dict shape ``frontend/streamlit_app.py``'s
progress-polling dialog expects.
"""

from __future__ import annotations

import functools
import queue
import threading
from pathlib import Path

from data_analysis_pipeline.orchestrator import RunCancelled, run_analysis
from . import report_assets


def run_analysis_worker(
    lat: float,
    lon: float,
    radius_km: float,
    log_path: Path,
    progress_queue: "queue.Queue[str]",
    cancel_event: threading.Event,
    result: dict,
) -> None:
    """Run ``run_analysis`` on a background thread, with static report
    assets (maps/charts) generated during the run itself via the
    ``on_assembled`` hook — saved to ``report_assets/`` next to
    ``log_path`` (i.e. this run's own output directory). Only plain Python
    objects (queue/event/dict) are touched here — never any ``st.*`` call —
    since Streamlit APIs aren't safe to call from a thread other than the
    script's own.

    Args:
        lat: Site latitude in decimal degrees.
        lon: Site longitude in decimal degrees.
        radius_km: AOI radius in kilometers.
        log_path: This run's own ``run.log`` path — its parent directory is
            also where ``report_assets/`` gets created.
        progress_queue: Queue this thread pushes status strings onto; the
            main thread drains it each rerun to update the visible status.
        cancel_event: Set by the main thread when "Stop" is clicked;
            ``run_analysis`` checks it cooperatively between stages.
        result: Dict this thread fills in with
            ``{"status": "done"|"cancelled"|"error", ...}`` once finished.
    """

    def progress_cb(message: str) -> None:
        progress_queue.put(message)

    assets_dir = Path(log_path).parent / "report_assets"
    on_assembled = functools.partial(report_assets.generate_report_assets, assets_dir=assets_dir)

    try:
        site_summary, map_html = run_analysis(
            lat,
            lon,
            radius_km=radius_km,
            progress_cb=progress_cb,
            should_cancel=cancel_event.is_set,
            log_path=log_path,
            on_assembled=on_assembled,
        )
        result["status"] = "done"
        result["site_summary"] = site_summary
        result["map_html"] = map_html
    except RunCancelled:
        result["status"] = "cancelled"
    except Exception as error:  # noqa: BLE001 - hand any unexpected error back to the main thread to display
        result["status"] = "error"
        result["error"] = str(error)
