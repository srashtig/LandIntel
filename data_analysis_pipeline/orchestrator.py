"""Top-level orchestration: AOI -> every ``get_*`` module -> assembled
``site_summary`` -> rendered map.

Every ``get_*`` call is individually fault-isolated (see :func:`_safe_call`):
a GEE outage, a missing/corrupt local file, or a network timeout in any one
module never crashes the run or blocks the other modules — it is recorded
as a failure envelope for that section instead, and the pipeline continues.
The short reason goes into that section's ``limitations`` (for report/chat
consumption); the full exception + traceback is written to a per-run log
file (see ``log_path`` below) for debugging.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .aoi import AOI, build_aoi
from .assemble_summary import assemble_summary
from .build_map import build_map
from .get_admin_context import get_admin_context
from .get_groundwater import get_groundwater
from .get_hydrology import get_hydrology
from .get_land_cover import get_land_cover
from .get_land_surface_temperature import get_air_temperature, get_land_surface_temperature
from .get_nearby_places import get_nearby_places
from .get_osm_features import get_railways, get_roads, get_water
from .get_rainfall import get_rainfall
from .get_sentinel_features import get_sentinel_features
from .get_terrain import get_terrain

ProgressCallback = Optional[Callable[[str], None]]
CancelCheck = Optional[Callable[[], bool]]


class RunCancelled(RuntimeError):
    """Raised by :func:`run_analysis` when ``should_cancel`` reports True
    between stages. Cancellation is cooperative and checked only at stage
    boundaries (before AOI build, before each get_* call, before summary
    assembly and map building) — a stage already in flight (e.g. a slow
    Overpass/GEE/SerpApi network call) always finishes before the run stops,
    it cannot be interrupted mid-call.
    """

# Sentinel confidence value used only for the synthetic fallback envelope a
# failed get_* call is replaced with — never returned by a module that
# actually ran (a module reporting zero/no-data-found uses "low", not this).
FAILURE_CONFIDENCE = "unavailable"

LOGGER_NAME = "data_analysis_pipeline"

# Some third-party retry logic has no upper bound at all — e.g. osmnx
# recursively retries a 429/504 from Overpass with a fixed pause and no
# retry cap, so it can hang forever without ever raising. A hard wall-clock
# timeout is the only thing that can bound that: every get_* call runs on
# its own thread and is abandoned (not killed — Python can't force-kill a
# thread) if it doesn't finish in time, so the orchestrator can move on to
# the rest of the pipeline. Generous enough to comfortably cover the
# slowest known legitimate case (get_nearby_places' up to ~8 sequential
# 60s-timeout SerpApi calls), while still being finite.
DEFAULT_STAGE_TIMEOUT_SECONDS = 600

# Timeout for the optional on_assembled hook (static report-asset
# generation) — generous for a handful of GEE thumbnail fetches + matplotlib
# renders (normally well under a minute), but still finite so a stuck
# network call there can never hang the whole run.
REPORT_ASSETS_TIMEOUT_SECONDS = 240


def _configure_run_logger(log_path: Optional[Path]) -> tuple[logging.Logger, logging.Handler]:
    """Attach a per-run file handler to the package logger.

    Defaults to ``runs/pipeline.log`` (a single shared file) when
    the caller doesn't have a dated run directory yet; ``main.py`` passes
    ``<out-dir>/run.log`` so each CLI run gets its own log alongside its
    ``site_summary.json``/``map.html``. The handler is returned so the
    caller can remove it again when the run finishes (this function may be
    called repeatedly in one long-lived process, e.g. from Stage B).

    Args:
        log_path: File to log to, or None to use the shared default
            ``runs/pipeline.log``.

    Returns:
        ``(logger, handler)`` — pass ``handler`` to
        ``logger.removeHandler(handler)`` (and ``handler.close()``) when done.
    """

    if log_path is None:
        from . import config
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.RUNS_DIR / "pipeline.log"
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    return logger, handler


def _failure_envelope(dataset: str, source: str, params: dict, error: Exception) -> dict:
    """Build the standard-shaped envelope substituted for a get_* module
    that raised an exception (see the FAILURE_CONFIDENCE note above).

    Args:
        dataset: The failed module's ``dataset`` name (as it would have
            appeared in a successful envelope).
        source: The failed module's ``source`` description.
        params: The call params that were passed to the failed module.
        error: The exception that was raised.

    Returns:
        An envelope dict with ``confidence="unavailable"``, empty
        ``observations``, and the short failure reason in ``limitations``.
    """

    return {
        "dataset": dataset,
        "source": source,
        "params": params,
        "observations": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": FAILURE_CONFIDENCE,
        "warnings": [],
        "limitations": [f"failed: {error}"],
    }


def _call_with_hard_timeout(fn: Callable, args: tuple, timeout_seconds: float):
    """Run ``fn(*args)`` on its own daemon thread and wait at most
    ``timeout_seconds`` for it — the one way to bound a call that might
    never return control on its own (see :data:`DEFAULT_STAGE_TIMEOUT_SECONDS`).

    If the thread hasn't finished by the deadline, this returns control to
    the caller anyway; the thread keeps running in the background (Python
    cannot force-kill a thread) until it eventually finishes or the process
    exits, but its result is simply discarded — nothing reads it, and
    nothing waits on it again.

    Args:
        fn: The function to call.
        args: Positional arguments to call it with.
        timeout_seconds: Maximum time to wait for it to finish.

    Returns:
        ``("timeout", None)`` if the deadline was reached; ``("error", exc)``
        if it raised within the deadline; ``("ok", return_value)`` if it
        returned normally within the deadline.
    """

    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = fn(*args)
        except Exception as error:  # noqa: BLE001 - handed back to the caller, not swallowed
            outcome["error"] = error

    thread, outcome = _start_hard_timeout_call(fn, args)
    return _join_hard_timeout_call(thread, outcome, timeout_seconds)


def _start_hard_timeout_call(fn: Callable, args: tuple) -> tuple[threading.Thread, dict]:
    """Start ``fn(*args)`` on its own daemon thread and return immediately
    (no waiting) — the "start" half of :func:`_call_with_hard_timeout`,
    split out so a slow I/O-bound call (e.g. an OSM Overpass query) can be
    kicked off early and joined later, overlapped with unrelated stages
    that run in between (see :func:`_join_hard_timeout_call` and the
    "OSM railways"/"OSM water" stages in :func:`run_analysis`, deliberately
    positioned after Sentinel-2 NDVI / groundwater respectively so their
    network wait overlaps with those stages' GEE calls instead of adding to
    the total wall time).

    Args:
        fn: The function to call.
        args: Positional arguments to call it with.

    Returns:
        ``(thread, outcome)`` — pass both to :func:`_join_hard_timeout_call`
        once ready to wait for the result.
    """

    outcome: dict = {}

    def target() -> None:
        try:
            outcome["value"] = fn(*args)
        except Exception as error:  # noqa: BLE001 - handed back to the caller, not swallowed
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def _join_hard_timeout_call(thread: threading.Thread, outcome: dict, timeout_seconds: float):
    """Wait at most ``timeout_seconds`` for a thread started by
    :func:`_start_hard_timeout_call` — the "join" half. If the thread was
    started well before this is called (because unrelated stages ran in
    between), most or all of ``timeout_seconds`` may already effectively be
    "free" wall-clock time that already elapsed while this process was busy
    elsewhere.

    Returns:
        ``("timeout", None)`` if the deadline was reached; ``("error", exc)``
        if it raised within the deadline; ``("ok", return_value)`` if it
        returned normally within the deadline.
    """

    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return "timeout", None
    if "error" in outcome:
        return "error", outcome["error"]
    return "ok", outcome.get("value")


def _safe_call(
    logger: logging.Logger,
    dataset: str,
    source: str,
    params: dict,
    fn: Callable,
    *args,
    timeout_seconds: float = DEFAULT_STAGE_TIMEOUT_SECONDS,
) -> dict:
    """Run one ``get_*`` module with a hard wall-clock timeout, converting
    any exception OR a timeout into a standard "unavailable" envelope
    instead of propagating it (or hanging forever). The short reason is
    recorded in the envelope's ``limitations``; a real exception's full
    traceback goes to the run log at ERROR level.

    Args:
        logger: Logger to record start/success/failure/timeout to.
        dataset: The module's ``dataset`` name, used in the fallback
            envelope and log messages if ``fn`` raises or times out.
        source: The module's ``source`` description, used in the fallback
            envelope if ``fn`` raises or times out.
        params: The call params, recorded in the fallback envelope if
            ``fn`` raises or times out.
        fn: The ``get_*`` function to call.
        *args: Positional arguments passed through to ``fn``.
        timeout_seconds: Maximum time to wait before giving up on ``fn``
            and moving on — see :data:`DEFAULT_STAGE_TIMEOUT_SECONDS`.

    Returns:
        ``fn(*args)``'s own envelope on success, or a
        :func:`_failure_envelope` if it raised or timed out.
    """

    logger.info("%s: starting", dataset)
    status, result = _call_with_hard_timeout(fn, args, timeout_seconds)

    if status == "timeout":
        logger.error("%s: timed out after %ss (it may still be running in the background; this run has moved on without it)", dataset, timeout_seconds)
        print(f"[data_analysis_pipeline] {dataset} timed out after {timeout_seconds}s -- continuing with remaining modules (see log).")
        return _failure_envelope(dataset, source, params, TimeoutError(f"timed out after {timeout_seconds}s"))

    if status == "error":
        error = result
        logger.error("%s: failed - %s", dataset, error, exc_info=(type(error), error, error.__traceback__))
        print(f"[data_analysis_pipeline] {dataset} failed: {error!r} -- continuing with remaining modules (see log).")
        return _failure_envelope(dataset, source, params, error)

    logger.info("%s: completed", dataset)
    return result


def _start_safe_call(logger: logging.Logger, dataset: str, fn: Callable, *args) -> tuple:
    """Kick off one ``get_*`` module on a background thread and return
    immediately — the "start" half of the background-stage pattern used for
    "OSM railways" and "OSM water" (see :func:`run_analysis`). Pass the
    returned handle to :func:`_finish_safe_call` later, at the point the
    envelope is actually needed.

    Args:
        logger: Logger to record the start to.
        dataset: The module's ``dataset`` name, used in logging and in the
            fallback envelope if the call later fails/times out.
        fn: The ``get_*`` function to call.
        *args: Positional arguments passed through to ``fn``.

    Returns:
        An opaque handle for :func:`_finish_safe_call`.
    """

    logger.info("%s: starting (background)", dataset)
    thread, outcome = _start_hard_timeout_call(fn, args)
    return (thread, outcome)


def _finish_safe_call(
    logger: logging.Logger,
    dataset: str,
    source: str,
    params: dict,
    handle: tuple,
    timeout_seconds: float = DEFAULT_STAGE_TIMEOUT_SECONDS,
) -> dict:
    """Wait for a background stage started by :func:`_start_safe_call` and
    convert its outcome into a standard envelope, same as :func:`_safe_call`
    does for a normal blocking stage.

    Args:
        logger: Logger to record success/failure/timeout to.
        dataset: The module's ``dataset`` name.
        source: The module's ``source`` description, used in the fallback
            envelope if the call raised or timed out.
        params: The call params, recorded in the fallback envelope if the
            call raised or timed out.
        handle: The tuple returned by :func:`_start_safe_call`.
        timeout_seconds: Maximum *additional* time to wait, on top of
            whatever already elapsed while other stages ran in between.

    Returns:
        The module's own envelope on success, or a :func:`_failure_envelope`
        if it raised or timed out.
    """

    thread, outcome = handle
    status, result = _join_hard_timeout_call(thread, outcome, timeout_seconds)

    if status == "timeout":
        logger.error("%s: timed out after %ss (it may still be running in the background; this run has moved on without it)", dataset, timeout_seconds)
        print(f"[data_analysis_pipeline] {dataset} timed out after {timeout_seconds}s -- continuing with remaining modules (see log).")
        return _failure_envelope(dataset, source, params, TimeoutError(f"timed out after {timeout_seconds}s"))

    if status == "error":
        error = result
        logger.error("%s: failed - %s", dataset, error, exc_info=(type(error), error, error.__traceback__))
        print(f"[data_analysis_pipeline] {dataset} failed: {error!r} -- continuing with remaining modules (see log).")
        return _failure_envelope(dataset, source, params, error)

    logger.info("%s: completed", dataset)
    return result


def _combine_osm_results(roads_env: dict, rail_env: dict, water_env: dict) -> dict:
    """Merge the three separately fetched/joined OSM sub-stage envelopes
    (see ``get_osm_features.get_roads``/``get_railways``/``get_water`` and
    the "OSM roads"/"OSM railways"/"OSM water" stages in
    :func:`run_analysis`) into the single combined "osm" envelope that
    ``assemble_summary.py`` and ``build_map.py`` expect (``observations``
    keyed by "roads"/"railways"/"water"). A sub-stage that failed or timed
    out is simply omitted from ``observations`` — those two already treat a
    missing key there as "unavailable", same as a whole-module failure
    always meant before this split.
    """

    warnings: list[str] = []
    observations: dict = {}
    for key, envelope in (("roads", roads_env), ("railways", rail_env), ("water", water_env)):
        warnings.extend(envelope.get("warnings", []))
        if envelope.get("confidence") == FAILURE_CONFIDENCE:
            reason = (envelope.get("limitations") or ["failed"])[0]
            warnings.append(f"{key}: {reason}")
        else:
            observations[key] = envelope["observations"]

    return {
        "dataset": "osm_features",
        "source": "OpenStreetMap via OSMnx",
        "params": roads_env.get("params", {}),
        "observations": observations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "high" if observations else FAILURE_CONFIDENCE,
        "warnings": warnings,
        "limitations": ["OSM coverage/tagging completeness varies by area."],
    }


def run_analysis(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    progress_cb: ProgressCallback = None,
    log_path: Optional[Path] = None,
    should_cancel: CancelCheck = None,
    on_assembled: Optional[Callable[[AOI, dict, dict], None]] = None,
):
    """Run the full Land Intelligence pipeline for one site: build the AOI,
    fetch every data source (fault-isolated — see module docstring),
    assemble the summary, and render the map. This is the single top-level
    entry point for an on-demand analysis.

    Args:
        lat: Site latitude in decimal degrees (WGS84 / EPSG:4326).
        lon: Site longitude in decimal degrees (WGS84 / EPSG:4326).
        radius_km: Radius of the circular area of interest, in kilometers.
            Defaults to 5.0.
        progress_cb: Optional callback invoked with a short status string
            before each stage starts (e.g. "Fetching Sentinel-2 NDVI...")
            and again with a "✓ <label>" / "✗ <label>" outcome once a
            fault-isolated get_* stage finishes (so a UI can render a live
            per-stage checklist, not just a single spinner). The
            orchestrator itself has no UI dependency — it just emits plain
            strings.
        log_path: File every module's start/success/failure is logged to.
            Defaults to a shared ``runs/pipeline.log``; ``main.py``
            passes ``<out-dir>/run.log`` so each CLI run gets its own log.
        should_cancel: Optional zero-arg callable returning True once the
            caller wants this run stopped (e.g. ``threading.Event().is_set``
            from a UI "Stop" button running the pipeline on a background
            thread). Checked at every stage boundary — see
            :class:`RunCancelled` for the exact granularity. Ignored if None
            (the default): the run always runs to completion.
        on_assembled: Optional callback invoked once, right after
            ``site_summary`` is assembled, with ``(aoi, results,
            site_summary)`` — ``results`` is the same raw per-module envelope
            dict (containing GeoDataFrames/``ee.Image``s) that
            :func:`~data_analysis_pipeline.build_map.build_map` itself consumes,
            not the JSON-safe ``site_summary``. Intended for a caller that
            wants to derive something else from the same run (e.g. static
            report assets) without re-fetching any data. Fault-isolated like
            every other stage (a timeout or exception here is logged and
            swallowed, never propagated) and reported through the same
            ``progress_cb`` start()/finish() pattern under the label "Report
            assets". Ignored if None (the default) — a no-op for every
            existing caller.

    Returns:
        ``(site_summary, map_html)`` — ``site_summary`` is the assembled
        dict documented in :mod:`assemble_summary` (its "run_success" field
        is False if one or more ``get_*`` modules failed, though the run
        still completes and returns valid partial output); ``map_html`` is
        the rendered map as a self-contained HTML string.

    Raises:
        RunCancelled: if ``should_cancel`` reports True at a stage boundary.
    """

    def start(label: str) -> None:
        # "… <label>" is the fixed prefix a UI matches a later "✓ <label>" /
        # "✗ <label>" outcome back to, so it can update the SAME displayed
        # line in place instead of appending a new one — label text must be
        # identical between the start() and finish() call for a given stage.
        # Cancellation is checked BEFORE announcing the next stage, so a
        # cancelled run's progress log never shows a stage that didn't
        # actually run.
        if should_cancel and should_cancel():
            logger.warning("run_analysis cancelled by caller before stage: %s", label)
            raise RunCancelled(f"cancelled before: {label}")
        if progress_cb:
            progress_cb(f"… {label}")

    def finish(label: str, ok: bool) -> None:
        # No cancellation check here — the work already happened, so its
        # outcome is always worth reporting even if Stop was requested
        # while it was running.
        if progress_cb:
            progress_cb(f"{'✓' if ok else '✗'} {label}")

    def run_stage(label: str, dataset: str, source: str, params: dict, fn: Callable, *args) -> dict:
        """Run one fault-isolated get_* module: announce it starting, run it
        through _safe_call, then report a "✓ <label>" / "✗ <label>" outcome."""

        start(label)
        envelope = _safe_call(logger, dataset, source, params, fn, *args)
        finish(label, envelope.get("confidence") != FAILURE_CONFIDENCE)
        return envelope

    def start_background_stage(label: str, dataset: str, fn: Callable, *args) -> tuple:
        """Announce and kick off one fault-isolated get_* module on a
        background thread WITHOUT waiting for it — pairs with
        :func:`finish_background_stage`, called later once its result is
        actually needed, after other unrelated stages have run in between
        (see the "OSM railways"/"OSM water" stages below, deliberately
        overlapped with Sentinel-2 NDVI / groundwater)."""

        start(label)
        handle = _start_safe_call(logger, dataset, fn, *args)
        return (label, handle)

    def finish_background_stage(dataset: str, source: str, params: dict, background: tuple) -> dict:
        """Wait for a stage started by :func:`start_background_stage` and
        report its "✓ <label>" / "✗ <label>" outcome."""

        label, handle = background
        envelope = _finish_safe_call(logger, dataset, source, params, handle)
        finish(label, envelope.get("confidence") != FAILURE_CONFIDENCE)
        return envelope

    logger, handler = _configure_run_logger(log_path)
    try:
        logger.info("run_analysis starting: lat=%s lon=%s radius_km=%s", lat, lon, radius_km)

        start("AOI geometry")
        aoi = build_aoi(lat, lon, radius_km)  # not fault-isolated: nothing downstream can run without it
        finish("AOI geometry", True)

        base_params = {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km}
        results: dict = {}

        # OSM's roads/railways/water are fetched as three separate stages
        # (not one combined "get_osm_features" call) so that railways (the
        # layer most often slowed down by a rate-limited/retrying Overpass
        # request, see get_osm_features.py) and water are each *started* in
        # the background and only *joined* later, right before their
        # envelope is actually needed — overlapping their network wait with
        # Sentinel-2 NDVI's and groundwater's unrelated GEE calls instead of
        # adding to the total wall time. Roads stays a normal blocking stage
        # since nothing precedes it to overlap with.
        results["osm_roads"] = run_stage(
            "OSM roads", "osm_roads", "OpenStreetMap via OSMnx", base_params, get_roads, aoi
        )
        osm_rail_background = start_background_stage("OSM railways", "osm_railways", get_railways, aoi)

        results["sentinel"] = run_stage(
            "Sentinel-2 NDVI", "sentinel_ndvi", "COPERNICUS/S2_SR_HARMONIZED", base_params, get_sentinel_features, aoi
        )

        results["osm_railways"] = finish_background_stage("osm_railways", "OpenStreetMap via OSMnx", base_params, osm_rail_background)

        results["land_cover"] = run_stage(
            "ESA WorldCover land cover", "land_cover", "ESA/WorldCover/v200", base_params, get_land_cover, aoi
        )
        results["lst"] = run_stage(
            "Landsat land-surface temperature",
            "land_surface_temperature",
            "LANDSAT/LC08/C02/T1_L2",
            base_params,
            get_land_surface_temperature,
            aoi,
        )
        results["air_temp"] = run_stage(
            "ERA5-Land air temperature", "air_temperature", "ECMWF/ERA5_LAND/MONTHLY_AGGR", base_params, get_air_temperature, aoi
        )
        results["terrain"] = run_stage(
            "SRTM terrain (elevation/slope)", "terrain", "USGS/SRTMGL1_003", base_params, get_terrain, aoi
        )

        osm_water_background = start_background_stage("OSM water", "osm_water", get_water, aoi)

        results["groundwater"] = run_stage(
            "CGWB groundwater levels",
            "groundwater",
            "CGWB manual monthly groundwater monitoring",
            base_params,
            get_groundwater,
            aoi,
        )

        results["osm_water"] = finish_background_stage("osm_water", "OpenStreetMap via OSMnx", base_params, osm_water_background)

        results["osm"] = _combine_osm_results(
            results.pop("osm_roads"), results.pop("osm_railways"), results.pop("osm_water")
        )

        results["hydrology"] = run_stage(
            "River floodplain and SAC water bodies",
            "hydrology",
            "river_polygon.GeoJSON + wb_sac_mp.GeoJSON",
            base_params,
            get_hydrology,
            aoi,
        )
        results["rainfall"] = run_stage(
            "CHIRPS + IMD rainfall metrics",
            "rainfall",
            "UCSB-CHC/CHIRPS/V3/DAILY_RNL + IMD 0.25°",
            base_params,
            get_rainfall,
            aoi,
        )
        results["nearby_places"] = run_stage(
            "Nearby settlements and places (SerpApi)", "nearby_places", "OpenStreetMap + SerpApi", base_params, get_nearby_places, aoi
        )
        results["admin"] = run_stage(
            "Administrative context (village/tehsil/district)",
            "admin_context",
            "Survey of India village boundaries (vb_soi_mp)",
            base_params,
            get_admin_context,
            aoi,
        )
        run_success = all(envelope.get("confidence") != FAILURE_CONFIDENCE for envelope in results.values())
        failed_modules = [name for name, envelope in results.items() if envelope.get("confidence") == FAILURE_CONFIDENCE]
        if failed_modules:
            logger.warning("run completed with failures in: %s", failed_modules)
            print(f"[data_analysis_pipeline] run completed with failures in: {failed_modules}")

        start("Site summary")
        site_summary = assemble_summary(aoi, results, run_success=run_success)
        finish("Site summary", True)

        if on_assembled is not None:
            start("Report assets")
            status, outcome = _call_with_hard_timeout(
                on_assembled, (aoi, results, site_summary), REPORT_ASSETS_TIMEOUT_SECONDS
            )
            if status == "timeout":
                logger.error("on_assembled hook timed out after %ss", REPORT_ASSETS_TIMEOUT_SECONDS)
            elif status == "error":
                logger.error("on_assembled hook failed - %s", outcome, exc_info=(type(outcome), outcome, outcome.__traceback__))
            finish("Report assets", status == "ok")

        start("Map")
        _map, map_html = build_map(aoi, results, site_summary)
        finish("Map", True)

        logger.info("run_analysis completed: run_success=%s", run_success)
        return site_summary, map_html
    finally:
        logger.removeHandler(handler)
        handler.close()
