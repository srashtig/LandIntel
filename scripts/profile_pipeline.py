"""Pre-deployment profiling: real wall-clock time, peak RSS, CPU%, and
external-call counts for one full analysis run, plus the Generate Report,
AI chat, and Compare add-on costs, plus cold-import cost. Not imported by
the app — run manually to inform Cloud Run sizing:

    conda activate land_intel
    python scripts/profile_pipeline.py                 # full driver run
    python scripts/profile_pipeline.py --phase imports  # one phase only

Design notes (see docs/PROFILING.md for the actual results):
- Each phase (pipeline/report/chat/compare) runs in its own subprocess
  (re-invoking this same file with --worker <phase>), so peak-RSS readings
  for one phase are never contaminated by another, and each gets a fresh
  GEE/Groq client. The parent samples the child's RSS/CPU via `psutil`
  from the outside — nothing inside the measured process is touched to
  make this work.
- Per-stage timing comes from `run_analysis`'s `progress_cb` — NOT the
  `pipeline.log` file logger. Confirmed by reading orchestrator.py: only
  the fault-isolated get_* stages go through the file logger; "AOI
  geometry"/"Site summary"/"Report assets"/"Map" only ever reach
  `progress_cb`. progress_cb is therefore the one mechanism that covers
  every stage, so it's used exclusively (the log file is left alone,
  purely for human debugging if a run fails).
- Overlap between stages (e.g. "OSM railways" kicked off in the
  background, joined much later) falls out naturally from comparing each
  stage's [start, finish] timestamp window against every other stage's —
  nothing is hardcoded about which stages overlap.
- Both `aI_agents/qa.py` and `aI_agents/qa_agent.py` do
  `from .groq_client import _create_completion`, a separate name binding
  each — so both module attributes are patched, not just
  `groq_client._create_completion` itself (reassigning the original
  wouldn't be seen by either caller).
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_LAT = 23.0401972
DEFAULT_LON = 76.2086806
DEFAULT_RADIUS_KM = 5.0
DEFAULT_OUT_DIR = _REPO_ROOT / "profiling_results"

_EXTERNAL_HOSTS = {
    "serpapi.com": "SerpApi",
    "overpass-api.de": "Overpass",
    "www.onefivenine.com": "onefivenine.com",
    "onefivenine.com": "onefivenine.com",
    "en.wikipedia.org": "Wikipedia",
    "router.project-osrm.org": "OSRM",
    "server.arcgisonline.com": "Esri basemap tiles",
    "earthengine.googleapis.com": "GEE (direct HTTP; most GEE calls go through the earthengine-api client, not counted here)",
}

# ---------------------------------------------------------------------
# Instrumentation (all monkeypatching, applied only inside a worker
# subprocess — never touches the real source files).
# ---------------------------------------------------------------------


def _patch_requests_counter() -> dict:
    """Wrap ``requests.get``/``requests.post`` to count calls by host.
    Confirmed both osmnx (Overpass) and this app's own SerpApi/OSRM calls
    go through the plain module-level functions, not a `requests.Session`
    instance, so patching these two functions catches every external HTTP
    call this app makes."""

    import requests

    counts: dict[str, int] = {}
    originals = {"get": requests.get, "post": requests.post}

    def _record(url: str) -> None:
        host = urlparse(url).hostname or "unknown"
        label = _EXTERNAL_HOSTS.get(host, host)
        counts[label] = counts.get(label, 0) + 1

    def _wrap(name: str):
        original = originals[name]

        @functools.wraps(original)
        def wrapped(url, *args, **kwargs):
            _record(url)
            return original(url, *args, **kwargs)

        return wrapped

    requests.get = _wrap("get")
    requests.post = _wrap("post")
    return counts


def _patch_report_asset_timers() -> dict:
    """Wrap each of report_assets.py's 9 named generator functions plus its
    2 shared basemap fetchers with per-call timing. Patched as module
    attributes — safe because `generate_report_assets`'s body looks these
    names up dynamically (module-global lookup) each time it runs, not via
    a closure captured at def time."""

    from aI_agents import report_assets

    names = [
        "_site_context_map",
        "_nearby_places_map",
        "_groundwater_map",
        "_water_bodies_map",
        "_reference_location_map",
        "_ndvi_map",
        "_land_cover_map",
        "_groundwater_chart",
        "_slope_class_chart",
        "_fetch_street_basemap",
        "_fetch_satellite_labels_basemap",
    ]
    timings: dict[str, float] = {}
    for name in names:
        original = getattr(report_assets, name)

        def _make_wrapper(fn, key):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)

            return wrapped

        setattr(report_assets, name, _make_wrapper(original, name.lstrip("_")))
    return timings


def _patch_groq_capture() -> list:
    """Wrap `_create_completion` in both `aI_agents.qa` and
    `aI_agents.qa_agent` (each holds its own `from .groq_client import
    _create_completion` binding — reassigning groq_client's own copy would
    not be seen by either) to record token usage per call, labeled by the
    immediate caller's function name."""

    from aI_agents import qa, qa_agent

    calls: list[dict] = []

    def _make_wrapper(original):
        @functools.wraps(original)
        def wrapped(client, **kwargs):
            caller = inspect.stack()[1].function
            response = original(client, **kwargs)
            usage = getattr(response, "usage", None)
            calls.append({
                "caller": caller,
                "max_completion_tokens_requested": kwargs.get("max_completion_tokens"),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            })
            return response

        return wrapped

    qa._create_completion = _make_wrapper(qa._create_completion)
    qa_agent._create_completion = _make_wrapper(qa_agent._create_completion)
    return calls


class _ProgressRecorder:
    """`run_analysis`'s `progress_cb` — the only mechanism that covers
    every stage including the ones the file logger never sees (AOI
    geometry, Site summary, Report assets, Map) — see module docstring."""

    def __init__(self):
        self.events: list[dict] = []

    def __call__(self, message: str) -> None:
        now = time.time()
        marker, _, label = message.partition(" ")
        if marker == "…":
            self.events.append({"label": label, "type": "start", "t": now})
        else:
            ok = marker == "✓"
            self.events.append({"label": label, "type": "finish", "t": now, "ok": ok})

    def stage_windows(self) -> list[dict]:
        """Collapse start/finish events into ``[{label, start, end, ok}]``,
        one entry per stage (matches by label, in order)."""

        starts: dict[str, float] = {}
        windows = []
        for event in self.events:
            if event["type"] == "start":
                starts[event["label"]] = event["t"]
            else:
                start_t = starts.pop(event["label"], event["t"])
                windows.append({
                    "label": event["label"],
                    "start": start_t,
                    "end": event["t"],
                    "seconds": event["t"] - start_t,
                    "ok": event["ok"],
                })
        return windows


# ---------------------------------------------------------------------
# Worker phases (each runs in its own subprocess)
# ---------------------------------------------------------------------


def _worker_pipeline(args) -> None:
    import functools as _functools

    from aI_agents import report_assets
    from data_analysis_pipeline.orchestrator import run_analysis

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    request_counts = _patch_requests_counter()
    asset_timings = _patch_report_asset_timers()
    recorder = _ProgressRecorder()

    assets_dir = out_dir / "report_assets"
    on_assembled = _functools.partial(report_assets.generate_report_assets, assets_dir=assets_dir)

    wall_start = time.time()
    site_summary, map_html = run_analysis(
        args.lat,
        args.lon,
        radius_km=args.radius,
        progress_cb=recorder,
        log_path=out_dir / "pipeline.log",
        on_assembled=on_assembled,
    )
    wall_total = time.time() - wall_start

    (out_dir / "site_summary.json").write_text(json.dumps(site_summary, indent=2, default=str))
    (out_dir / "map.html").write_text(map_html)
    (out_dir / "profile_extra.json").write_text(json.dumps({
        "wall_total_seconds": wall_total,
        "stage_windows": recorder.stage_windows(),
        "request_counts": request_counts,
        "report_asset_timings": asset_timings,
    }, indent=2))
    print(f"[worker:pipeline] done in {wall_total:.1f}s -> {out_dir}")


def _worker_report(args) -> None:
    from aI_agents.build_report import generate_report

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    site_summary = json.loads(summary_path.read_text())
    # A raw pipeline run has no buying_purpose set (that's a separate,
    # UI-driven step) — generate_report() only makes its one Groq call
    # `if purpose:` (confirmed by reading build_report.py directly), so
    # without this the measured cost would silently exclude the AI call
    # that's actually present on every real report a user generates.
    site_summary.setdefault("buying_purpose", {"key": "farming", "label": "Farming"})

    request_counts = _patch_requests_counter()
    groq_calls = _patch_groq_capture()

    wall_start = time.time()
    generate_report(site_summary, summary_path)
    wall_total = time.time() - wall_start

    (out_dir / "profile_extra.json").write_text(json.dumps({
        "wall_total_seconds": wall_total,
        "request_counts": request_counts,
        "groq_calls": groq_calls,
    }, indent=2))
    print(f"[worker:report] done in {wall_total:.1f}s")


def _worker_chat(args) -> None:
    from aI_agents.qa_agent import route_and_answer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    site_summary = json.loads(summary_path.read_text())

    request_counts = _patch_requests_counter()
    groq_calls = _patch_groq_capture()

    questions = [
        "What is the groundwater situation and dominant land cover here?",
        "Are there any petrol pumps nearby?",
    ]
    results = []
    history: list[dict] = []
    for question in questions:
        start = time.time()
        result = route_and_answer(site_summary, question, history=history, site_label=summary_path.parent.name)
        elapsed = time.time() - start
        results.append({"question": question, "seconds": elapsed, "tool_used": result.get("tool_used")})
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

    (out_dir / "profile_extra.json").write_text(json.dumps({
        "questions": results,
        "request_counts": request_counts,
        "groq_calls": groq_calls,
    }, indent=2))
    print(f"[worker:chat] done, {len(questions)} questions")


def _worker_compare(args) -> None:
    from aI_agents.compare import compare

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_a = json.loads(Path(args.summary_path).read_text())
    summary_b = json.loads(Path(args.other_summary_path).read_text())
    label_a = Path(args.summary_path).parent.name
    label_b = Path(args.other_summary_path).parent.name

    request_counts = _patch_requests_counter()
    groq_calls = _patch_groq_capture()

    wall_start = time.time()
    compare([(label_a, summary_a), (label_b, summary_b)], purpose="housing")
    wall_total = time.time() - wall_start

    (out_dir / "profile_extra.json").write_text(json.dumps({
        "wall_total_seconds": wall_total,
        "sites": [label_a, label_b],
        "request_counts": request_counts,
        "groq_calls": groq_calls,
    }, indent=2))
    print(f"[worker:compare] done in {wall_total:.1f}s")


_WORKERS = {
    "pipeline": _worker_pipeline,
    "report": _worker_report,
    "chat": _worker_chat,
    "compare": _worker_compare,
}


# ---------------------------------------------------------------------
# Driver: launches each phase as a subprocess, samples RSS/CPU from
# outside, and assembles the final report.
# ---------------------------------------------------------------------


def _run_imports_phase() -> dict:
    results = {}
    for label, import_stmt in [
        ("data_analysis_pipeline (pipeline only)", "import data_analysis_pipeline.orchestrator"),
        ("aI_agents.build_report (+ matplotlib/contextily/PIL)", "import aI_agents.build_report"),
    ]:
        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", import_stmt],
            cwd=str(_REPO_ROOT), capture_output=True, text=True,
        )
        wall = time.time() - start
        lines = [l for l in proc.stderr.splitlines() if l.startswith("import time:")]
        top = sorted(
            (l for l in lines[1:]),
            key=lambda l: -int(l.split("|")[1].strip() or 0),
        )[:15]
        results[label] = {"wall_seconds": wall, "top_frames": top}
        print(f"[imports] {label}: {wall:.2f}s")
    return results


def _launch_worker(phase: str, out_dir: Path, force: bool = False, **kwargs) -> tuple[float, float, float]:
    """Launch `python scripts/profile_pipeline.py --worker <phase> ...`,
    sample the child's RSS/CPU from this (parent) process until it exits.
    Skips re-running (and re-spending real GEE/Overpass/SerpApi/Groq calls)
    if `<out_dir>/_driver_meta.json` already exists from a prior run of
    this same phase — pass `force=True` to always re-run.

    Returns:
        (wall_seconds, peak_rss_mb, avg_cpu_percent, samples)
    """

    import psutil

    meta_path = out_dir / "_driver_meta.json"
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        samples = json.loads((out_dir / f"{phase}_samples.json").read_text())
        print(f"[driver] {phase}: reusing cached result ({meta['wall']:.1f}s wall, {meta['peak_rss']:.0f} MB peak RSS)")
        return meta["wall"], meta["peak_rss"], meta["avg_cpu"], samples

    cmd = [sys.executable, str(_THIS_FILE), "--worker", phase, "--out-dir", str(out_dir)]
    for key, value in kwargs.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]

    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
    ps_proc = psutil.Process(proc.pid)
    ps_proc.cpu_percent(interval=None)  # prime the counter

    samples = []
    while proc.poll() is None:
        try:
            rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
            cpu = ps_proc.cpu_percent(interval=None)
            samples.append((time.time(), rss_mb, cpu))
        except psutil.NoSuchProcess:
            break
        time.sleep(0.25)
    proc.wait()

    wall = time.time() - start
    peak_rss = max((s[1] for s in samples), default=0.0)
    avg_cpu = sum(s[2] for s in samples) / len(samples) if samples else 0.0

    samples_path = out_dir / f"{phase}_samples.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps(samples))
    meta_path.write_text(json.dumps({"wall": wall, "peak_rss": peak_rss, "avg_cpu": avg_cpu}))

    print(f"[driver] {phase}: {wall:.1f}s wall, {peak_rss:.0f} MB peak RSS, {avg_cpu:.0f}% avg CPU")
    return wall, peak_rss, avg_cpu, samples


def _stage_peak_rss(stage: dict, samples: list) -> float:
    in_window = [s[1] for s in samples if stage["start"] <= s[0] <= stage["end"]]
    return max(in_window) if in_window else 0.0


def _stage_avg_cpu(stage: dict, samples: list) -> float:
    in_window = [s[2] for s in samples if stage["start"] <= s[0] <= stage["end"]]
    return sum(in_window) / len(in_window) if in_window else 0.0


def _find_overlaps(windows: list[dict]) -> dict[str, list[str]]:
    overlaps: dict[str, list[str]] = {w["label"]: [] for w in windows}
    for i, a in enumerate(windows):
        for b in windows[i + 1:]:
            if a["start"] < b["end"] and b["start"] < a["end"]:
                overlaps[a["label"]].append(b["label"])
                overlaps[b["label"]].append(a["label"])
    return overlaps


def _run_driver(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = ["# Pipeline Profiling Report", ""]
    report_lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} for lat={args.lat}, lon={args.lon}, radius={args.radius}km.")
    report_lines.append("")
    report_lines.append(
        "Based on a single representative run (`manual`/Budasa). The tool "
        "supports any lat/lon and pairing any two existing runs for "
        "Compare — a fuller multi-site sweep would sharpen this further; "
        "treat the numbers below as directionally correct, not exact for "
        "every possible site."
    )
    report_lines.append("")

    phases_to_run = args.phase.split(",") if args.phase != "all" else ["imports", "pipeline", "report", "chat", "compare"]

    if "imports" in phases_to_run:
        report_lines.append("## Startup / import cost")
        report_lines.append("")
        import_results = _run_imports_phase()
        for label, data in import_results.items():
            report_lines.append(f"**{label}**: {data['wall_seconds']:.2f}s cold import")
            report_lines.append("")
            report_lines.append("Slowest individual frames (self time):")
            report_lines.append("```")
            report_lines.extend(data["top_frames"][:10])
            report_lines.append("```")
            report_lines.append("")

    pipeline_dir = out_dir / "pipeline"
    stage_windows: list[dict] = []
    samples: list = []
    extra: dict = {}
    if "pipeline" in phases_to_run:
        wall, peak_rss, avg_cpu, samples = _launch_worker(
            "pipeline", pipeline_dir, lat=args.lat, lon=args.lon, radius=args.radius,
        )
        extra = json.loads((pipeline_dir / "profile_extra.json").read_text())
        stage_windows = extra["stage_windows"]
        overlaps = _find_overlaps(stage_windows)

        report_lines.append("## Full pipeline run")
        report_lines.append("")
        report_lines.append(f"Total wall time: **{extra['wall_total_seconds']:.1f}s** ({extra['wall_total_seconds']/60:.1f} min). Peak RSS: **{peak_rss:.0f} MB**. Avg CPU: {avg_cpu:.0f}%.")
        report_lines.append("")
        report_lines.append("| Stage | Time | Peak RSS | Avg CPU | Overlaps with |")
        report_lines.append("|---|---|---|---|---|")
        for stage in stage_windows:
            stage_peak = _stage_peak_rss(stage, samples)
            stage_cpu = _stage_avg_cpu(stage, samples)
            overlap_note = ", ".join(overlaps.get(stage["label"], [])) or "-"
            ok_note = "" if stage["ok"] else " ⚠️failed/timed out"
            report_lines.append(
                f"| {stage['label']}{ok_note} | {stage['seconds']:.1f}s | {stage_peak:.0f} MB | {stage_cpu:.0f}% | {overlap_note} |"
            )
        report_lines.append("")

        report_lines.append("**External calls made** (this run):")
        report_lines.append("")
        for host, count in extra["request_counts"].items():
            report_lines.append(f"- {host}: {count} call(s)")
        if not extra["request_counts"]:
            report_lines.append("- none recorded")
        report_lines.append("")

        if extra["report_asset_timings"]:
            report_lines.append("**Report assets sub-stage breakdown** (part of the \"Report assets\" row above):")
            report_lines.append("")
            report_lines.append("| Asset | Time |")
            report_lines.append("|---|---|")
            for name, seconds in sorted(extra["report_asset_timings"].items(), key=lambda kv: -kv[1]):
                report_lines.append(f"| {name} | {seconds:.1f}s |")
            report_lines.append("")

    summary_path = pipeline_dir / "site_summary.json"

    def _run_addon(phase: str, **kwargs):
        addon_dir = out_dir / phase
        wall, peak_rss, avg_cpu, _samples = _launch_worker(phase, addon_dir, **kwargs)
        addon_extra = json.loads((addon_dir / "profile_extra.json").read_text())
        return wall, peak_rss, avg_cpu, addon_extra

    def _format_groq_calls(calls: list[dict]) -> list[str]:
        lines = []
        for call in calls:
            lines.append(
                f"- `{call['caller']}`: {call['prompt_tokens']} prompt + "
                f"{call['completion_tokens']} completion = {call['total_tokens']} total tokens "
                f"(requested max {call['max_completion_tokens_requested']})"
            )
        return lines

    if "report" in phases_to_run and summary_path.exists():
        wall, peak_rss, avg_cpu, addon_extra = _run_addon("report", summary_path=str(summary_path))
        report_lines.append("## Generate Report add-on cost")
        report_lines.append("")
        report_lines.append(f"Time: **{wall:.1f}s**. Peak RSS: {peak_rss:.0f} MB.")
        report_lines.append("")
        report_lines.extend(_format_groq_calls(addon_extra["groq_calls"]))
        report_lines.append("")

    if "chat" in phases_to_run and summary_path.exists():
        wall, peak_rss, avg_cpu, addon_extra = _run_addon("chat", summary_path=str(summary_path))
        report_lines.append("## AI chat add-on cost")
        report_lines.append("")
        for q in addon_extra["questions"]:
            report_lines.append(f"- \"{q['question']}\" — {q['seconds']:.1f}s (tool used: {q['tool_used'] or 'none'})")
        report_lines.append("")
        report_lines.extend(_format_groq_calls(addon_extra["groq_calls"]))
        report_lines.append("")

    if "compare" in phases_to_run and summary_path.exists():
        other_dir = _REPO_ROOT / "runs" / args.other_run
        other_summary = other_dir / "site_summary.json"
        if other_summary.exists():
            wall, peak_rss, avg_cpu, addon_extra = _run_addon(
                "compare", summary_path=str(summary_path), other_summary_path=str(other_summary),
            )
            report_lines.append("## Compare add-on cost")
            report_lines.append("")
            report_lines.append(f"Paired with existing run `{args.other_run}` (no fresh pipeline run needed). Time: **{wall:.1f}s**. Peak RSS: {peak_rss:.0f} MB.")
            report_lines.append("")
            report_lines.extend(_format_groq_calls(addon_extra["groq_calls"]))
            report_lines.append("")
        else:
            report_lines.append(f"## Compare add-on cost\n\nSkipped — `runs/{args.other_run}/site_summary.json` not found.\n")

    if "pipeline" in phases_to_run:
        report_lines.append("## Cloud Run sizing")
        report_lines.append("")
        peak_overall = max(_stage_peak_rss(s, samples) for s in stage_windows) if stage_windows else 0.0
        report_lines.append(
            f"Observed peak RSS during the full pipeline run: **{peak_overall:.0f} MB**. "
            f"With headroom for concurrent requests and GC slack, a reasonable starting "
            f"point is **{max(1, round(peak_overall / 1024 * 2))} GB memory**, "
            f"revisited once a multi-site sweep confirms this isn't the lightest case. "
            f"CPU is mostly I/O-bound waiting on GEE/Overpass/SerpApi — 1-2 vCPU should "
            f"be sufficient unless report-asset chart/map rendering (matplotlib) becomes "
            f"a bottleneck under concurrent load."
        )
        report_lines.append("")

    report_path = _REPO_ROOT / "docs" / "PROFILING.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport written to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=list(_WORKERS.keys()), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--phase", default="all", help="Comma-separated: imports,pipeline,report,chat,compare (default: all)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon", type=float, default=DEFAULT_LON)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument("--other-summary-path", type=str, default=None)
    parser.add_argument("--other-run", type=str, default="betma", help="Existing runs/<name> to pair with for Compare")
    args = parser.parse_args()

    if args.worker:
        _WORKERS[args.worker](args)
    else:
        _run_driver(args)


if __name__ == "__main__":
    main()
