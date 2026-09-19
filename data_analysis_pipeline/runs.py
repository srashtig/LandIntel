"""Pure (no Streamlit) helpers for working with ``config.RUNS_DIR`` — run
directory naming and listing. Lives in the pipeline layer (not
``frontend/``) because both ``frontend/`` and ``aI_agents/`` (whose chat
tool routing lists previous runs to resolve a mentioned site) need it, and
``aI_agents/`` must not depend on ``frontend/``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

DEFAULT_LAT = 23.0401972
DEFAULT_LON = 76.2086806
DEFAULT_RADIUS_KM = 5.0


def reset_runs_dir(new_dir: str) -> None:
    """Change ``config.RUNS_DIR`` at runtime (persisted to ``.env``) — the
    sidebar's "Run directory" field calls this rather than
    ``config.reset_runs_dir`` directly, kept as a thin wrapper so callers
    only ever need to import this module, not ``config`` too."""

    config.reset_runs_dir(new_dir)


def slugify_run_label(label: str, max_length: int = 60) -> str:
    """Turn a free-text run label into a filesystem-safe directory-name
    fragment: lowercased, whitespace/anything-not-alnum collapsed to single
    underscores, trimmed of leading/trailing underscores, capped in length.

    Args:
        label: Free-text label as typed by the user.
        max_length: Maximum length of the returned slug.

    Returns:
        The sanitized slug, or "" if nothing valid remains (e.g. the label
        was empty or pure punctuation) — callers should fall back to the
        default naming scheme in that case, not use an empty directory name.
    """

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip()).strip("_").lower()
    return slug[:max_length].rstrip("_")


def make_run_dir_name(
    lat: float, lon: float, radius_km: float, label: Optional[str] = None, when: Optional[datetime] = None
) -> str:
    """Build the sortable, human-readable run directory name used under
    ``config.RUNS_DIR``: ``<timestamp>_<label>`` if a usable label was
    given, else the default ``<timestamp>_<lat>_<lon>_<radius>km``.

    Args:
        lat: Site latitude in decimal degrees.
        lon: Site longitude in decimal degrees.
        radius_km: AOI radius in kilometers.
        label: Optional free-text run label (see :func:`slugify_run_label`);
            None or a label that sanitizes to "" falls back to the default
            lat/lon/radius naming.
        when: Timestamp to use; defaults to ``datetime.now()`` (local time)
            if not given.

    Returns:
        A directory-name-safe string, e.g. ``20260909_231045_23.0402_76.2087_5km``
        or, with a label of "Farmhouse candidate 1", ``20260909_231045_farmhouse_candidate_1``.
    """

    when = when or datetime.now()
    stamp = when.strftime("%Y%m%d_%H%M%S")
    slug = slugify_run_label(label) if label else ""
    if slug:
        return f"{stamp}_{slug}"
    return f"{stamp}_{lat:.4f}_{lon:.4f}_{radius_km:g}km"


def resolve_run_output_dir(
    lat: float, lon: float, radius_km: float, label: Optional[str] = None, when: Optional[datetime] = None
) -> Path:
    """Full path (under ``config.RUNS_DIR``) a fresh analysis run should be
    persisted to; the directory itself is not created here."""

    return config.RUNS_DIR / make_run_dir_name(lat, lon, radius_km, label=label, when=when)


def unique_dir(base: Path) -> Path:
    """Return ``base`` if it doesn't exist yet, otherwise ``base`` with a
    numeric suffix (``_2``, ``_3``, ...) appended until a free path is
    found. Used both for a brand-new run directory and when renaming an
    existing one to a label that collides with another run.

    Args:
        base: The desired path.

    Returns:
        ``base``, or a sibling path with a numeric suffix, that does not
        currently exist on disk.
    """

    if not base.exists():
        return base
    n = 2
    while True:
        candidate = base.with_name(f"{base.name}_{n}")
        if not candidate.exists():
            return candidate
        n += 1


def list_previous_runs() -> list[dict]:
    """List completed analysis runs under ``config.RUNS_DIR`` that have a
    ``site_summary.json`` + ``map.html`` (the ``default/`` run is
    included — it's a real past run too), newest first.

    Runs are discovered from disk on every call rather than tracked only in
    ``st.session_state``, so a run from a previous app session (or produced
    by the CLI directly) shows up here too, not just ones started from this
    browser session.

    Returns:
        A list of dicts, one per run directory:
        ``{"dir_name", "summary_path", "map_path", "label", "mtime"}``.
        ``label`` is a human-readable one-liner (place, lat/lon, run status)
        built from that run's own ``site_summary.json``, for display in a
        selection widget.
    """

    if not config.RUNS_DIR.exists():
        return []

    runs = []
    for entry in config.RUNS_DIR.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "site_summary.json"
        map_path = entry / "map.html"
        if not summary_path.exists() or not map_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        admin = summary.get("administration") or {}
        site = summary.get("site") or {}
        place = admin.get("village") or admin.get("district") or "unknown location"
        district = admin.get("district")
        status = "OK" if summary.get("run_success") else "partial failure"
        place_label = f"{place}, {district}" if district and district != place else place
        label = f"{entry.name} — {place_label} (lat {site.get('latitude')}, lon {site.get('longitude')}, {status})"

        runs.append(
            {
                "dir_name": entry.name,
                "summary_path": summary_path,
                "map_path": map_path,
                "label": label,
                "mtime": summary_path.stat().st_mtime,
            }
        )

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def rename_run_dir(old_dir: Path, new_label: str) -> Path:
    """Rename a run directory to a new (sanitized) label — this is how a
    run's "label" is edited after the fact, since the directory name *is*
    the label (see :func:`make_run_dir_name`/:func:`slugify_run_label`).

    Args:
        old_dir: The run directory to rename (e.g. one returned by
            :func:`list_previous_runs`).
        new_label: Free-text new name as typed by the user.

    Returns:
        The new path, or ``old_dir`` unchanged if ``new_label`` sanitizes
        to the same name it already has (a no-op rename).

    Raises:
        ValueError: If ``new_label`` sanitizes to an empty string (see
            :func:`slugify_run_label`) — callers should show this to the
            user rather than silently doing nothing.
    """

    slug = slugify_run_label(new_label)
    if not slug:
        raise ValueError("Label must contain at least one letter or digit.")
    if slug == old_dir.name:
        return old_dir
    new_dir = unique_dir(old_dir.with_name(slug))
    old_dir.rename(new_dir)
    return new_dir


def failed_sections(site_summary: dict) -> list[str]:
    """List the top-level section names whose ``get_*`` module failed
    during this run (``{"available": False, ...}``), for the partial-
    failure warning banner. Ignores non-dict / non-section top-level keys
    like ``schema_version``/``run_success``/``generated_at``/``site``."""

    skip = {"schema_version", "run_success", "generated_at", "site"}
    failed = []
    for key, value in site_summary.items():
        if key in skip:
            continue
        if isinstance(value, dict) and value.get("available") is False:
            failed.append(key)
    return failed
