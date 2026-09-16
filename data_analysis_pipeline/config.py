"""Paths, environment variables and Earth Engine bootstrap for the Land
Intelligence pipeline, plus the Groq API settings the AI layer
(``aI_agents/``) reads from here too — one consolidated config for the whole
package, loaded from one top-level ``.env``.

Path layout (this file lives at
``<repo root>/data_analysis_pipeline/config.py``)::

    BASE_DIR   -> <repo root>/       (parents[1])
    DATA_DIR   -> LAND_INTEL_DATA_DIR env var, else <repo root>/data/
                  (the ~2.3 GB reference dataset — see LAND_INTEL_DATA_DIR
                  in .env.example, and the "Download reference data" sidebar
                  button, which fetches it from REFERENCE_DATA_BUNDLE_URL)
    CACHE_DIR  -> <repo root>/cache/
    RUNS_DIR   -> <repo root>/runs/

Every setting below (API keys + DATA_DIR) can also be changed at runtime —
see :func:`set_env_key` / :func:`reset_data_dir` — which is what the
Streamlit sidebar's "Save keys" / "Data directory" controls call. Every
consumer elsewhere in this codebase reads these as ``config.ATTR`` (never a
``from data_analysis_pipeline.config import ATTR``-style name import), so
mutating the module attribute here takes effect immediately, no restart
needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENGINE_VERSION = "0.1.0"

# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------

# .../data_analysis_pipeline/config.py
#   parents[0] -> data_analysis_pipeline/
#   parents[1] -> repo root
_THIS_FILE = Path(__file__).resolve()
BASE_DIR = _THIS_FILE.parents[1]

# Loaded early (before DATA_DIR below) so LAND_INTEL_DATA_DIR can be set via
# .env, not just the real shell environment.
load_dotenv(BASE_DIR / ".env")

CACHE_DIR = BASE_DIR / "cache"

# Where analysis runs (site_summary.json/map.html/report_assets/ per run)
# are saved to and loaded from. Overridable via LAND_INTEL_RUNS_DIR (e.g.
# the sidebar's "Run directory" field, or a shared/synced folder) so runs
# aren't locked to living inside this package.
RUNS_DIR = Path(
    os.environ.get("LAND_INTEL_RUNS_DIR", str(BASE_DIR / "runs"))
).expanduser().resolve()


def reset_runs_dir(new_dir: str) -> None:
    """Change ``RUNS_DIR`` at runtime, and persist it — the sidebar's "Run
    directory" field calls this."""

    global RUNS_DIR
    new_dir = new_dir.strip()
    os.environ["LAND_INTEL_RUNS_DIR"] = new_dir
    RUNS_DIR = Path(new_dir).expanduser().resolve()
    _upsert_env_file("LAND_INTEL_RUNS_DIR", new_dir)


def _recompute_data_paths() -> None:
    """(Re)compute ``DATA_DIR`` and every path derived from it, as module
    globals. Called once below at import time, and again by
    :func:`reset_data_dir` after the directory changes at runtime (e.g. via
    the sidebar) — these seven files/dirs are the only unchanging part of
    that recomputation, ``DATA_DIR`` itself is the only real input.

    Defaults to ``data/`` (inside the repo root) — self-contained, no
    assumption that a folder exists one level above the repo root.
    Overridable via ``LAND_INTEL_DATA_DIR`` so a reviewer's data can live
    anywhere.
    """

    global DATA_DIR, VILLAGE_BOUNDARY_FILE, RIVER_POLYGON_FILE, SAC_WATERBODY_FILE
    global CENSUS_FEATURES_FILE, GROUNDWATER_FILE, GUIDELINE_RATES_DIR, IMD_RAINFALL_FILES

    DATA_DIR = Path(
        os.environ.get("LAND_INTEL_DATA_DIR", str(BASE_DIR / "data"))
    ).expanduser().resolve()

    # Source data files (read-only, never modified by this package).
    VILLAGE_BOUNDARY_FILE = DATA_DIR / "vb_soi_mp.GeoJSON"
    RIVER_POLYGON_FILE = DATA_DIR / "river_polygon.GeoJSON"
    SAC_WATERBODY_FILE = DATA_DIR / "wb_sac_mp.GeoJSON"
    CENSUS_FEATURES_FILE = DATA_DIR / "census_derieved.csv"
    GROUNDWATER_FILE = DATA_DIR / "ground_water_level_manual_monthly_madhya_pradesh_1974_2025.csv"
    # MP Govt district guideline-rate (circle-rate) tables, FY2026-27, one CSV
    # per district -- see scripts/build_district_guideline_tables.py.
    GUIDELINE_RATES_DIR = DATA_DIR / "2026_guidelines_MP_dfs"
    IMD_RAINFALL_FILES = {
        2023: DATA_DIR / "RF25_ind2023_rfp25.nc",
        2024: DATA_DIR / "RF25_ind2024_rfp25.nc",
        2025: DATA_DIR / "RF25_ind2025_rfp25.nc",
    }


_recompute_data_paths()

# Small, deliberately-committed precomputed asset (built by
# scripts/build_mp_boundary_cache.py) — ships alongside the package code
# itself, not in CACHE_DIR (which also holds ephemeral request-cache blobs
# that aren't meant to be committed).
MP_BOUNDARY_CACHE = BASE_DIR / "data_analysis_pipeline" / "mp_boundary.geojson"
LOCATION_SEARCH_INDEX = CACHE_DIR / "mp_location_index.parquet"

# ---------------------------------------------------------------------
# Runtime setting helpers — used by the sidebar's key/data-directory
# controls to change a setting without restarting the process.
# ---------------------------------------------------------------------


def _upsert_env_file(name: str, value: str) -> None:
    """Replace the ``NAME=...`` line in ``.env`` if present,
    else append one. Preserves every other line untouched."""

    env_path = BASE_DIR / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    prefix = f"{name}="
    new_line = f"{name}={value}"
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    env_path.write_text("\n".join(lines) + "\n")


def set_env_key(name: str, value: str) -> None:
    """Set one API-key-style setting at runtime: updates ``os.environ``,
    this module's matching attribute, and upserts it into
    ``.env`` — so a value entered via the sidebar takes effect
    immediately for every future call (every consumer reads ``config.ATTR``
    fresh, not a name-imported copy) and survives a restart.

    Args:
        name: The setting's name — also its attribute name on this module
            and its ``.env``/env-var key (e.g. ``"GROQ_API_KEY"``).
        value: The new value. Stored as ``None`` if blank, so existing
            ``if not config.X`` checks throughout the codebase keep working
            unchanged.
    """

    value = value.strip()
    os.environ[name] = value
    globals()[name] = value or None
    _upsert_env_file(name, value)


def reset_data_dir(new_dir: str) -> None:
    """Change ``DATA_DIR`` (and everything derived from it) at runtime, and
    persist it — the sidebar's "Data directory" field calls this."""

    new_dir = new_dir.strip()
    os.environ["LAND_INTEL_DATA_DIR"] = new_dir
    _recompute_data_paths()
    _upsert_env_file("LAND_INTEL_DATA_DIR", new_dir)


# ---------------------------------------------------------------------
# API keys (.env loading is done above, right after BASE_DIR is resolved)
# ---------------------------------------------------------------------

# The original notebook read the SerpApi/Groq keys from these absolute,
# out-of-repo paths. SERPAPI_KEY/GROQ_API_KEY (from .env or the
# real environment) take precedence; these paths are kept only as a
# fallback so existing local setups keep working without extra
# configuration.
_LEGACY_SERPAPI_KEY_FILE = Path(
    "/Users/srashtigoyal/Documents/ML_stuffs/ai_engineering course/serpapi_key"
)
_LEGACY_GROQ_KEY_FILE = Path(
    "/Users/srashtigoyal/Documents/ML_stuffs/ai_engineering course/groq_key.txt"
)


def _resolve_serpapi_key() -> str:
    """Resolve the SerpApi API key: ``SERPAPI_KEY`` env var first (from
    .env or the real environment), then the notebook's legacy
    absolute-path key file, as a backward-compatible fallback.

    Returns:
        The API key string, or ``""`` if neither source has one — every
        consumer already degrades gracefully via ``if not
        config.SERPAPI_KEY``, so this must not raise (a fresh install with
        zero keys configured needs to be able to start the app at all, and
        add a key later from the sidebar).
    """

    key = os.environ.get("SERPAPI_KEY", "").strip()
    if key:
        return key

    if _LEGACY_SERPAPI_KEY_FILE.exists():
        key = _LEGACY_SERPAPI_KEY_FILE.read_text().strip()
        if key:
            return key

    return ""


SERPAPI_KEY = _resolve_serpapi_key()

# Temporary kill switch for when the SerpApi account is at/near its usage
# limit — every SerpApi call site (get_nearby_places, custom_facts's
# transit lookup, location_search's place search, and aI_agents.qa_agent's
# chat tool) already degrades gracefully to "unavailable" when SERPAPI_KEY
# itself is missing; this flag routes them down that exact same
# already-existing path without needing to remove/break the key itself. Set
# DISABLE_SERPAPI=1 in .env to turn it on; unset (or 0/false) to
# go back to normal.
SERPAPI_DISABLED = os.environ.get("DISABLE_SERPAPI", "").strip().lower() in ("1", "true", "yes")

GEE_PROJECT_ID = os.environ.get("GEE_PROJECT_ID", "").strip() or None


def _resolve_groq_key() -> str | None:
    """Resolve the Groq API key: ``GROQ_API_KEY`` env var first (from
    .env or the real environment), then a legacy absolute-path
    key file, as a backward-compatible fallback.

    Unlike :func:`_resolve_serpapi_key`, this does not raise if no key is
    found — the AI layer (``aI_agents/``) must degrade gracefully (raise
    ``ChatUnavailable``, caught by the UI) rather than crash the app when no
    key is configured.

    Returns:
        The API key string, or ``None`` if neither source has one.
    """

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    if _LEGACY_GROQ_KEY_FILE.exists():
        key = _LEGACY_GROQ_KEY_FILE.read_text().strip()
        if key:
            return key

    return None


GROQ_API_KEY = _resolve_groq_key()

# gpt-oss-20b: confirmed live on Groq (via a real /models list + a real
# chat.completions.create call, both JSON-mode and plain-text) at the time
# this was written — small/cheap, good fit for structured preference
# parsing and grounded explanation. Override via GROQ_MODEL for a different
# Groq-hosted model as their lineup changes.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"

# Set once by whoever deploys this app (in .env) — not exposed
# as a sidebar text input. Points at a zip of the reference dataset (see
# DATA_DIR above); the sidebar's "Download reference data" button fetches
# from this fixed URL, so no end user ever needs to paste a link in
# themselves. See data_analysis_pipeline/data_bundle.py.
REFERENCE_DATA_BUNDLE_URL = os.environ.get("REFERENCE_DATA_BUNDLE_URL", "").strip() or None

# ---------------------------------------------------------------------
# Earth Engine bootstrap
# ---------------------------------------------------------------------

_ee_initialized = False


def init_earth_engine() -> None:
    """Initialize Earth Engine once per process.

    Uses ``GEE_PROJECT_ID`` if set; otherwise falls back to whatever
    already-authenticated ADC/default behavior works today (this mirrors
    the notebook's bare ``ee.Initialize()`` call, which succeeds without
    an explicit project on this machine).
    """

    global _ee_initialized
    if _ee_initialized:
        return

    import ee

    try:
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if GEE_PROJECT_ID:
            ee.Initialize(project=GEE_PROJECT_ID)
        else:
            ee.Initialize()

    _ee_initialized = True


def reset_earth_engine() -> None:
    """Force the next :func:`init_earth_engine` call to actually
    re-initialize — call this after changing ``GEE_PROJECT_ID`` at runtime
    (e.g. via the sidebar), since it otherwise no-ops for the life of the
    process once already initialized once."""

    global _ee_initialized
    _ee_initialized = False
