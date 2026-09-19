"""Hardened extraction/validation for the "Upload analysis" feature —
the only place in this app that treats a file as untrusted, adversarial
input (a public Streamlit deployment accepting arbitrary user uploads).
The zip is treated purely as data: bytes are read and JSON is parsed, but
nothing inside it is ever executed, imported, or trusted before every
check below passes. Security checks (path safety, size/count caps, the
per-run file allowlist) are fail-closed for the *whole* upload — one bad
member anywhere rejects everything, nothing is written to disk. Per-run
*schema* validation, by contrast, degrades gracefully — one invalid run
inside an otherwise-good multi-site zip is dropped, not the whole upload.
"""

from __future__ import annotations

import io
import json
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

from .runs import slugify_run_label

MAX_UPLOAD_ZIP_BYTES = 20 * 1024 * 1024  # 20 MB compressed
MAX_EXTRACTED_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MB uncompressed, across every run in the zip
MAX_MEMBER_COUNT = 600  # a real run has ~10-12 files; comfortably covers several runs
MAX_RUNS_PER_UPLOAD = 8  # generous over the "2-3 sites" use case, still a firm ceiling
_MAX_COMPRESSION_RATIO = 100  # file_size / compress_size above this looks like a zip bomb

_IGNORED_NAMES = {"__MACOSX", ".DS_Store"}
_TOP_LEVEL_ALLOWED = {"site_summary.json", "map.html", "report.pdf", "run.log"}
# Substituted for map.html when a run is uploaded without one — map.html is
# allowed but not mandatory (only site_summary.json is); this keeps every
# uploaded run's dict shape identical to a saved run's (a real, existing
# map_path), so nothing downstream needs to special-case a missing map.
_MISSING_MAP_PLACEHOLDER_HTML = (
    "<div style=\"padding:48px;text-align:center;color:#666;"
    "font-family:-apple-system,sans-serif;\">"
    "<p>No interactive map was included in this upload for this site.</p></div>"
)


class UploadValidationError(Exception):
    """A user-facing reason an uploaded zip (or one run inside it) was
    rejected — always safe to show directly in the UI."""


def _is_ignored(parts: tuple[str, ...]) -> bool:
    return any(part in _IGNORED_NAMES or part.startswith("._") for part in parts)


def _reject_unsafe_member(member: zipfile.ZipInfo) -> None:
    """Raise if this member's path could escape the extraction directory,
    or if it's a symlink (which could point somewhere unsafe once read
    later, even though it can't itself misdirect the extraction)."""

    name = member.filename
    if name.startswith("/") or name.startswith("\\"):
        raise UploadValidationError(f"Unsafe path in zip: {name!r}")
    if len(name) > 1 and name[1] == ":":  # Windows drive-letter absolute path
        raise UploadValidationError(f"Unsafe path in zip: {name!r}")
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if any(part == ".." for part in parts):
        raise UploadValidationError(f"Unsafe path in zip: {name!r}")

    mode = member.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise UploadValidationError(f"Symlinks aren't allowed in an uploaded zip: {name!r}")


def _relative_allowed(rel_parts: tuple[str, ...]) -> bool:
    """``rel_parts`` is a member's path with its run-folder prefix (if any)
    already stripped — must match one of the exact shapes a real LandIntel
    run produces."""

    if len(rel_parts) == 1:
        return rel_parts[0] in _TOP_LEVEL_ALLOWED
    if len(rel_parts) == 2 and rel_parts[0] == "report_assets":
        name = rel_parts[1]
        return name == "manifest.json" or name.lower().endswith(".png")
    return False


def extract_uploaded_bundle(zip_bytes: bytes, upload_filename: str) -> tuple[list[dict], list[str]]:
    """Safely extract one or more LandIntel runs from an uploaded zip.

    Two zip shapes are accepted: a flat single run (``site_summary.json``
    sitting at the zip root) or a multi-run bundle (one top-level folder
    per site, each shaped like a normal run directory).

    Returns:
        ``(runs, warnings)`` — ``runs`` is a list of dicts shaped exactly
        like ``data_analysis_pipeline.runs.list_previous_runs``'s output
        (``dir_name``, ``summary_path``, ``map_path``, ``label``,
        ``mtime``), so callers can treat uploaded and saved runs
        identically with no special-casing. ``warnings`` names any run
        that was dropped (schema validation failure) and why.

    Raises:
        UploadValidationError: the whole upload is rejected — oversized,
            not a valid zip, contains an unsafe or unexpected member, more
            sites than ``MAX_RUNS_PER_UPLOAD``, or zero runs validated.
    """

    if len(zip_bytes) > MAX_UPLOAD_ZIP_BYTES:
        raise UploadValidationError(
            f"That zip is too large ({len(zip_bytes) / 1e6:.1f} MB) — the limit is "
            f"{MAX_UPLOAD_ZIP_BYTES / 1e6:.0f} MB."
        )

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as error:
        raise UploadValidationError(f"That doesn't look like a valid zip file: {error}") from error

    members = [m for m in zf.infolist() if not m.is_dir()]
    if len(members) > MAX_MEMBER_COUNT:
        raise UploadValidationError(f"Too many files in this zip ({len(members)} > {MAX_MEMBER_COUNT}).")

    total_size = 0
    safe_members: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    for member in members:
        parts = PurePosixPath(member.filename.replace("\\", "/")).parts
        if _is_ignored(parts):
            continue
        _reject_unsafe_member(member)
        total_size += member.file_size
        if total_size > MAX_EXTRACTED_TOTAL_BYTES:
            raise UploadValidationError(
                f"This zip's uncompressed contents exceed the {MAX_EXTRACTED_TOTAL_BYTES / 1e6:.0f} MB limit."
            )
        if member.file_size > 0 and member.file_size / max(member.compress_size, 1) > _MAX_COMPRESSION_RATIO:
            raise UploadValidationError(f"Suspicious file in zip (compression ratio too high): {member.filename!r}")
        safe_members.append((member, parts))

    if not safe_members:
        raise UploadValidationError("This zip is empty.")

    # Detect layout: a flat single run (site_summary.json at the zip root)
    # vs. a multi-run bundle (one top-level folder per site).
    is_flat = any(parts == ("site_summary.json",) for _member, parts in safe_members)

    groups: dict[str, list[tuple[zipfile.ZipInfo, tuple[str, ...]]]] = {}
    if is_flat:
        run_name = slugify_run_label(Path(upload_filename).stem or "uploaded_run") or "uploaded_run"
        groups[run_name] = safe_members
    else:
        # Slugifying two different top-level folder names to the same slug
        # is rare but possible — keep them as distinct runs (via a numeric
        # suffix) rather than silently merging their files together.
        slug_for_raw_name: dict[str, str] = {}
        for member, parts in safe_members:
            if len(parts) < 2:
                raise UploadValidationError(
                    f"Unexpected file at the top level of a multi-site zip: {member.filename!r}"
                )
            raw_run_name, rel_parts = parts[0], parts[1:]
            if raw_run_name not in slug_for_raw_name:
                base_slug = slugify_run_label(raw_run_name) or "uploaded_run"
                run_name = base_slug
                suffix = 2
                while run_name in slug_for_raw_name.values():
                    run_name = f"{base_slug}_{suffix}"
                    suffix += 1
                slug_for_raw_name[raw_run_name] = run_name
            groups.setdefault(slug_for_raw_name[raw_run_name], []).append((member, rel_parts))

    if len(groups) > MAX_RUNS_PER_UPLOAD:
        raise UploadValidationError(
            f"This zip has {len(groups)} sites — the limit is {MAX_RUNS_PER_UPLOAD} per upload."
        )

    # Allowlist check, per group, BEFORE anything is written to disk.
    for run_name, group_members in groups.items():
        for member, rel_parts in group_members:
            if not _relative_allowed(rel_parts):
                raise UploadValidationError(
                    f"Unexpected file in '{run_name}': {member.filename!r} — only site_summary.json, "
                    "map.html, report.pdf, run.log, and report_assets/ are allowed."
                )

    # Every check passed for every member — now actually write bytes to disk.
    root_dir = Path(tempfile.mkdtemp(prefix="landintel_upload_"))
    for run_name, group_members in groups.items():
        run_dir = root_dir / run_name
        for member, rel_parts in group_members:
            target = run_dir.joinpath(*rel_parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))

    # Per-run schema validation — degrades gracefully, one bad run doesn't
    # sink the others.
    runs: list[dict] = []
    warnings: list[str] = []
    for run_name in groups:
        run_dir = root_dir / run_name
        try:
            site_summary = validate_run_directory(run_dir)
        except UploadValidationError as error:
            warnings.append(f"'{run_name}': {error}")
            continue

        map_path = run_dir / "map.html"
        if not map_path.exists():
            map_path.write_text(_MISSING_MAP_PLACEHOLDER_HTML)

        # Label format matches data_analysis_pipeline.runs.list_previous_runs
        # exactly, so uploaded and saved runs read identically wherever
        # they're offered side by side (e.g. Compare's candidate list).
        admin = site_summary.get("administration") or {}
        site = site_summary.get("site") or {}
        place = admin.get("village") or admin.get("district") or "unknown location"
        district = admin.get("district")
        place_label = f"{place}, {district}" if district and district != place else place
        status = "OK" if site_summary.get("run_success") else "partial failure"
        runs.append({
            "dir_name": run_name,
            "summary_path": run_dir / "site_summary.json",
            "map_path": run_dir / "map.html",
            "label": f"{run_name} — {place_label} (lat {site.get('latitude')}, lon {site.get('longitude')}, {status})",
            "mtime": time.time(),
        })

    if not runs:
        reason = f" ({'; '.join(warnings)})" if warnings else ""
        raise UploadValidationError(f"No valid LandIntel run was found in this zip.{reason}")

    return runs, warnings


def validate_run_directory(run_dir: Path) -> dict:
    """Validate one extracted run directory has the minimum shape this app
    needs, and that its ``report_assets/manifest.json`` (if any) can't be
    used to read files outside its own directory — a crafted manifest
    entry like ``"../../../etc/passwd"`` would otherwise be read verbatim
    by ``aI_agents.build_report._img_tag`` (``run_dir / rel_path``, no
    containment check there), so this re-checks every asset path here as
    defense in depth before the run is ever trusted.

    Returns:
        The parsed ``site_summary`` dict.

    Raises:
        UploadValidationError: with the specific reason.
    """

    summary_path = run_dir / "site_summary.json"
    if not summary_path.exists():
        raise UploadValidationError("missing site_summary.json")
    # map.html is allowed but not mandatory — see _MISSING_MAP_PLACEHOLDER_HTML.

    try:
        site_summary = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UploadValidationError(f"site_summary.json isn't valid JSON: {error}") from error
    if not isinstance(site_summary, dict):
        raise UploadValidationError("site_summary.json isn't a JSON object")
    if "schema_version" not in site_summary:
        raise UploadValidationError("site_summary.json is missing 'schema_version'")

    site = site_summary.get("site")
    if (
        not isinstance(site, dict)
        or not isinstance(site.get("latitude"), (int, float))
        or not isinstance(site.get("longitude"), (int, float))
    ):
        raise UploadValidationError("site_summary.json is missing a valid site.latitude/site.longitude")

    manifest_path = run_dir / "report_assets" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise UploadValidationError(f"report_assets/manifest.json isn't valid JSON: {error}") from error
        assets = (manifest or {}).get("assets") or {}
        run_dir_resolved = run_dir.resolve()
        for key, rel_path in assets.items():
            if not isinstance(rel_path, str):
                continue
            if Path(rel_path).is_absolute():
                raise UploadValidationError(f"manifest.json asset {key!r} is an absolute path")
            candidate = (run_dir / rel_path).resolve()
            if candidate != run_dir_resolved and run_dir_resolved not in candidate.parents:
                raise UploadValidationError(f"manifest.json asset {key!r} points outside the run directory")

    return site_summary
