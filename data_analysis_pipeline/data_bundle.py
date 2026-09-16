"""Fetch the ~2.3 GB reference dataset (see ``config.DATA_DIR``) from a
pre-configured Google Drive zip (``config.REFERENCE_DATA_BUNDLE_URL``) — the
sidebar's "Download reference data" button calls :func:`download_and_extract`
directly. Optional: only needed to analyse a brand-new site; the bundled
demo runs under ``runs/`` never touch this.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

# Zip metadata that should never be treated as real data, regardless of
# where it appears in the archive (macOS's Finder "Compress" adds this).
_IGNORED_NAMES = {"__MACOSX", ".DS_Store"}


def download_and_extract(
    url: str,
    dest_dir: Path,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> None:
    """Download a zip from ``url`` (a Google Drive share link) and extract
    it into ``dest_dir``.

    Uses ``gdown`` rather than a plain HTTP GET because a Drive share link
    is a webpage, not a direct file URL, and large files (this one included)
    trigger Drive's "can't scan this file for viruses" confirmation
    interstitial — ``gdown`` handles both.

    Normalizes the archive's layout before moving anything into
    ``dest_dir``: a zip made by right-click-compressing a folder (confirmed
    with the actual bundle this was built against) contains one wrapping
    directory plus a macOS ``__MACOSX`` metadata folder — extracting that
    directly into ``dest_dir`` would nest everything one level too deep
    (``dest_dir/data/vb_soi_mp.GeoJSON`` instead of
    ``dest_dir/vb_soi_mp.GeoJSON``). This unwraps that single top-level
    folder if present (ignoring ``__MACOSX``/``.DS_Store``), so the result
    is the same either way: a zip with or without a wrapping folder both
    land as direct children of ``dest_dir``.

    Args:
        url: A Google Drive share link
            (``https://drive.google.com/file/d/<id>/view?usp=sharing``).
        dest_dir: Directory the reference files should end up directly
            inside of (e.g. ``config.DATA_DIR``) — created if missing.
        progress_callback: Called as ``progress_callback(bytes_so_far,
            bytes_total)`` after each downloaded chunk (``bytes_total`` is
            ``None`` if Drive didn't report a Content-Length) — e.g. to
            drive a ``st.progress`` bar. Optional.

    Raises:
        RuntimeError: If the download or extraction fails, with a message
            suitable for showing directly in the UI.
    """

    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        tmp_zip = tmp_dir_path / "reference_data.zip"
        try:
            result = gdown.download(
                url=url, output=str(tmp_zip), quiet=False, progress=progress_callback
            )
        except Exception as error:
            raise RuntimeError(f"Download failed: {error}") from error
        if result is None or not tmp_zip.exists():
            raise RuntimeError(
                "Download did not produce a file — the link may be wrong, "
                "not shared publicly, or Drive is rate-limiting this file."
            )

        extract_dir = tmp_dir_path / "extracted"
        try:
            with zipfile.ZipFile(tmp_zip) as archive:
                archive.extractall(extract_dir)
        except zipfile.BadZipFile as error:
            raise RuntimeError(
                f"Downloaded file isn't a valid zip: {error}. Drive may have "
                "served an HTML warning page instead of the file."
            ) from error

        source_dir = _unwrap_single_folder(extract_dir)
        for item in source_dir.iterdir():
            if item.name in _IGNORED_NAMES:
                continue
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))


def _unwrap_single_folder(extract_dir: Path) -> Path:
    """If ``extract_dir`` contains exactly one real entry (ignoring
    ``__MACOSX``/``.DS_Store``) and that entry is itself a directory, return
    it — the archive was a single wrapping folder around the actual data.
    Otherwise return ``extract_dir`` unchanged (the archive was already
    flat)."""

    entries = [p for p in extract_dir.iterdir() if p.name not in _IGNORED_NAMES]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir
