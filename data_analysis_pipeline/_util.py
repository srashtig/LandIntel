"""Tiny stateless helpers shared by more than one ``get_*.py`` module —
kept separate from :mod:`aoi` since they're generic data-cleaning, not
geometry."""

from __future__ import annotations

import pandas as pd


def clean_text(value) -> str | None:
    """Return a stripped string, or None for null/empty values."""

    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None
