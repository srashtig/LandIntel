"""Compare (Phase 4) — a ranked shortlist across 2+ sites, no scoring engine.

:func:`aI_agents.qa.rank_sites` grounds an LLM-judged score/verdict per site
in each site's curated data (never a deterministic formula) — this module is
a thin wrapper adding a side-by-side fact table drawn from the exact same
curated data the ranking was grounded in, so the table and the ranking never
disagree on the underlying numbers.
"""

from __future__ import annotations

from .qa import _MAX_RANK_SITES, ChatUnavailable, _curate_site_context  # noqa: F401 — re-exported
from .qa_agent import rank_sites_with_tools

MAX_RANK_SITES = _MAX_RANK_SITES

def _round(value, decimals: int = 1):
    """Round a numeric fact value for display — fixes raw floats like
    ``650.37498650989`` showing up verbatim in the side-by-side table."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return int(round(value)) if decimals == 0 else round(value, decimals)


_FACT_FIELDS = [
    ("Village / Tehsil", lambda c: f"{(c.get('administration') or {}).get('village')} / {(c.get('administration') or {}).get('tehsil')}"),
    # 5-year mean, not the latest single reading — pre/post-monsoon levels
    # swing by several metres, so "latest" alone is misleading as "the"
    # groundwater depth (see qa.py's curated-context comment on this).
    ("Groundwater depth, 5yr mean (m bgl)", lambda c: _round((c.get("groundwater") or {}).get("five_year_mean_m_bgl"), 1)),
    ("Groundwater trend", lambda c: (c.get("groundwater") or {}).get("dominant_trend")),
    ("Nearest developed town", lambda c: (c.get("nearest_developed_town") or {}).get("name")),
    ("  ...distance (km)", lambda c: _round((c.get("nearest_developed_town") or {}).get("distance_km"), 1)),
    ("Nearest road (m)", lambda c: _round(c.get("nearest_road_m"), 0)),
    ("Mean slope (deg)", lambda c: _round((c.get("terrain") or {}).get("slope_mean_deg"), 1)),
    ("NDVI mean", lambda c: _round(c.get("ndvi_mean"), 2)),
    ("Land price (total, INR)", lambda c: _round((c.get("land_price") or {}).get("total_price"), 0)),
    ("Price per sqft (INR)", lambda c: _round((c.get("land_price") or {}).get("price_per_sqft"), 2)),
]


def side_by_side_facts(sites: list[tuple[str, dict]]) -> list[tuple[str, list]]:
    """``[(field_label, [value_per_site, ...]), ...]`` in the same order as
    ``sites``, for a simple N-column comparison table."""

    contexts = [_curate_site_context(summary) for _label, summary in sites]
    return [(label, [getter(ctx) for ctx in contexts]) for label, getter in _FACT_FIELDS]


def compare(
    sites: list[tuple[str, dict]],
    purpose: str,
    preferences: str | None = None,
) -> tuple[dict, list[tuple[str, list]]]:
    """Returns ``(ranking_result, facts_table)`` for 2+ ``(label,
    site_summary)`` sites — see :func:`aI_agents.qa.rank_sites` for
    ``ranking_result``'s shape and :func:`side_by_side_facts` for the table.

    Raises:
        ValueError: Fewer than 2 or more than :data:`aI_agents.qa._MAX_RANK_SITES` sites.
        ChatUnavailable: If no Groq API key is configured.
    """

    ranking_result = rank_sites_with_tools(sites, purpose, preferences=preferences)
    facts = side_by_side_facts(sites)
    return ranking_result, facts
