"""Beautified, PDF-downloadable HTML report for Compare (Phase 4).

Reuses :mod:`aI_agents.build_report`'s CSS/helpers (cards, icons, number
formatting, ``report_to_pdf_bytes``) for the same visual style as Generate
Report, adding: one "facts" card per site (built from :mod:`aI_agents.qa`'s
curated context — the exact same data the ranking was grounded in, so the
card and the AI reasoning never disagree on a number), a rank/score/verdict
badge per site, the "thinking" paragraph per site (see
:func:`aI_agents.qa.rank_sites`), and an overall summary.
"""

from __future__ import annotations

from .build_report import _CARD_CLOSE, _REPORT_CSS, _ai_box, _card_open, _esc, _fact, _fmt_num, _logo_img_tag
from .qa import _curate_site_context

_VERDICT_COLORS = {"Excellent": "#16a34a", "Good": "#2563eb", "Moderate": "#d97706", "Poor": "#dc2626"}


def _rank_badge(rank: int | None, score, verdict) -> str:
    if rank is None:
        return ""
    color = _VERDICT_COLORS.get(verdict, "#6b7280")
    score_text = f"{score:.0f}/100" if isinstance(score, (int, float)) else "n/a"
    return (
        f'<p style="margin:0 0 10px 0;"><span style="background:{color};color:#fff;'
        f'padding:2px 8px;font-size:11px;font-weight:700;">'
        f'#{rank} · {_esc(verdict or "?")} · {score_text}</span></p>'
    )


def _site_facts_card(label: str, site_summary: dict, rank: int | None, score, verdict) -> str:
    ctx = _curate_site_context(site_summary)
    admin = ctx.get("administration") or {}
    groundwater = ctx.get("groundwater") or {}
    land_cover = ctx.get("land_cover_classes") or []
    dominant = land_cover[0] if land_cover else {}
    town = ctx.get("nearest_developed_town") or {}
    price = ctx.get("land_price") or {}

    parts = [
        f'<div class="li-card"><h2>{_esc(label)}</h2>'
        + _rank_badge(rank, score, verdict)
    ]
    parts.append(_fact(
        "Village / Tehsil / District",
        f"{admin.get('village')} / {admin.get('tehsil')} / {admin.get('district')}",
    ))
    parts.append(_fact(
        # 5-year mean, not the latest single reading — see qa.py's curated-
        # context comment on why "latest" alone is misleading here.
        "Groundwater depth (5yr mean)", _fmt_num(groundwater.get("five_year_mean_m_bgl"), 1, " m bgl"),
        note=groundwater.get("dominant_trend"),
    ))
    if dominant:
        parts.append(_fact("Dominant land cover", dominant.get("name"), note=_fmt_num(dominant.get("percentage"), 1, "%")))
    if town:
        parts.append(_fact(
            "Nearest developed town", town.get("name"),
            note=_fmt_num(town.get("distance_km"), 1, " km") if town.get("distance_km") is not None else None,
        ))
    parts.append(_fact("Nearest road", _fmt_num(ctx.get("nearest_road_m"), 0, " m")))
    if price:
        parts.append(_fact(
            "Land price",
            f"₹{price['total_price']:,.0f}" if price.get("total_price") else None,
            note=f"₹{price['price_per_sqft']:,.2f}/sqft" if price.get("price_per_sqft") else None,
        ))
    parts.append(_CARD_CLOSE)
    return "".join(parts)


def _facts_table(site_labels: list[str], facts: list[tuple[str, list]]) -> str:
    header = "".join(f"<th>{_esc(label)}</th>" for label in site_labels)
    rows = "".join(
        "<tr><td>" + _esc(field) + "</td>" +
        "".join(f"<td>{_esc(v) if v is not None else 'n/a'}</td>" for v in values) +
        "</tr>"
        for field, values in facts
    )
    return (
        f'<table class="li-table"><thead><tr><th>Field</th>{header}</tr></thead><tbody>{rows}</tbody></table>'
    )


def generate_compare_html(
    sites: list[tuple[str, dict]],
    ranking_result: dict,
    facts: list[tuple[str, list]],
    purpose: str,
) -> str:
    """Build the full comparison report HTML — same card/icon/PDF-portable
    style as :func:`aI_agents.build_report.generate_report`.

    Args:
        sites: ``[(label, site_summary), ...]`` in the order compared.
        ranking_result: From :func:`aI_agents.qa_agent.rank_sites_with_tools` —
            ``{"ranking": [...], "summary": str, "tool_used": str | None}``.
        facts: From :func:`aI_agents.compare.side_by_side_facts`.
        purpose: The shared stated purpose these sites were compared for.
    """

    ranking = ranking_result.get("ranking") or []
    rank_info = {
        entry.get("label"): (i, entry.get("score"), entry.get("verdict"))
        for i, entry in enumerate(ranking, start=1)
    }

    tool_used = ranking_result.get("tool_used")
    tool_note = f" &middot; used a live tool while reasoning: {_esc(tool_used)}" if tool_used else ""
    hero = (
        '<div class="li-hero"><table class="li-hero-table" cellpadding="0" cellspacing="0" border="0"><tr><td>'
        '<p><span class="title-line">Site Comparison</span><br>'
        f"{len(sites)} sites compared for: {_esc(purpose)}{tool_note}</p>"
        f'</td><td class="li-hero-logo-cell">{_logo_img_tag()}</td></tr></table></div>'
    )

    cards = []
    for label, summary in sites:
        rank, score, verdict = rank_info.get(label, (None, None, None))
        cards.append(_site_facts_card(label, summary, rank, score, verdict))

    reasoning_parts = [_card_open("Reasoning")]
    for entry in ranking:
        verdict = entry.get("verdict") or "?"
        score = entry.get("score")
        score_text = f"{score:.0f}/100" if isinstance(score, (int, float)) else "n/a"
        reasoning_parts.append(f"<h3>{_esc(entry.get('label'))} — {_esc(verdict)} ({score_text})</h3>")
        reasoning_parts.append(_ai_box(entry.get("thinking")))
    reasoning_parts.append(_CARD_CLOSE)

    site_labels = [label for label, _ in sites]
    facts_card = (
        _card_open("Side-by-Side Facts")
        + _facts_table(site_labels, facts)
        + _CARD_CLOSE
    )

    overall_card = (
        _card_open("Overall Summary")
        + f'<div class="li-overall">{_esc(ranking_result.get("summary") or "No summary available.")}</div>'
        + _CARD_CLOSE
    )

    body = hero + "".join(cards) + "".join(reasoning_parts) + facts_card + overall_card

    # _REPORT_CSS's `.li-card` forces a new PDF page per card — right for
    # the main report, where each of its 5-6 chapters is genuinely long,
    # but a comparison of 2-4 sites has many small cards (one per site,
    # plus reasoning/facts/summary) that would otherwise pad a short
    # comparison out to several near-empty pages. Overridden back to a
    # natural page flow here (more specific selector wins the cascade
    # regardless of source order).
    _COMPARE_CSS_OVERRIDE = "<style>.li-report .li-card { page-break-before: auto; }</style>"

    return _REPORT_CSS + _COMPARE_CSS_OVERRIDE + '<div class="li-report">' + body + "</div>"
