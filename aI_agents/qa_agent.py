"""Phase 5 — free-form Q&A chat with tool routing.

Works for both a single site and a multi-site comparison context (pass
``additional_sites`` for the latter — same shape as
:func:`aI_agents.qa.answer_question`'s).

Routes a question through: (1) is this answerable at all, given known
tools/data? (2) is the curated site_summary excerpt (see
:mod:`aI_agents.qa`) enough on its own? (3) if not, call one bounded live
tool — an ad-hoc SerpApi nearby-category search, a onefivenine.com
tehsil-page fetch, or a Wikipedia summary — and ground the final answer in
that too.

Deliberately narrow (per the agreed MVP 0.1.1 scope): a small fixed tool
registry, not open-ended web search or arbitrary pipeline re-runs.
"""

from __future__ import annotations

import json

import requests
from bs4 import BeautifulSoup

from data_analysis_pipeline import config
from data_analysis_pipeline.get_nearby_places import _serpapi_search
from data_analysis_pipeline.runs import list_previous_runs
from .groq_client import _REASONING_EFFORT, _create_completion, _require_client

from .qa import _MAX_RANK_SITES, ChatUnavailable, _curate_site_context, answer_question  # noqa: F401 — re-exported

_ONEFIVENINE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_TOOLS_DESCRIPTION = """Available tools (pick at most one per question, or none if not needed):
- "nearby_category_search": an ad-hoc LIVE search (Google Maps, via SerpApi) for a category of place \
NOT already covered by this site's saved categories (tourist_attractions, restaurants, hospitals, \
schools, markets, hotels, industrial_businesses, warehouses_logistics) — e.g. "petrol pump", "bank", \
"ATM", "temple", "police station". Args: {"query": "<short category>"}.
- "onefivenine_tehsil_info": fetches a public page (onefivenine.com) with tehsil-level info NOT in \
this app's own pipeline data — village list, population, rivers, nearby railway stations, registered \
companies, schools/hospitals, bus stops, weather — for the site's own tehsil/district. No args needed. \
Only use for the PRIMARY site's tehsil (not available per comparison site).
- "wikipedia_summary": a short Wikipedia summary for a named place (a village/town/river/etc.), for \
general background not in this app's own data at all. Args: {"query": "<place name>"}.
"""


def _routing_system_prompt(
    known_context_keys: list[str], has_comparison_sites: bool, other_run_choices: list[dict]
) -> str:
    comparison_note = (
        "\nThis question may be about a comparison across `site` and one or more `comparison_sites`. "
        "The SAME data fields are already available for every one of those sites — a question like "
        "\"which has better road access\" or \"compare these for X\" is ALWAYS answerable from that data "
        "alone (`can_answer: true`, `tool: null`) without needing any tool at all. The tools below only "
        "ever fetch MORE data than what's already available, and only for the primary `site` — never "
        "treat 'a tool can't do X for comparison_sites' as a reason to refuse; the data (not a tool) "
        "already covers comparisons.\n"
        if has_comparison_sites
        else ""
    )
    other_sites_note = ""
    if other_run_choices:
        choices_text = "\n".join(f'  - "{r["dir_name"]}": {r["label"]}' for r in other_run_choices)
        other_sites_note = (
            "\n\nOther saved sites the user could be referring to by name, even if none are currently "
            f"selected for comparison (match loosely — a nickname, village name, or partial match is fine):\n"
            f"{choices_text}\n\n"
            "If the question names or clearly refers to one or more of these (e.g. \"compare X and Y\", "
            "\"which of A, B, C has...\"), list their exact ids in `mentioned_runs` so their data can be "
            "pulled in too — this alone is never a reason to set `can_answer` false.\n"
        )
    return (
        "You are the routing layer of a land-suitability Q&A assistant. Given a user's question, decide "
        "how it should be answered. You do not answer the question yourself here — only the routing.\n\n"
        f"Data already available about the site (and, in a comparison, every comparison site too — no "
        f"tool needed for any of this): {', '.join(known_context_keys)}."
        f"{comparison_note}{other_sites_note}\n\n{_TOOLS_DESCRIPTION}\n"
        "Decide:\n"
        "- `can_answer`: default to true whenever the question is about this land/site (or a comparison "
        "of the given sites) — the data already available covers most such questions on its own, with "
        "or without a tool. Set false ONLY for things genuinely outside all of that: legal advice, a "
        "specific person's identity, real-time information no tool here provides, or a topic unrelated "
        "to land/site suitability entirely. Never say false just because a tool's own docs mention a "
        "limitation (e.g. a tool only covering the primary site) — the underlying DATA may still answer "
        "it directly without that tool.\n"
        "- `tool`: the single most useful tool name from the list above, or null if the already-available "
        "data is enough on its own, or null if `can_answer` is false.\n"
        "- `tool_args`: the args object for that tool (per its spec above), or null.\n"
        "- `mentioned_runs`: a list of ids (exactly as given above, e.g. \"" +
        (other_run_choices[0]["dir_name"] if other_run_choices else "some_run_id") +
        "\") for any of the other saved sites listed above that this question names or clearly refers "
        "to — empty list if none, or if none were listed above.\n"
        "- `refusal_reason`: a short, plain-language reason, ONLY if `can_answer` is false, else null.\n\n"
        'Respond with ONLY a JSON object: {"can_answer": <true|false>, "tool": "<name>|null", '
        '"tool_args": <object|null>, "mentioned_runs": ["<id>", ...], "refusal_reason": "<string>|null"}'
    )


def _fetch_onefivenine_tehsil_info(site_summary: dict) -> str | None:
    admin = site_summary.get("administration") or {}
    district, tehsil = admin.get("district"), admin.get("tehsil")
    if not district or not tehsil:
        return None
    url = f"https://www.onefivenine.com/india/villag/{district.strip().replace(' ', '-')}/{tehsil.strip().replace(' ', '-')}"
    try:
        response = requests.get(url, headers={"User-Agent": _ONEFIVENINE_USER_AGENT}, timeout=15)
        response.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    if not text:
        return None
    # Bounded so this stays a small addition to the prompt, not a second
    # multi-KB context block — see aI_agents.qa's module docstring on why the
    # site's own curated context is kept small for the Groq free tier's
    # 8000-tokens-per-minute cap.
    return text[:6000]


def _nearby_category_search_text(site_summary: dict, query: str) -> str | None:
    if not query:
        return None
    site = site_summary.get("site") or {}
    lat, lon = site.get("latitude"), site.get("longitude")
    radius_km = site.get("radius_km") or 10.0
    if lat is None or lon is None or not config.SERPAPI_KEY or config.SERPAPI_DISABLED:
        return None
    try:
        results = _serpapi_search(lat, lon, config.SERPAPI_KEY, query, radius_km, max_results=10)
    except Exception:
        return None
    if not results:
        return f"Live search found no '{query}' within {radius_km} km of the site."
    lines = [
        f"- {place.get('name')} ({place.get('distance_km', 0):.2f} km) — {place.get('address', '')}"
        for place in results
    ]
    return f"Live search results for '{query}' near the site:\n" + "\n".join(lines)


def _wikipedia_summary_text(query: str) -> str | None:
    if not query:
        return None
    try:
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}",
            headers={"User-Agent": "LandIntel/1.0 (research tool)"},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        extract = response.json().get("extract")
        return f"Wikipedia summary for '{query}': {extract}" if extract else None
    except Exception:
        return None


_TOOL_FUNCS = {
    "onefivenine_tehsil_info": lambda site_summary, args: _fetch_onefivenine_tehsil_info(site_summary),
    "nearby_category_search": lambda site_summary, args: _nearby_category_search_text(
        site_summary, (args or {}).get("query", "")
    ),
    "wikipedia_summary": lambda site_summary, args: _wikipedia_summary_text((args or {}).get("query", "")),
}


def _other_run_choices(exclude_summary_path: str | None, limit: int = 30) -> list[dict]:
    """Up to ``limit`` other saved runs (excluding the current site), as
    ``{"dir_name", "label"}`` — offered to the routing prompt so a question
    can name another site by nickname/village even if it isn't already
    selected in Compare. Capped since this list is prompt-visible text."""

    try:
        runs = list_previous_runs()
    except Exception:
        return []
    choices = [
        {"dir_name": r["dir_name"], "label": r["label"]}
        for r in runs
        if str(r["summary_path"]) != exclude_summary_path
    ]
    return choices[:limit]


def _decide_and_fetch_tool(
    site_summary: dict,
    question: str,
    has_comparison_sites: bool = False,
    exclude_summary_path: str | None = None,
) -> tuple[bool, str | None, str | None, list[str], str | None]:
    """The shared routing step: decide whether ``question`` can be answered
    at all, whether one live tool should be fetched first (grounded against
    ``site_summary`` only — tools never run per comparison site, see
    ``_TOOLS_DESCRIPTION``), and whether it names any other saved site by
    nickname/village (e.g. "which of X and Y has better security").

    Returns:
        ``(can_answer, tool_name, tool_result_text, mentioned_dir_names,
        refusal_reason)``.

    Raises:
        ChatUnavailable: If no Groq API key is configured.
    """

    client = _require_client()

    context_keys = sorted(_curate_site_context(site_summary).keys())
    other_run_choices = _other_run_choices(exclude_summary_path)
    system_prompt = _routing_system_prompt(context_keys, has_comparison_sites, other_run_choices)

    response = _create_completion(
        client,
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        reasoning_effort=_REASONING_EFFORT,
        max_completion_tokens=300,
    )
    raw = response.choices[0].message.content or "{}"
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {}
    if not isinstance(decision, dict):
        decision = {}

    if decision.get("can_answer") is False:
        reason = decision.get("refusal_reason") or "I don't have a way to answer that with this app's data/tools."
        return False, None, None, [], reason

    tool_name = decision.get("tool")
    tool_result_text = None
    if tool_name in _TOOL_FUNCS:
        tool_result_text = _TOOL_FUNCS[tool_name](site_summary, decision.get("tool_args"))

    valid_ids = {r["dir_name"] for r in other_run_choices}
    mentioned = [d for d in (decision.get("mentioned_runs") or []) if isinstance(d, str) and d in valid_ids]

    return True, (tool_name if tool_result_text else None), tool_result_text, mentioned, None


def route_and_answer(
    site_summary: dict,
    question: str,
    purpose: str | None = None,
    preferences: str | None = None,
    history: list[dict[str, str]] | None = None,
    site_label: str | None = None,
    additional_sites: list[tuple[str, dict]] | None = None,
    summary_path: str | None = None,
) -> dict:
    """Route ``question`` and answer it — works for a single site or (via
    ``additional_sites``) a multi-site comparison chat, same as
    :func:`aI_agents.qa.answer_question`. Also resolves other saved sites named
    in the question itself (e.g. "which of X and Y has better security"),
    even if they weren't pre-selected via ``additional_sites`` — per your
    steer that chat should understand that kind of question directly.

    Args:
        summary_path: This site's own ``site_summary.json`` path (as a
            string) — used only to exclude it from the "other saved sites"
            list offered to the routing step.

    Returns:
        ``{"answer": str, "tool_used": str | None, "refused": bool,
        "mentioned_sites": list[str]}`` — the labels of any other sites
        pulled in by name, for the caller to show ("also grounded in: ...").

    Raises:
        ChatUnavailable: If no Groq API key is configured.
    """

    can_answer, tool_name, tool_result_text, mentioned_dir_names, refusal_reason = _decide_and_fetch_tool(
        site_summary, question, has_comparison_sites=bool(additional_sites), exclude_summary_path=summary_path
    )
    if not can_answer:
        return {"answer": refusal_reason, "tool_used": None, "refused": True, "mentioned_sites": []}

    combined_sites = list(additional_sites or [])
    already_included = {label for label, _ in combined_sites}
    if mentioned_dir_names:
        for run in list_previous_runs():
            if run["dir_name"] not in mentioned_dir_names or run["dir_name"] in already_included:
                continue
            try:
                mentioned_summary = json.loads(run["summary_path"].read_text())
            except Exception:
                continue
            combined_sites.append((run["dir_name"], mentioned_summary))
            already_included.add(run["dir_name"])

    # Keep the combined grounding context within the Groq free tier's
    # 8000-tokens-per-minute cap (same reasoning as aI_agents.qa._MAX_RANK_SITES).
    if len(combined_sites) > _MAX_RANK_SITES - 1:
        combined_sites = combined_sites[: _MAX_RANK_SITES - 1]

    answer = answer_question(
        site_summary,
        question,
        purpose=purpose,
        preferences=preferences,
        history=history,
        site_label=site_label,
        additional_sites=combined_sites or None,
        extra_context=tool_result_text,
    )
    included_labels = {label for label, _ in combined_sites}
    return {
        "answer": answer,
        "tool_used": tool_name,
        "refused": False,
        "mentioned_sites": [d for d in mentioned_dir_names if d in included_labels],
    }


def rank_sites_with_tools(
    sites: list[tuple[str, dict]],
    purpose: str,
    preferences: str | None = None,
) -> dict:
    """:func:`aI_agents.qa.rank_sites`, but first lets the routing layer fetch one
    live tool result (grounded against the primary/first site — e.g. its
    tehsil's onefivenine.com page, or an ad-hoc nearby-category search) if
    the purpose/preferences suggest one would help, per your "verbal
    (thinking out loud based on facts and tools)" steer — the ranking then
    reasons over both the sites' real data AND that live context.

    Returns:
        Same shape as :func:`aI_agents.qa.rank_sites`, plus a ``"tool_used"`` key
        (``str | None``).

    Raises:
        ValueError: Fewer than 2 or more than
            :data:`aI_agents.qa._MAX_RANK_SITES` sites.
        ChatUnavailable: If no Groq API key is configured.
    """

    from .qa import rank_sites

    primary_label, primary_summary = sites[0]
    question = (
        f"Ranking {len(sites)} sites for the purpose '{purpose}'. Is there a live tool that would "
        "meaningfully help judge the primary site (and, by extension, this ranking) — e.g. tehsil-level "
        "context, or a specific nearby-category search implied by the purpose/preferences?"
    )
    _can_answer, tool_name, tool_result_text, _mentioned, _refusal = _decide_and_fetch_tool(
        primary_summary, question, has_comparison_sites=len(sites) > 1
    )

    result = rank_sites(sites, purpose, preferences=preferences, extra_context=tool_result_text)
    result["tool_used"] = tool_name
    return result
