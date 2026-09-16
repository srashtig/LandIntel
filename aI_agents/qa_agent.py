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
from data_analysis_pipeline.aoi import haversine_km
from data_analysis_pipeline.custom_facts import fetch_road_travel_time
from data_analysis_pipeline.get_nearby_places import _serpapi_search
from data_analysis_pipeline.location_search import search_google_maps
from data_analysis_pipeline.runs import list_previous_runs
from .groq_client import _REASONING_EFFORT, _create_completion, _require_client

from .qa import (  # noqa: F401 — re-exported
    _MAX_HISTORY_TURNS,
    _MAX_RANK_SITES,
    ChatUnavailable,
    _curate_site_context,
    answer_question,
)

_ONEFIVENINE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_TOOLS_DESCRIPTION = """Available tools (pick at most one per question, or none if not needed):
- "nearby_category_search": an ad-hoc LIVE search (Google Maps, via SerpApi) for a category of place. \
Use it for a category NOT already covered by this site's saved categories (tourist_attractions, \
restaurants, hospitals, schools, markets, hotels, industrial_businesses, warehouses_logistics) — e.g. \
"petrol pump", "bank", "ATM", "temple", "police station" — OR for a SAVED category too if the question \
asks about a radius wider than this site's own analysis radius (e.g. "tourist attractions within 50 km" \
when the site was analyzed at a smaller radius). Args: {"query": "<short category>", "radius_km": \
<number, ONLY if the question states a specific distance — omit entirely to use the site's own default>, \
"sort_by": "rating" <ONLY if the question asks for the "most popular/famous/best/top-rated" places, \
to rank by actual rating + review count instead of nearest-first — omit entirely otherwise>}. Results \
always include each place's rating and number of reviews when available.
- "onefivenine_tehsil_info": fetches a public page (onefivenine.com) with tehsil-level info NOT in \
this app's own pipeline data — village list, population, rivers, nearby railway stations, registered \
companies, schools/hospitals, bus stops, weather — for the site's own tehsil/district. No args needed. \
Only use for the PRIMARY site's tehsil (not available per comparison site).
- "wikipedia_summary": a short Wikipedia summary for a named place (a village/town/river/etc.), for \
general background not in this app's own data at all. Args: {"query": "<place name>"}.
- "road_distance_to_place": geocodes a named place (via Google Maps/SerpApi) and computes the road \
driving distance/time from the PRIMARY site to it (via OSRM), plus straight-line distance as a \
fallback figure — use this for ANY "how far/how long to <named place>" or "distance to <named place>" \
question about a place that isn't the site's own saved reference_location. Args: {"place": "<place name>"}.
"""


def _format_recent_history(history: list[dict[str, str]] | None, max_turns: int = _MAX_HISTORY_TURNS) -> str:
    """Render the last ``max_turns`` (user, assistant) exchanges from
    ``history`` as plain text, for follow-up resolution in the routing
    prompt — deliberately NOT passed as actual chat messages (see
    :func:`_decide_and_fetch_tool`'s docstring for why: mixing prose
    assistant turns into this JSON-mode call was confirmed to make the
    model occasionally emit a native-style tool call instead of the
    requested plain JSON, which Groq then rejects outright).

    Each turn's content is capped at 300 characters so a long prior answer
    can't blow up this call's token budget.
    """

    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()[:300]
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _routing_system_prompt(
    known_context_keys: list[str],
    has_comparison_sites: bool,
    other_run_choices: list[dict],
    recent_history_text: str = "",
) -> str:
    history_note = (
        f"\n\nRecent conversation so far (most recent last — use this ONLY to resolve a short/vague "
        f"follow-up question like \"by road?\" or \"what about the second one?\"; the question you're "
        f"routing right now is the LATEST one, given separately below, not part of this block):\n"
        f"{recent_history_text}\n"
        if recent_history_text
        else ""
    )
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
        "how it should be answered. You do not answer the question yourself here — only the routing. A "
        "short or vague-looking question (e.g. \"by road?\", \"what about the second one?\") is very often "
        "a follow-up that only makes sense together with the immediately preceding turn — if a recent "
        "conversation is given below, resolve it using that before ever concluding it's incomplete; only "
        "treat it as genuinely unanswerable if it's still unclear once that context is taken into "
        "account.\n"
        f"{history_note}\n"
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


_DEFAULT_NEARBY_SEARCH_RADIUS_KM = 10.0
# Bounds how far a user-requested radius can push a single SerpApi call —
# large enough for a genuine "within 50 km" question, small enough that one
# search can't balloon into a huge, expensive result set/tool_result_text.
_MAX_NEARBY_SEARCH_RADIUS_KM = 100.0
# Displayed results are capped small on purpose: this tool's output goes
# straight into the answering call's prompt, and a long list here is exactly
# the kind of per-call growth that eats into the Groq free tier's tight
# tokens-per-minute/tokens-per-day budget for no real benefit — 5 good
# matches answers "what/where" just as well as 10 would.
_MAX_NEARBY_RESULTS = 5


def _nearby_category_search_text(site_summary: dict, query: str, radius_km=None, sort_by: str | None = None) -> str | None:
    if not query:
        return None
    site = site_summary.get("site") or {}
    lat, lon = site.get("latitude"), site.get("longitude")
    try:
        radius_km = float(radius_km)
        if radius_km <= 0:
            raise ValueError
    except (TypeError, ValueError):
        # `site["aoi_radius_km"]` is this run's own analysis radius (the key
        # a previous version of this function got wrong: it read a
        # nonexistent "radius_km" field and always silently fell back to
        # 10.0, ignoring even that).
        radius_km = site.get("aoi_radius_km") or _DEFAULT_NEARBY_SEARCH_RADIUS_KM
    radius_km = min(radius_km, _MAX_NEARBY_SEARCH_RADIUS_KM)
    if lat is None or lon is None or not config.SERPAPI_KEY or config.SERPAPI_DISABLED:
        return None
    by_rating = sort_by == "rating"
    try:
        # Only ranking by rating needs a bigger pool to choose from first
        # (SerpApi's own distance-nearest ordering isn't what we want to
        # truncate on before re-ranking, or a genuinely most-famous-but-
        # farther-out place could get cut before it's ever considered) — the
        # final displayed list is capped at _MAX_NEARBY_RESULTS either way,
        # to keep this tool's contribution to the Groq prompt small.
        results = _serpapi_search(
            lat, lon, config.SERPAPI_KEY, query, radius_km,
            max_results=20 if by_rating else _MAX_NEARBY_RESULTS,
        )
    except Exception:
        return None
    if not results:
        return f"Live search found no '{query}' within {radius_km:g} km of the site."
    if by_rating:
        results = sorted(
            results,
            key=lambda place: (place.get("rating") or 0, place.get("reviews") or 0),
            reverse=True,
        )
    results = results[:_MAX_NEARBY_RESULTS]
    lines = []
    for place in results:
        rating = place.get("rating")
        reviews = place.get("reviews")
        rating_text = "N/A" if rating is None else f"{rating}"
        if reviews is not None:
            rating_text += f" ({reviews} reviews)"
        lines.append(
            f"- {place.get('name')} ({place.get('distance_km', 0):.2f} km) — "
            f"rating: {rating_text} — {place.get('address', '')}"
        )
    order_note = "sorted by rating (highest first)" if by_rating else "nearest first"
    return f"Live search results for '{query}' within {radius_km:g} km of the site ({order_note}):\n" + "\n".join(lines)


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


def _road_distance_text(site_summary: dict, place: str) -> str | None:
    if not place:
        return None
    site = site_summary.get("site") or {}
    site_lat, site_lon = site.get("latitude"), site.get("longitude")
    if site_lat is None or site_lon is None:
        return None

    if not config.SERPAPI_KEY or config.SERPAPI_DISABLED:
        return f"Live place lookup for '{place}' is unavailable (SerpApi not configured), so a distance couldn't be computed."

    try:
        results = search_google_maps(place, limit=1)
    except Exception:
        results = []
    if not results:
        return f"Could not find/geocode '{place}' via live place search."

    match = results[0]
    place_lat, place_lon = match.get("centroid_lat"), match.get("centroid_lon")
    if place_lat is None or place_lon is None:
        return f"Could not find/geocode '{place}' via live place search."
    resolved_name = match.get("name") or place
    address = match.get("address")
    resolved_label = f"{resolved_name} ({address})" if address else resolved_name
    # Always echo the ORIGINAL place the user asked about, explicitly tied to
    # whatever the live search actually resolved it to — a real live search
    # ("Mhow" -> "Dr. Ambedkar Nagar", its official renamed listing) can come
    # back with a place name that doesn't textually match the question at
    # all; without this explicit link the answering model has no way to
    # confidently connect the two and was observed refusing to answer even
    # with a correct, fully-grounded distance already in hand.
    label = resolved_label if resolved_name.lower() == place.strip().lower() else f"{place} (found as: {resolved_label})"

    straight_km = haversine_km(site_lat, site_lon, place_lat, place_lon)
    travel = fetch_road_travel_time(site_lat, site_lon, place_lat, place_lon)
    if travel:
        return (
            f"Route from the site to {label}: approximately {travel['travel_distance_km']:.1f} km by road, "
            f"~{travel['travel_time_min']:.0f} min driving (via OSRM, a free public routing demo — treat as "
            f"an estimate, not authoritative). Straight-line distance: {straight_km:.1f} km."
        )
    return (
        f"Could not compute a road route to {label} (routing service unavailable or no route found). "
        f"Straight-line distance only: {straight_km:.1f} km."
    )


_TOOL_FUNCS = {
    "onefivenine_tehsil_info": lambda site_summary, args: _fetch_onefivenine_tehsil_info(site_summary),
    "nearby_category_search": lambda site_summary, args: _nearby_category_search_text(
        site_summary, (args or {}).get("query", ""), (args or {}).get("radius_km"), (args or {}).get("sort_by")
    ),
    "wikipedia_summary": lambda site_summary, args: _wikipedia_summary_text((args or {}).get("query", "")),
    "road_distance_to_place": lambda site_summary, args: _road_distance_text(
        site_summary, (args or {}).get("place", "")
    ),
}

# Human-readable labels for `route_and_answer`'s `tool_used`, for the chat UI
# to show which live tool (if any) grounded an answer — kept next to
# `_TOOL_FUNCS` so a renamed/added/removed tool can't silently drift out of
# sync with what's shown to the user.
TOOL_DISPLAY_NAMES = {
    "onefivenine_tehsil_info": "onefivenine.com tehsil lookup",
    "nearby_category_search": "live nearby-places search (Google Maps)",
    "wikipedia_summary": "Wikipedia summary",
    "road_distance_to_place": "road-distance lookup (Google Maps + OSRM)",
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
    history: list[dict[str, str]] | None = None,
) -> tuple[bool, str | None, str | None, list[str], str | None]:
    """The shared routing step: decide whether ``question`` can be answered
    at all, whether one live tool should be fetched first (grounded against
    ``site_summary`` only — tools never run per comparison site, see
    ``_TOOLS_DESCRIPTION``), and whether it names any other saved site by
    nickname/village (e.g. "which of X and Y has better security").

    Args:
        history: Prior chat turns, same shape as
            :func:`aI_agents.qa.answer_question`'s — folded into the system
            prompt as plain text (see :func:`_format_recent_history`), NOT
            passed as actual chat messages, so a context-dependent
            follow-up (e.g. "by road?" right after a distance question) can
            be resolved instead of misread as an incomplete question in
            isolation. Deliberately not raw messages: this call is
            JSON-mode-only (no ``tools=`` registered anywhere in this app),
            and a prose assistant turn mixed into that message array was
            confirmed to make the model occasionally emit a native-style
            tool call instead of the requested JSON object, which Groq
            rejects outright — a failure a retry can't fix since it's
            deterministic at ``temperature=0.0``, not transient noise.

    Returns:
        ``(can_answer, tool_name, tool_result_text, mentioned_dir_names,
        refusal_reason)``.

    Raises:
        ChatUnavailable: If no Groq API key is configured.
    """

    client = _require_client()

    context_keys = sorted(_curate_site_context(site_summary).keys())
    other_run_choices = _other_run_choices(exclude_summary_path)
    recent_history_text = _format_recent_history(history)
    system_prompt = _routing_system_prompt(
        context_keys, has_comparison_sites, other_run_choices, recent_history_text
    )

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
        site_summary,
        question,
        has_comparison_sites=bool(additional_sites),
        exclude_summary_path=summary_path,
        history=history,
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
