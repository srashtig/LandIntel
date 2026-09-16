"""LandIntel — purpose-driven Q&A, report & compare.

Pick a location in Madhya Pradesh (default site, a previous run, or a fresh
pin-drop/search/Google-Maps-link), state a buying purpose, and get a real
geospatial+socio-economic analysis with a grounded AI Q&A, a generated
report, and multi-site comparison — no scoring engine, just an LLM reasoning
about pros/cons/trade-offs grounded in the site's real data (see
:mod:`aI_agents.qa`).

Run with:

    conda run -n land_intel streamlit run frontend/streamlit_app.py

(from the repo root).
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

# A bare `python`/`streamlit run` invocation puts this script's own
# directory on sys.path automatically, but that isn't guaranteed across
# every way this file can be loaded (e.g. Streamlit's AppTest harness does
# not) — so both directories are added explicitly: this one (for the
# sibling `location_picker` import below) and its parent, the repo root (for
# the sibling data_analysis_pipeline/ and aI_agents/ packages).
_FRONTEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DIR = _FRONTEND_DIR.parent
for _dir in (_FRONTEND_DIR, _REPO_ROOT_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

import streamlit as st

from data_analysis_pipeline import config, data_bundle
from data_analysis_pipeline import runs as runs_module
from data_analysis_pipeline.custom_facts import merge_custom_facts
from data_analysis_pipeline.main import _json_safe
from data_analysis_pipeline.mp_boundary import is_in_mp
from data_analysis_pipeline.runs import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_RADIUS_KM,
    failed_sections,
    list_previous_runs,
    load_default_site,
    resolve_run_output_dir,
    unique_dir,
)

from aI_agents import groq_client
from aI_agents.qa import ChatUnavailable, generate_strengths_concerns
from aI_agents.build_report import generate_report, report_to_pdf_bytes
from aI_agents.compare import MAX_RANK_SITES, compare
from aI_agents.build_compare_report import generate_compare_html
from aI_agents.qa_agent import TOOL_DISPLAY_NAMES, route_and_answer
from aI_agents.run_worker import run_analysis_worker

from location_picker import MP_CENTER, render_location_picker_with_satellite

PURPOSE_OPTIONS = [
    "farming",
    "farmhouse",
    "housing",
    "resort",
    "hotel",
    "car_showroom",
    "warehouse",
    "industry",
]

# Fixed count of named stages orchestrator.run_analysis reports progress
# for. 13 data stages come from only 10 get_*.py files, since
# get_osm_features.py contributes 3 (roads/railways/water, not 1) and
# get_land_surface_temperature.py contributes 2 (LST + air temperature) —
# plus AOI geometry, Site summary, Report assets, and Map = 17 total. See
# data_analysis_pipeline/orchestrator.py's start()/finish() calls for the
# exact list. Used to turn the raw progress log into a real (not
# fake/animated) progress bar.
_TOTAL_PIPELINE_STAGES = 17


def _fmt_hm(minutes: float) -> str:
    """Format a duration in minutes as "H:MM hr", e.g. 170 -> "2:50 hr"."""

    total_minutes = round(minutes)
    return f"{total_minutes // 60}:{total_minutes % 60:02d} hr"


def _init_session_state() -> None:
    defaults = {
        "picked_lat": None,
        "picked_lon": None,
        "ref_lat": None,
        "ref_lon": None,
        "site_summary": None,
        "map_html": None,
        "summary_path": None,
        "source_label": None,
        "run_in_progress": False,
        "run_thread": None,
        "run_out_dir": None,
        "run_cancel_event": None,
        "run_progress_queue": None,
        "run_result": None,
        "run_progress_log": [],
        "chat_history": [],  # [{"role": "user"|"assistant", "content": str}, ...] — one running session
        "_compare_selection": [],  # [(label, site_summary), ...] currently picked in Compare, reused by chat
        "view_mode": None,  # None | "edit_purpose" | "view_analysis" | "generate_report" | "compare"
        "_active_summary_path": None,  # tracks which site's results are showing, see _sync_active_site
        "run_dialog_open": False,  # whether the "Running analysis" popup should be shown
        "pending_site_details": None,  # step-2 form values, snapshotted at click time — see below
        "_force_source_choice": None,  # set by "Pick a different site" — see main()'s radio setup
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _sync_active_site(summary_path: Path) -> None:
    """Reset any open dialog / cached dialog results whenever the site
    being shown changes. Needed because ``st.dialog``'s built-in close (X)
    doesn't tell our code anything — without this, closing a dialog and
    then loading a different site (or even just interacting with anything
    else on the page) would reopen the same dialog for the new site, since
    ``view_mode`` was never cleared."""

    key = str(summary_path)
    if st.session_state.get("_active_summary_path") != key:
        st.session_state._active_summary_path = key
        st.session_state.view_mode = None
        for cache_key in (
            "_last_report_html", "_last_report_pdf", "_last_pros_cons",
            "_last_compare_html", "_last_compare_pdf",
        ):
            st.session_state.pop(cache_key, None)
        st.session_state._compare_selection = []


# Layer names exactly as data_analysis_pipeline's build_map.py names them
# (see data_analysis_pipeline/build_map.py:346-589). Toggled on by a small JS
# snippet appended to the map HTML, since folium bakes each layer's default
# shown/hidden state into static JS at generation time — there's no
# post-hoc Python option without forking that ~600-line function.
_DEFAULT_ON_MAP_LAYERS = [
    "Satellite",  # base layer (radio) — switching it on also switches "Street" off
    "Roads (overlay)",
    "Place names (overlay)",
    "Water",
    "Groundwater",
    "Nearby places (10 km)",
]


def _map_html_with_default_layers(map_html: str) -> str:
    """Appends two post-load fixup scripts to the map: one that ticks
    :data:`_DEFAULT_ON_MAP_LAYERS` on first load (clicking their real,
    Leaflet-rendered layer-control checkboxes/radios if not already in the
    desired state), and one that disables scroll-wheel zoom (see inline
    comment below) so a normal scroll gesture always scrolls the popup."""

    layers_json = json.dumps(_DEFAULT_ON_MAP_LAYERS)
    script = f"""
<script>
(function() {{
  var desiredOn = {layers_json};
  function applyDefaults(attemptsLeft) {{
    var labels = document.querySelectorAll('.leaflet-control-layers-list label');
    if (!labels.length) {{
      if (attemptsLeft > 0) setTimeout(function() {{ applyDefaults(attemptsLeft - 1); }}, 200);
      return;
    }}
    labels.forEach(function(label) {{
      var text = label.textContent.trim();
      var input = label.querySelector('input');
      if (!input) return;
      var shouldBeOn = desiredOn.some(function(name) {{ return text.indexOf(name) !== -1; }});
      if (shouldBeOn && !input.checked) {{ input.click(); }}
    }});
  }}
  setTimeout(function() {{ applyDefaults(15); }}, 300);
}})();
</script>
"""
    # Disables scroll-wheel zoom on the embedded Leaflet map. Without this,
    # a normal mouse-wheel scroll gesture zooms the map instead of scrolling
    # the popup whenever the cursor happens to be over it — and there's no
    # visual cue telling a user to move their cursor off the map first to
    # scroll past it. Zooming is still available via the +/- buttons and
    # double-click; only the "wheel = zoom" behavior is turned off.
    disable_scroll_zoom_script = """
<script>
(function() {
  function disableScrollZoom(attemptsLeft) {
    var found = false;
    for (var key in window) {
      try {
        var candidate = window[key];
        if (candidate && candidate instanceof L.Map && candidate.scrollWheelZoom) {
          candidate.scrollWheelZoom.disable();
          found = true;
        }
      } catch (e) { /* ignore cross-origin/getter errors while scanning window */ }
    }
    if (!found && attemptsLeft > 0) {
      setTimeout(function() { disableScrollZoom(attemptsLeft - 1); }, 200);
    }
  }
  setTimeout(function() { disableScrollZoom(15); }, 300);
})();
</script>
"""
    return map_html + script + disable_scroll_zoom_script


def _key_status(value: str | None) -> str:
    """Placeholder text for a password field showing whether a key is
    already configured, without ever displaying it in cleartext."""

    if not value:
        return "not set"
    return f"configured (…{value[-4:]})" if len(value) > 4 else "configured"


def _render_api_key_settings() -> None:
    """API keys + data/run directory settings — session-independent: saved
    to `.env` (persists across restarts), and takes effect
    immediately for this running process via `config.set_env_key`/
    `config.reset_data_dir`/`runs_module.reset_runs_dir` (no restart
    needed). Process-wide, not isolated per browser tab — fine for this
    locally-run, single-user app."""

    st.markdown(
        '<p style="font-size:1.3rem;font-weight:800;color:#1e3a2f;margin:0 0 2px 0;">⚙️ Settings</p>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")
    st.markdown(
        '<p style="font-size:0.95rem;color:#5c5c5c;margin:0;">Either put these in .env directly, '
        "or paste your keys here and click Save.</p>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")

    groq_key = st.text_input(
        "GROQ_API_KEY", type="password", key="_input_groq_key",
        placeholder=_key_status(config.GROQ_API_KEY),
    )
    st.caption(
        "Required for chat, AI Insights, report interpretation, and Compare. "
        "[Get a free key](https://console.groq.com/keys)"
    )

    run_dir = st.text_input(
        "Run directory", key="_input_run_dir",
        value=str(config.RUNS_DIR),
    )
    st.caption(
        "Where analysis runs (site_summary.json/map.html/report_assets/) "
        "are saved to and loaded from — change this to point at a "
        "different runs/ folder, e.g. one shared or synced from elsewhere."
    )

    serp_key = st.text_input(
        "SERPAPI_KEY (optional)", type="password", key="_input_serp_key",
        placeholder=_key_status(config.SERPAPI_KEY),
    )
    st.caption(
        "Optional — only needed to analyse a **new** site's nearby places, "
        "or the chat's ad-hoc category search. Bundled demo runs already "
        "have this data. [Get a key](https://serpapi.com/manage-api-key)"
    )

    gee_id = st.text_input(
        "GEE_PROJECT_ID (optional)", key="_input_gee_id",
        value=config.GEE_PROJECT_ID or "",
        placeholder="your-gcp-project-id",
    )
    st.caption(
        "Optional — only needed to analyse a **new** site (not for browsing "
        "bundled demo runs). [Set up a Google Earth Engine project]"
        "(https://code.earthengine.google.com/register) — if this machine "
        "has never authenticated Earth Engine before, the first analysis "
        "run will still open a one-time browser sign-in; entering a project "
        "ID here doesn't skip that step."
    )

    data_dir = st.text_input(
        "Data directory (optional)", key="_input_data_dir",
        value=str(config.DATA_DIR),
    )
    st.caption(
        "Optional — only needed to analyse a **new** site; browsing, "
        "reporting on, and chatting about the bundled demo runs all work "
        "without this."
    )

    if config.REFERENCE_DATA_BUNDLE_URL:
        download_clicked = st.button("📥 Download source data (~500 MB, optional)", key="download_data_bundle_button")
    else:
        download_clicked = False
        st.caption("Source-data download: not yet available.")

    if st.button("Save settings", key="save_api_keys_button"):
        if groq_key.strip():
            config.set_env_key("GROQ_API_KEY", groq_key)
            groq_client.reset_client()
        if serp_key.strip():
            config.set_env_key("SERPAPI_KEY", serp_key)
        if gee_id.strip() != (config.GEE_PROJECT_ID or ""):
            config.set_env_key("GEE_PROJECT_ID", gee_id)
            config.reset_earth_engine()
        if data_dir.strip() and data_dir.strip() != str(config.DATA_DIR):
            config.reset_data_dir(data_dir)
        if run_dir.strip() and run_dir.strip() != str(config.RUNS_DIR):
            runs_module.reset_runs_dir(run_dir)
        st.success("Saved.")
        st.rerun()

    if download_clicked:
        progress_bar = st.progress(0.0, text="Starting download...")

        def _on_progress(current: int, total: int | None) -> None:
            if total:
                fraction = min(current / total, 1.0)
                progress_bar.progress(
                    fraction,
                    text=f"Downloading... {current / 1e6:.0f} / {total / 1e6:.0f} MB "
                    f"({fraction * 100:.0f}%)",
                )
            else:
                progress_bar.progress(0.0, text=f"Downloading... {current / 1e6:.0f} MB")

        try:
            data_bundle.download_and_extract(
                config.REFERENCE_DATA_BUNDLE_URL, config.DATA_DIR, progress_callback=_on_progress
            )
        except Exception as error:
            progress_bar.empty()
            st.error(str(error))
        else:
            progress_bar.progress(1.0, text="Done.")
            st.session_state["_data_download_result_path"] = str(config.DATA_DIR)
            st.session_state.view_mode = "data_download_success"
            st.rerun()

    if st.session_state.view_mode == "data_download_success":
        _data_download_success_dialog(st.session_state.get("_data_download_result_path", ""))

    st.caption(
        "Keys are saved to this app's own .env file (not shared beyond this "
        "running instance)."
    )


@st.dialog("Source data downloaded", width="large")
def _data_download_success_dialog(path: str) -> None:
    st.success("✅ Data downloaded successfully and extracted to:")
    st.code(path, language=None)
    st.caption(
        'Copy this into the "Data directory" field above if it doesn\'t '
        "already match, then click Save settings."
    )
    if st.button("Close", key="close_data_download_dialog"):
        st.session_state.view_mode = None
        st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        _render_api_key_settings()


def _render_chat(site_summary: dict, summary_path: Path) -> None:
    """Chat box — works for a single site or (if the user has an active
    selection in Compare) a multi-site comparison chat. Rendered in the main
    page, below the action buttons.

    History flows freely in the page (not boxed into a fixed-height
    container) — ``st.chat_input`` still docks to the bottom of the whole
    browser viewport regardless (a Streamlit quirk, not something this app
    controls), so a long history just grows the page."""

    st.write("")
    st.write("")
    header_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with header_col:
        st.markdown(
            '<div style="background:#eafaf1;border-left:4px solid #2f855a;border-radius:10px;'
            'padding:10px 16px;margin-bottom:2px;">'
            '<span style="font-size:1.15rem;font-weight:700;color:#1e3a2f;">'
            "💬 Ask anything about the land</span></div>",
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.session_state.chat_history and st.button("New chat", key="new_chat_button"):
            st.session_state.chat_history = []
            st.session_state.view_mode = None
            st.rerun()

    additional_sites = st.session_state.get("_compare_selection") or None
    if additional_sites:
        other_labels = ", ".join(label for label, _ in additional_sites)
        st.caption(f"Context: this site + {other_labels} (from your Compare selection).")
    else:
        st.caption(
            "Context: this site — but you can also just name another saved site in your question "
            '(e.g. "which of this and kishangarh has better security?") and it\'ll be pulled in.'
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("How can I help you?")
    if question:
        # Asking a question means the user has moved past whatever
        # View Analysis/Generate Report/Compare dialog was open — without
        # this, the next rerun (triggered by this very chat submission)
        # would still see the old `view_mode` and reopen that dialog on
        # top of the answer.
        st.session_state.view_mode = None
        st.session_state.chat_history.append({"role": "user", "content": question})
        purpose = site_summary.get("buying_purpose")
        purpose_label = (purpose or {}).get("label") or (purpose or {}).get("key") if purpose else None
        preferences = site_summary.get("buying_preferences")

        try:
            with st.spinner("Thinking..."):
                result = route_and_answer(
                    site_summary,
                    question,
                    purpose=purpose_label,
                    preferences=preferences,
                    history=st.session_state.chat_history[:-1],
                    site_label=summary_path.parent.name,
                    additional_sites=additional_sites,
                    summary_path=str(summary_path),
                )
            answer = result["answer"]
            if result.get("tool_used"):
                tool_label = TOOL_DISPLAY_NAMES.get(result["tool_used"], result["tool_used"])
                answer += f"\n\n_(Used live tool: {tool_label})_"
            if result.get("mentioned_sites"):
                answer += f"\n\n_(Also pulled in: {', '.join(result['mentioned_sites'])})_"
        except ChatUnavailable as error:
            answer = f"Q&A unavailable: {error}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


def _site_header(site_summary: dict, summary_path: Path) -> None:
    """Compact identity line — lat/lon, run name, and the current buying
    purpose — with a button that opens the purpose editor as a popup
    (see :func:`_edit_purpose_dialog`) instead of an always-visible inline
    form. No divider/heading of its own (kept to one tight row) — the goal
    is for the whole post-analysis view to fit on screen without scrolling."""

    site = site_summary.get("site", {})
    lat, lon = site.get("latitude"), site.get("longitude")
    name = summary_path.parent.name
    purpose = site_summary.get("buying_purpose")
    purpose_label = (purpose.get("label") if purpose else None) or "not set"
    coord_text = (
        f"lat <code>{lat:.6f}</code>, lon <code>{lon:.6f}</code>"
        if lat is not None and lon is not None else "location unknown"
    )

    with st.container(border=True):
        col_info, col_change = st.columns([6, 1], vertical_alignment="center")
        with col_info:
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;">'
                f"<span><b>📍 {name}</b> — {coord_text}</span>"
                f"<span><b>Buying purpose:</b> {purpose_label}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        with col_change:
            button_label = "✏️ Change" if purpose else "➕ Set purpose"
            with st.container(key="change_purpose_btn_wrap"):
                if st.button(button_label, key="btn_change_purpose"):
                    st.session_state.view_mode = "edit_purpose"
                    st.rerun()

    if st.session_state.view_mode == "edit_purpose":
        _edit_purpose_dialog(site_summary, summary_path)


@st.dialog("Buying purpose", width="large")
def _edit_purpose_dialog(site_summary: dict, summary_path: Path) -> None:
    """Dropdown + free-text "Other" purpose field, saved directly into this
    run's site_summary.json."""

    st.caption(
        "What is this plot for? Used as context for the Q&A/pros-cons/report/compare "
        "features below — there's no separate scoring engine, an LLM reasons about pros "
        "and cons grounded in this site's real data for the stated purpose."
    )

    existing = site_summary.get("buying_purpose")
    if existing:
        st.markdown(f"**Current purpose:** {existing.get('label', existing)}")

    default_index = 0
    if existing and existing.get("key") in PURPOSE_OPTIONS:
        default_index = PURPOSE_OPTIONS.index(existing["key"])
    purpose_key = st.selectbox(
        "Purpose", options=PURPOSE_OPTIONS + ["other"], index=default_index, key="buying_purpose_select"
    )
    other_text = ""
    if purpose_key == "other":
        other_text = st.text_input(
            "Describe the purpose",
            value=(existing.get("free_text") or "") if existing and existing.get("is_other") else "",
            key="buying_purpose_other",
        )
    preferences = st.text_area(
        "Any specific preferences? (optional — e.g. \"near a well-developed town with schools\", "
        "\"good financial growth potential\")",
        value=site_summary.get("buying_preferences") or "",
        key="buying_preferences_input",
    )

    if st.button("Save purpose", key="save_buying_purpose_button"):
        if purpose_key == "other" and not other_text.strip():
            st.warning("Describe the purpose, or pick one from the list.")
        else:
            updated = dict(site_summary)
            updated["buying_purpose"] = {
                "key": purpose_key,
                "label": other_text.strip() if purpose_key == "other" else purpose_key,
                "is_other": purpose_key == "other",
                "free_text": other_text.strip() if purpose_key == "other" else None,
            }
            updated["buying_preferences"] = preferences.strip() or None
            summary_path.write_text(json.dumps(updated, indent=2, default=_json_safe))
            # "Run a new analysis" renders from st.session_state.site_summary
            # (an in-memory copy), not a fresh disk read like the other two
            # source branches — without this, the rerun below would still
            # show the pre-save (purpose-less) copy.
            if st.session_state.get("summary_path") == str(summary_path):
                st.session_state.site_summary = updated
            st.session_state.view_mode = None
            st.success("Saved.")
            st.rerun()


def _key_points_section(site_summary: dict, summary_path: Path) -> None:
    """Auto-fetched (no button — just appears) AI strengths/concerns/
    trade-offs — the same judgment used in the "Overall Summary &
    Suitability" report chapter (:func:`aI_agents.qa.generate_strengths_concerns`)
    — cached per site so re-rendering the dialog (e.g. after clicking
    Bookmark) doesn't re-trigger a Groq call every time."""

    st.subheader("🤖 AI Insights")
    purpose = site_summary.get("buying_purpose")
    if not purpose:
        st.info("Set a buying purpose above first — key points are grounded in that.")
        return

    cache_key = f"_key_points_{summary_path}"
    if cache_key not in st.session_state:
        purpose_label = purpose.get("label") or purpose.get("key")
        preferences = site_summary.get("buying_preferences")
        try:
            with st.spinner("Reasoning about this site (grounded in its saved data)..."):
                st.session_state[cache_key] = generate_strengths_concerns(site_summary, purpose_label, preferences=preferences)
        except ChatUnavailable as error:
            st.session_state[cache_key] = {"_error": str(error)}

    result = st.session_state[cache_key]
    if result.get("_error"):
        st.warning(f"⚠️ Q&A unavailable: {result['_error']}")
        return

    strengths = result.get("strengths") or []
    concerns = result.get("concerns") or []
    trade_offs = result.get("trade_offs") or []
    if not (strengths or concerns or trade_offs):
        st.info("Nothing returned — try setting a more specific buying purpose, or try again.")
        return

    col_strengths, col_concerns = st.columns(2)
    with col_strengths:
        st.markdown("**✅ Strengths**")
        for item in strengths:
            st.markdown(f"- {item}")
    with col_concerns:
        st.markdown("**⚠️ Concerns**")
        for item in concerns:
            st.markdown(f"- {item}")
    if trade_offs:
        st.markdown("**⚖️ Trade-offs**")
        for item in trade_offs:
            st.markdown(f"- {item}")


def _custom_facts_section(site_summary: dict, summary_path: Path) -> None:
    """Reference location + plot price/area editor. The map itself already
    shows a fuller version of the same site data as an overlay panel
    (``build_map.py``'s ``_build_panel_html``), so a separate "Key facts"
    summary next to it would be redundant."""

    with st.expander("Custom facts (optional): reference location, plot price/area"):
        existing_ref = site_summary.get("reference_location")
        if existing_ref:
            if existing_ref.get("travel_time_min") is not None:
                travel_note = (
                    f", {existing_ref['travel_time_min']:.0f} min ({existing_ref['travel_distance_km']:.1f} km) by road"
                )
            else:
                travel_note = ", road travel time unavailable"
            st.markdown(
                f"**Current reference location:** {existing_ref.get('label') or 'Reference location'} "
                f"— {existing_ref['straight_line_km']:.2f} km straight-line{travel_note}"
            )
            transit = existing_ref.get("transit")
            if transit and (transit.get("bus_duration_min") or transit.get("train_duration_min")):
                legs = []
                if transit.get("bus_duration_min") is not None:
                    legs.append(f"{_fmt_hm(transit['bus_duration_min'])} bus")
                if transit.get("train_duration_min") is not None:
                    legs.append(f"{_fmt_hm(transit['train_duration_min'])} train")
                walk_note = f" + {transit['walk_km']:.1f} km walk" if transit.get("walk_km") else ""
                st.markdown(f"**Public transit (fastest option found):** {' + '.join(legs)}{walk_note}")

        existing_price = site_summary.get("land_price") or {}
        if existing_price.get("source") == "User-supplied single-plot price entry":
            st.markdown(
                f"**Current plot:** ₹{existing_price['total_price']:,.0f} total — "
                f"{existing_price['area_sqft']:,.0f} sqft — ₹{existing_price['price_per_sqft']:,.2f} / sqft"
            )

        site_lat, site_lon = site_summary["site"]["latitude"], site_summary["site"]["longitude"]
        render_location_picker_with_satellite(
            "ref", "Reference location (e.g. your workplace, a family home, a market)",
            default_lat=MP_CENTER[0], default_lon=MP_CENTER[1],
            secondary_marker=(site_lat, site_lon, "Analyzed site"),
        )
        ref_label = st.text_input("Label for this reference location (optional)", key="ref_label_input")
        col_price, col_area, col_unit = st.columns(3)
        with col_price:
            plot_price = st.number_input("Plot price (INR)", min_value=0.0, value=0.0, step=1000.0, key="custom_plot_price")
        with col_area:
            plot_area = st.number_input("Plot area", min_value=0.0, value=0.0, step=0.1, key="custom_plot_area")
        with col_unit:
            area_unit = st.selectbox("Area unit", options=["acre", "hectare", "sqft", "sqm"], key="custom_area_unit")

        if st.button("Save to this run", key="save_custom_facts_button"):
            ref_lat, ref_lon = st.session_state.ref_lat, st.session_state.ref_lon
            reference_location = (
                {"lat": ref_lat, "lon": ref_lon, "label": ref_label or None}
                if ref_lat is not None and ref_lon is not None else None
            )
            has_price = plot_price > 0 and plot_area > 0
            if reference_location is None and not has_price:
                st.warning("Pick a reference location and/or enter a price and area first.")
            else:
                with st.spinner("Computing distance/travel time/transit..."):
                    updated_summary, _ref_map_html = merge_custom_facts(
                        site_summary,
                        reference_location=reference_location,
                        plot_price=plot_price if has_price else None,
                        plot_area=plot_area if has_price else None,
                        area_unit=area_unit if has_price else None,
                    )
                summary_path.write_text(json.dumps(updated_summary, indent=2, default=_json_safe))
                if st.session_state.get("summary_path") == str(summary_path):
                    st.session_state.site_summary = updated_summary
                st.success("Saved to this run's site_summary.json.")
                st.rerun()


_GLOBAL_CSS = """
<style>
div.block-container {
    padding-top: 2rem;
}
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    font-size: 1.05rem;
    transition: transform 0.05s ease, box-shadow 0.15s ease;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}
.stButton > button:hover {
    box-shadow: 0 3px 10px rgba(0,0,0,0.12);
    transform: translateY(-1px);
}
button[kind="primary"] {
    box-shadow: 0 2px 6px rgba(47,133,90,0.35);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
}
/* Slightly larger text on every interactive control (buttons, inputs,
   selectboxes, radios, checkboxes, chat input) — Streamlit's default is a
   touch small; this nudges it up without changing layout/spacing. */
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p,
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] * ,
[data-testid="stChatInput"] textarea {
    font-size: 1.05rem !important;
}
.st-key-change_purpose_btn_wrap .stButton > button {
    font-size: 12px;
    padding: 2px 10px;
    min-height: 0;
}
/* The sidebar is much narrower than the main page — the same font bump
   that reads fine in the main content wraps awkwardly here (a label like
   "GEE_PROJECT_ID (optional)" spilling onto two lines, oversized input
   text), so it's scaled back down and the vertical rhythm tightened up. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600;
}
[data-testid="stSidebar"] .stTextInput input {
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    font-size: 0.85rem !important;
    padding: 0.35rem 0.75rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    line-height: 1.35;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem;
}
</style>
"""

_DIALOG_SCROLLBAR_CSS = """
<style>
div[data-testid="stDialog"] div[role="dialog"],
div[data-testid="stDialogContent"] {
    max-height: 85vh;
    overflow-y: scroll !important;
    scrollbar-width: auto;
    scrollbar-color: #9ca3af #f0efec;
    padding-top: 0.5rem !important;
}
div[data-testid="stDialog"] div[role="dialog"]::-webkit-scrollbar,
div[data-testid="stDialogContent"]::-webkit-scrollbar {
    width: 12px;
}
div[data-testid="stDialog"] div[role="dialog"]::-webkit-scrollbar-thumb,
div[data-testid="stDialogContent"]::-webkit-scrollbar-thumb {
    background-color: #9ca3af;
    border-radius: 6px;
    border: 3px solid #f0efec;
}
div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
</style>
"""


@st.dialog("View Analysis", width="large")
def _view_analysis_dialog(site_summary: dict, map_html: str, summary_path: Path) -> None:
    st.markdown(_DIALOG_SCROLLBAR_CSS, unsafe_allow_html=True)

    if site_summary.get("run_success") is False:
        failed = failed_sections(site_summary)
        st.warning(
            "This run completed with partial failures — some sections are unavailable "
            f"({', '.join(failed) if failed else 'see run.log'}).",
            icon="⚠️",
        )

    # One compact row right under the title — combines the layers hint and
    # the "jump past the map" link (below) instead of stacking them as
    # separate blocks, to close up the gap before the map.
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'gap:12px;margin:0 0 6px;">'
        '<span style="color:#898781;font-size:13px;">💡 Use the layers control (top-right of the '
        "map) to switch between Street/Satellite view and toggle other analysis layers.</span>"
        '<a href="#ai-insights-anchor" style="flex-shrink:0;display:inline-block;background:#2563eb;'
        "color:#fff;padding:8px 16px;border-radius:8px;font-size:15px;font-weight:700;"
        'text-decoration:none;">⬇️ Jump to AI Insights</a>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.components.v1.html(_map_html_with_default_layers(map_html), height=800, scrolling=True)
    _custom_facts_section(site_summary, summary_path)
    st.divider()
    st.markdown('<div id="ai-insights-anchor"></div>', unsafe_allow_html=True)
    _key_points_section(site_summary, summary_path)


@st.dialog("Generate Report", width="large")
def _generate_report_dialog(site_summary: dict, summary_path: Path) -> None:
    st.markdown(_DIALOG_SCROLLBAR_CSS, unsafe_allow_html=True)
    if st.button("Build report", key="build_report_button"):
        with st.spinner(
            "Assembling report from pre-generated analysis maps/charts and writing the AI "
            "interpretation sections..."
        ):
            try:
                html, updated_summary = generate_report(site_summary, summary_path)
            except ChatUnavailable as error:
                st.error(f"Report generation hit a Q&A error: {error}")
            else:
                st.session_state["_last_report_html"] = html
                st.session_state["_last_report_pdf"] = report_to_pdf_bytes(html)
                _ = updated_summary  # already persisted to summary_path by generate_report

    if st.session_state.get("_last_report_html"):
        pdf_bytes = st.session_state.get("_last_report_pdf")
        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF", data=pdf_bytes,
                file_name=f"{summary_path.parent.name}_report.pdf", mime="application/pdf",
                key="download_report_pdf",
            )
        else:
            st.caption("PDF export unavailable for this report — showing the on-screen version below.")
        st.components.v1.html(st.session_state["_last_report_html"], height=1400, scrolling=True)


@st.dialog("Compare with other sites", width="large")
def _compare_dialog(site_summary: dict, summary_path: Path) -> None:
    st.markdown(_DIALOG_SCROLLBAR_CSS, unsafe_allow_html=True)
    st.caption(
        f"Pick up to {MAX_RANK_SITES - 1} other runs to rank alongside this one. Each site gets an "
        "LLM-judged score/verdict — reasoned out loud from its real data (and a live tool if useful), "
        "not a scoring formula."
    )

    other_runs = [r for r in list_previous_runs() if r["summary_path"] != summary_path]
    if not other_runs:
        st.info("No other runs to compare against yet.")
        return

    labels = [r["label"] for r in other_runs]
    chosen_indices = st.multiselect(
        "Compare against",
        options=range(len(other_runs)),
        format_func=lambda i: labels[i],
        key="compare_run_multiselect",
        max_selections=MAX_RANK_SITES - 1,
    )

    this_label = summary_path.parent.name
    others = [(other_runs[i]["dir_name"], json.loads(other_runs[i]["summary_path"].read_text())) for i in chosen_indices]

    # Kept in session state so the chat box can also ground itself in
    # whatever comparison set is currently selected here.
    st.session_state["_compare_selection"] = others

    if len(others) < 1:
        st.info("Select at least one other run to compare against.")
        return

    purpose = site_summary.get("buying_purpose")
    if not purpose:
        st.info("Set a buying purpose above first — the comparison is grounded in that.")
        return
    purpose_label = purpose.get("label") or purpose.get("key")
    preferences = site_summary.get("buying_preferences")

    all_sites = [(this_label, site_summary), *others]

    if st.button("Rank sites", key="run_compare_button"):
        try:
            with st.spinner("Reasoning through each site's data (and a live tool, if useful)..."):
                ranking_result, facts = compare(all_sites, purpose_label, preferences=preferences)
        except (ChatUnavailable, ValueError) as error:
            st.error(f"Compare unavailable: {error}")
        else:
            html = generate_compare_html(all_sites, ranking_result, facts, purpose_label)
            st.session_state["_last_compare_html"] = html
            st.session_state["_last_compare_pdf"] = report_to_pdf_bytes(html)

    if st.session_state.get("_last_compare_html"):
        pdf_bytes = st.session_state.get("_last_compare_pdf")
        if pdf_bytes:
            st.download_button(
                "⬇️ Download PDF", data=pdf_bytes,
                file_name=f"{this_label}_comparison.pdf", mime="application/pdf",
                key="download_compare_pdf",
            )
        else:
            st.caption("PDF export unavailable for this comparison — showing the on-screen version below.")
        st.components.v1.html(st.session_state["_last_compare_html"], height=1400, scrolling=True)


@st.dialog("Running analysis", width="large")
def _run_progress_dialog() -> None:
    """Shown while a fresh 'Run a new analysis' is in flight. Polls (drain
    the queue, re-render, sleep, rerun — while the dialog function keeps
    getting called each rerun, it stays open). On success it shows "Ready"
    and auto-closes onto the normal results view; on Stop it just closes
    immediately (the cancellation itself is fire-and-forget — the
    background thread notices and winds down on its own)."""

    while not st.session_state.run_progress_queue.empty():
        st.session_state.run_progress_log.append(st.session_state.run_progress_queue.get())

    thread = st.session_state.run_thread

    if thread.is_alive():
        completed_stages = sum(
            1 for message in st.session_state.run_progress_log if message[:1] in ("✓", "✗")
        )
        st.info("⏳ Analysis is in progress — this usually takes just a couple of minutes.")
        st.progress(min(completed_stages / _TOTAL_PIPELINE_STAGES, 0.98))

        with st.expander("Technical details", expanded=False):
            for message in st.session_state.run_progress_log:
                st.write(message)

        if st.button("Stop analysis", key="stop_analysis_button"):
            st.session_state.run_cancel_event.set()
            st.session_state.run_in_progress = False
            st.session_state.run_dialog_open = False
            st.rerun()

        time.sleep(1.0)
        st.rerun()
        return

    # Thread finished — figure out how.
    result = st.session_state.run_result
    out_dir = st.session_state.run_out_dir
    st.session_state.run_in_progress = False

    if result.get("status") == "done":
        site_summary = result["site_summary"]
        map_html = result["map_html"]

        # Bake in the purpose/custom features collected in step 2 (before
        # the run) — snapshotted into pending_site_details at click time
        # (see the run-button handler), since the form's own widget-keyed
        # session state (e.g. new_run_purpose_select) no longer exists by
        # the time we get here — Streamlit drops it once the form stops
        # being rendered.
        pending = st.session_state.pending_site_details
        site_summary["buying_purpose"] = pending["buying_purpose"]
        site_summary["buying_preferences"] = pending["buying_preferences"]

        ref_lat, ref_lon = st.session_state.get("new_run_ref_lat"), st.session_state.get("new_run_ref_lon")
        has_price = pending["plot_price"] > 0 and pending["plot_area"] > 0
        if ref_lat is not None or has_price:
            site_summary, _ref_map_html = merge_custom_facts(
                site_summary,
                reference_location=(
                    {"lat": ref_lat, "lon": ref_lon, "label": pending["ref_label"]}
                    if ref_lat is not None else None
                ),
                plot_price=pending["plot_price"] if has_price else None,
                plot_area=pending["plot_area"] if has_price else None,
                area_unit=pending["area_unit"] if has_price else None,
            )

        summary_path = out_dir / "site_summary.json"
        map_path = out_dir / "map.html"
        summary_path.write_text(json.dumps(site_summary, indent=2, default=_json_safe))
        map_path.write_text(map_html)

        st.session_state.site_summary = site_summary
        st.session_state.map_html = map_html
        st.session_state.summary_path = str(summary_path)

        st.success("✅ Ready!")
        st.session_state.run_dialog_open = False
        time.sleep(1.0)
        st.rerun()
    elif result.get("status") == "cancelled":
        st.session_state.run_dialog_open = False
        st.rerun()
    else:
        st.error(
            f"Analysis failed: {result.get('error', 'unknown error')}. "
            f"Full traceback is in {out_dir / 'run.log'}."
        )
        if st.button("Close", key="close_run_error_dialog"):
            st.session_state.run_dialog_open = False
            st.rerun()


def _action_buttons(site_summary: dict, map_html: str, summary_path: Path) -> None:
    st.write("")
    col_view, col_report = st.columns(2)
    with col_view:
        if st.button("🔍 View Analysis Results", key="btn_view_analysis", use_container_width=True, type="primary"):
            st.session_state.view_mode = "view_analysis"
            st.rerun()
    with col_report:
        if st.button("📄 Generate Report", key="btn_generate_report", use_container_width=True, type="primary"):
            st.session_state.view_mode = "generate_report"
            st.rerun()

    if st.session_state.view_mode == "view_analysis":
        _view_analysis_dialog(site_summary, map_html, summary_path)
    elif st.session_state.view_mode == "generate_report":
        _generate_report_dialog(site_summary, summary_path)


def _secondary_actions(site_summary: dict, summary_path: Path) -> None:
    """Compare with other sites, or go back to setting up a fresh analysis
    — same two options regardless of which of the three sources (default
    site / previous run / new run) is currently shown."""

    col_pick, col_compare = st.columns(2)
    with col_pick:
        if st.button("📍 Pick a different site", key="pick_different_site_button", use_container_width=True):
            st.session_state["_force_source_choice"] = "Run a new analysis"
            st.session_state.view_mode = None
            for key in ("site_summary", "map_html", "summary_path", "picked_lat", "picked_lon"):
                st.session_state[key] = None
            st.rerun()
    with col_compare:
        if st.button("⚖️ Compare with other site(s)", key="btn_compare", use_container_width=True):
            st.session_state.view_mode = "compare"
            st.rerun()

    if st.session_state.view_mode == "compare":
        _compare_dialog(site_summary, summary_path)


def _render_results(site_summary: dict, map_html: str, summary_path: Path) -> None:
    """The shared post-analysis view for all three sources (default site,
    a previous run, or a just-completed new run) — the site identity line
    (lat/lon, name, buying purpose), the action buttons, the compare/
    pick-a-different-site buttons, and the chat box, identical every time."""

    _sync_active_site(summary_path)
    _site_header(site_summary, summary_path)
    _action_buttons(site_summary, map_html, summary_path)
    _secondary_actions(site_summary, summary_path)
    _render_chat(site_summary, summary_path)


def main() -> None:
    st.set_page_config(page_title="LandIntel", layout="wide")
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    _init_session_state()
    _render_sidebar()

    st.markdown("## 🤖 LandIntel")
    st.markdown(
        '<p style="font-size:1.05rem;color:#5c5c5c;margin-top:-8px;">'
        "AI powered tool to analyse a land, grounded in data. "
        "<i>Know it before you buy it!</i></p>",
        unsafe_allow_html=True,
    )

    # Streamlit forbids setting a widget-bound session_state key (here,
    # "source_choice") after that widget has already been instantiated in
    # the current run — so "Pick a different site" (in _secondary_actions,
    # called much later in this same run) can't set it directly. It instead
    # sets this separate, non-widget flag and reruns; applying it here,
    # right before the radio widget below is created, is what actually
    # switches the selected source.
    if st.session_state.get("_force_source_choice"):
        st.session_state["source_choice"] = st.session_state.pop("_force_source_choice")

    source = st.radio(
        "Pick a site to analyse:",
        options=["Use existing default site", "Browse a previous run", "Run a new analysis"],
        index=0,
        key="source_choice",
        horizontal=True,
    )

    # A dialog (View Analysis/Generate Report/Compare) stays open across
    # reruns only because `view_mode` still names it — but that means an
    # unrelated rerun (switching this source tab, asking a chat question)
    # re-opens whatever dialog was last open, since `view_mode` doesn't
    # otherwise know the user has moved on. Switching source tabs is an
    # unambiguous "moved on" signal, so it always clears `view_mode` here.
    if st.session_state.get("_prev_source_choice") != source:
        st.session_state.view_mode = None
        st.session_state["_prev_source_choice"] = source

    if source == "Use existing default site":
        try:
            site_summary, map_html = load_default_site()
        except FileNotFoundError as error:
            st.error(str(error))
            return
        st.success(
            f"Loaded default site: {runs_module.DEFAULT_SUMMARY_PATH.name} "
            f"(lat {DEFAULT_LAT}, lon {DEFAULT_LON}, radius {DEFAULT_RADIUS_KM} km)."
        )
        _render_results(site_summary, map_html, runs_module.DEFAULT_SUMMARY_PATH)
        return

    if source == "Browse a previous run":
        runs = list_previous_runs()
        if not runs:
            st.info("No previous runs found yet under `runs/`.")
            return
        labels = [r["label"] for r in runs]
        selected = st.radio(
            f"Previous runs ({len(runs)} found, newest first)",
            options=range(len(runs)),
            format_func=lambda i: labels[i],
            key="previous_run_selected",
        )
        chosen = runs[selected]
        site_summary = json.loads(chosen["summary_path"].read_text())
        map_html = chosen["map_path"].read_text()
        st.success(f"Loaded run: {chosen['dir_name']}")
        _render_results(site_summary, map_html, chosen["summary_path"])
        return

    # ---------------- Run a new analysis ----------------
    # The setup form (steps 1-4) is only shown before a run starts — once
    # it's in progress or has completed, this collapses to just the
    # loading indicator (below) and then the same buttons+chat view every
    # other source uses.
    if not st.session_state.run_in_progress and st.session_state.site_summary is None:
        st.subheader("1. Pick a site")
        render_location_picker_with_satellite("picked", "")

        lat = st.session_state.picked_lat
        lon = st.session_state.picked_lon

        st.divider()
        st.subheader("2. Add details")
        run_label = st.text_input(
            "Site label",
            key="run_label_input",
            placeholder="e.g. farmhouse candidate 1 — defaults to lat, lon if left blank",
            help="Used as this run's folder name under runs/ — defaults to "
            "<timestamp>_<lat>_<lon>_<radius>km if left blank.",
        )
        new_run_purpose_key = st.selectbox(
            "Purpose", options=PURPOSE_OPTIONS + ["other"], index=0, key="new_run_purpose_select"
        )
        new_run_purpose_other = ""
        if new_run_purpose_key == "other":
            new_run_purpose_other = st.text_input("Describe the purpose", key="new_run_purpose_other")
        new_run_preferences = st.text_area(
            "Specific preferences (optional)",
            placeholder='e.g. "near a well-developed town with schools", "good financial growth potential"',
            key="new_run_preferences_input",
        )
        radius_km = st.number_input(
            "Analysis radius (km) — 5 km recommended",
            min_value=0.5,
            max_value=25.0,
            value=DEFAULT_RADIUS_KM,
            step=0.5,
            key="radius_input",
        )
        col_area, col_unit, col_price = st.columns(3)
        with col_area:
            new_run_plot_area = st.number_input("Land area (optional)", min_value=0.0, value=0.0, step=0.1, key="new_run_plot_area")
        with col_unit:
            new_run_area_unit = st.selectbox("Area unit", options=["acre", "hectare", "sqft", "sqm"], key="new_run_area_unit")
        with col_price:
            new_run_plot_price = st.number_input("Land price (optional)", min_value=0.0, value=0.0, step=1000.0, key="new_run_plot_price")

        with st.expander("Add a Reference Location to find Distance to the site (optional)"):
            render_location_picker_with_satellite(
                "new_run_ref", "Reference location",
                default_lat=MP_CENTER[0], default_lon=MP_CENTER[1],
                secondary_marker=(lat, lon, "Selected site") if lat is not None and lon is not None else None,
            )
            new_run_ref_label = st.text_input("Label for this reference location (optional)", key="new_run_ref_label_input")

        st.divider()

        if lat is None or lon is None:
            st.info("Pick a location above (pin-drop, lat/lon, or search) to continue.")
            run_enabled = False
        else:
            in_mp = is_in_mp(lat, lon)
            st.markdown(f"**Selected location:** lat `{lat:.6f}`, lon `{lon:.6f}`, radius `{radius_km:g}` km")
            if not in_mp:
                st.error(
                    "This location is outside Madhya Pradesh. The underlying datasets are "
                    "MP-only, so analysis is disabled for points outside the state."
                )
            run_enabled = bool(in_mp)

        run_clicked = st.button(
            "Run full analysis", disabled=(not run_enabled) or st.session_state.run_in_progress, type="primary"
        )

        if run_clicked and run_enabled and not st.session_state.run_in_progress:
            out_dir = unique_dir(resolve_run_output_dir(lat, lon, radius_km, label=run_label or None))
            out_dir.mkdir(parents=True, exist_ok=True)
            # Snapshot the step-2 form values into stable (non-widget)
            # session state now, while the form is still rendered — once
            # run_in_progress flips, the form (and its widget-keyed session
            # state, e.g. new_run_purpose_select) disappears, and reading
            # those directly from _run_progress_dialog later raises
            # AttributeError (Streamlit drops widget state for widgets that
            # stop being drawn).
            st.session_state.pending_site_details = {
                "buying_purpose": {
                    "key": new_run_purpose_key,
                    "label": new_run_purpose_other.strip() if new_run_purpose_key == "other" else new_run_purpose_key,
                    "is_other": new_run_purpose_key == "other",
                    "free_text": new_run_purpose_other.strip() if new_run_purpose_key == "other" else None,
                },
                "buying_preferences": new_run_preferences.strip() or None,
                "ref_label": new_run_ref_label or None,
                "plot_price": new_run_plot_price,
                "plot_area": new_run_plot_area,
                "area_unit": new_run_area_unit,
            }
            st.session_state.run_out_dir = out_dir
            st.session_state.run_cancel_event = threading.Event()
            st.session_state.run_progress_queue = queue.Queue()
            st.session_state.run_progress_log = []
            st.session_state.run_result = {}
            st.session_state.run_in_progress = True
            st.session_state.run_dialog_open = True
            thread = threading.Thread(
                target=run_analysis_worker,
                args=(
                    lat,
                    lon,
                    radius_km,
                    out_dir / "run.log",
                    st.session_state.run_progress_queue,
                    st.session_state.run_cancel_event,
                    st.session_state.run_result,
                ),
                daemon=True,
            )
            st.session_state.run_thread = thread
            thread.start()
            st.rerun()

    if st.session_state.run_dialog_open:
        _run_progress_dialog()

    if st.session_state.site_summary is not None:
        st.divider()
        st.subheader("Results")
        _render_results(
            st.session_state.site_summary,
            st.session_state.map_html,
            Path(st.session_state.summary_path),
        )


if __name__ == "__main__":
    main()
