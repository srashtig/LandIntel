# Architecture

## Three top-level packages, one app

```
<repo root>/
├── data_analysis_pipeline/   # the analysis pipeline: orchestrator + one
│                              #   get_*.py per data source, plus two small
│                              #   precomputed assets (mp_boundary.geojson,
│                              #   mp_location_index.parquet)
├── aI_agents/                 # the purpose-driven AI layer: Q&A, report,
│                              #   compare, static report-asset generation
├── frontend/                  # the one Streamlit app
├── runs/                      # shared run store: site_summary.json,
│                              #   map.html, report_assets/ per run
├── cache/                     # ephemeral OSM/Overpass HTTP cache (gitignored)
├── scripts/                   # one-time offline precompute scripts
└── docs/                      # this file + DATA_SOURCES.md + REPORT_METHODOLOGY.md
```

All three top-level packages import each other as plain packages (no
sys.path tricks, no cross-package aliasing) — `frontend` and `aI_agents`
both depend on `data_analysis_pipeline`, and `frontend` depends on
`aI_agents`, but never the other way (this is why `data_analysis_pipeline`
has its own small `runs.py` for run-directory listing, rather than
`aI_agents`'s chat-tool routing reaching into `frontend` for it — that
would have been a cycle). Every pipeline fix/data source lives in exactly
one place (`data_analysis_pipeline/`), read by both `aI_agents` and
`frontend`. See [DESIGN_CHOICES_AND_LIMITATIONS.md](DESIGN_CHOICES_AND_LIMITATIONS.md)
for why the app is laid out this way.

## The actual user workflow (`frontend/streamlit_app.py::main()`)

```
Pick a site               Add details            Run / load           Act on results
─────────────             ────────────            ──────────           ───────────────
default site       ┐
previous run        ├──►  (skip — already   ──►  load from disk  ──┐
new pin/search/URL ┘      has purpose/price)      or run pipeline   │
                                                                     ▼
                    buying purpose + prefs                 View Analysis / Generate
                    (+ reference location,                 Report / Compare / Pick a
                     price/area, optional)                 different site — then chat
```

1. **Pick a site** — `st.radio` between "Use existing default site",
   "Browse a previous run" (`data_analysis_pipeline.runs.list_previous_runs`),
   or "Run a new analysis" (`frontend.location_picker.render_location_picker_with_satellite`
   — search box / pasted Google Maps link / pin-drop, MP-boundary-checked via
   `data_analysis_pipeline.mp_boundary.is_in_mp`).
2. **For a new analysis only** — a short form collects buying purpose +
   preferences, an optional reference location, and optional plot
   price/area, *before* the run starts; clicking "Run full analysis"
   launches `aI_agents.run_worker.run_analysis_worker` on a background
   thread and shows `frontend.streamlit_app._run_progress_dialog` (a polling
   `st.dialog`) until it finishes, then merges those step-2 answers into the
   freshly-produced `site_summary.json` via `data_analysis_pipeline.custom_facts.merge_custom_facts`.
3. **Shared results view** (`frontend.streamlit_app._render_results`, used
   by all three sources) — a compact site-identity header
   (`_site_header`, with a "Change purpose" popup,
   `_edit_purpose_dialog`), then action buttons:
   - **View Analysis** (`_view_analysis_dialog`) — the interactive Folium
     map (`data_analysis_pipeline.build_map.build_map`, pre-built during the
     run) plus the Custom Facts editor (`_custom_facts_section`) and
     auto-fetched AI Insights (`_key_points_section` →
     `aI_agents.qa.generate_strengths_concerns`).
   - **Generate Report** (`_generate_report_dialog`) — builds and shows the
     PDF/HTML report (`aI_agents.build_report.generate_report` +
     `report_to_pdf_bytes`) from the run's pre-generated `report_assets/`.
   - **Compare** (`_compare_dialog`) — pick up to `MAX_RANK_SITES - 1` other
     runs, then `aI_agents.compare.compare` produces a ranked shortlist +
     side-by-side fact table rendered via `aI_agents.build_compare_report.generate_compare_html`.
   - **Pick a different site** — resets back to step 1.
   - **Chat** (`_render_chat`) — a free-flowing chat box grounded in the
     current site (and, if active, the Compare selection) via
     `aI_agents.qa_agent.route_and_answer`.

## The pipeline: `data_analysis_pipeline/orchestrator.py`

`run_analysis(lat, lon, radius_km, ...)` builds an AOI (area of interest)
polygon, then runs one stage per data source, in this order:

1. OSM roads (blocking)
2. OSM railways — **started in the background**, joined later
3. Sentinel-2 NDVI (GEE)
4. *(railways joined here — its network wait has overlapped with step 3)*
5. ESA WorldCover land cover (GEE)
6. Landsat land-surface temperature (GEE)
7. ERA5-Land air temperature (GEE)
8. SRTM terrain / elevation / slope (GEE)
9. OSM water bodies/waterways — **started in the background**, joined later
10. CGWB groundwater levels (local CSV)
11. *(OSM water joined here — its wait has overlapped with step 10)*
12. River floodplain + SAC water bodies (local GeoJSON)
13. CHIRPS + IMD rainfall (GEE + local NetCDF)
14. Nearby places (OSM settlements + SerpApi)
15. Administrative context — village/tehsil/district, census, guideline rates
16. Assemble `site_summary.json`
17. Report assets (charts/maps for the report — see below)
18. Build the interactive Folium map

**Fault isolation**: every stage runs through `_call_with_hard_timeout`/
`_safe_call`, which catches any exception or timeout and records a
`FAILURE_CONFIDENCE` envelope for that dataset instead of aborting the run.
A run can complete successfully with some sections marked "unavailable" —
this is a normal outcome, not an error state, and every downstream consumer
(the UI panel, the Q&A grounding, the report) is written to handle a missing
section by disclosing it rather than fabricating a value.

**Background-thread overlap**: OSM railway and water-body queries go through
the public Overpass API, which is the pipeline's least reliable and often
slowest dependency. Rather than block on them in sequence, the orchestrator
`start()`s each one on a background thread immediately, keeps running
unrelated GEE stages, and only `join()`s the result right before it's
actually needed for `assemble_summary()`. In practice this fully overlaps
the Overpass wait with Sentinel-2/groundwater's GEE calls instead of adding
to total wall time.

## The `on_assembled` hook (the one deliberate coupling point)

`run_analysis()` returns only `(site_summary, map_html)` — the raw
per-module `results` (GeoDataFrames, `ee.Image`s) are local variables, never
returned. Report generation needs those same raw objects to build static
maps (so it doesn't have to re-run GEE/OSM/SerpApi queries a second time),
so the orchestrator exposes one small optional hook:

```python
def run_analysis(..., on_assembled: Optional[Callable[[AOI, dict, dict], None]] = None):
    ...
    site_summary = assemble_summary(aoi, results, run_success=run_success)
    if on_assembled is not None:
        on_assembled(aoi, results, site_summary)   # fault-isolated, same
                                                     # timeout/progress
                                                     # machinery as every
                                                     # other stage
    ...
```

`aI_agents/run_worker.py` binds this to
`report_assets.generate_report_assets`. If `on_assembled` is `None` (true
for every direct caller of `run_analysis`), this is a complete no-op: zero
behavior or performance change for anything that doesn't pass it.

## One grounded Q&A function, reused for every AI surface

`aI_agents/qa.py` has one function (`answer_question`) that grounds an LLM
call directly in the relevant sections of `site_summary.json` for a stated
purpose and preferences, and reasons about the site in plain language,
citing the real numbers it used, rather than reducing suitability to a
single weighted-factor score. Every "smart" surface in the app is a
differently-worded call through this same function (or its `qa.py`
siblings) and the same guardrail system prompt — see
[DESIGN_CHOICES_AND_LIMITATIONS.md](DESIGN_CHOICES_AND_LIMITATIONS.md) for
the full rationale and the exact guardrail rules every call is held to:

| Surface | Call |
|---|---|
| View Analysis → AI Insights | `qa.generate_strengths_concerns()` |
| Free-form chat | `qa_agent.route_and_answer()` → `qa.answer_question()`, optionally after one live tool call |
| Report chapters 1-5's "AI Interpretation" boxes | `qa.generate_report_interpretations()` (one consolidated call, 7 JSON fields) |
| Compare (2+ sites) | `qa.rank_sites()` / `qa_agent.rank_sites_with_tools()` |

---

## Module reference

Every module, and every top-level function/class in it. Leading-underscore
names are internal helpers (not imported by other modules); everything else
is a real public entry point another module calls.

### `data_analysis_pipeline/`

**`config.py`** — paths (`DATA_DIR`, `CACHE_DIR`, `RUNS_DIR`), API keys, and
Earth Engine bootstrap; loads `.env` once at import time. Also backs the
sidebar's runtime settings — every consumer elsewhere reads these as
`config.ATTR`, never a name-import, so changes below take effect
immediately, no restart needed.
- `init_earth_engine()` — authenticate/initialize GEE once per process.
- `_resolve_serpapi_key()` / `_resolve_groq_key()` — env var first, then a
  legacy out-of-repo key-file fallback.
- `set_env_key(name, value)` — set one API-key-style setting at runtime:
  updates `os.environ`, this module's matching attribute, and upserts it
  into `.env`. Used by the sidebar's "Save settings" button.
- `reset_data_dir(new_dir)` / `reset_runs_dir(new_dir)` — same idea for
  `DATA_DIR` (and everything derived from it: `VILLAGE_BOUNDARY_FILE`,
  `RIVER_POLYGON_FILE`, etc., recomputed via `_recompute_data_paths()`) and
  `RUNS_DIR`. Used by the sidebar's "Data directory"/"Run directory" fields.
- `reset_earth_engine()` — force the next `init_earth_engine()` call to
  actually re-initialize, after `GEE_PROJECT_ID` changes at runtime.
- `_upsert_env_file(name, value)` — shared helper: replace a `NAME=...`
  line in `.env` if present, else append one.

**`aoi.py`** — the area-of-interest polygon every `get_*` module takes.
- `class AOI` — lat/lon/radius + its polygon in a few CRSes.
- `build_aoi(lat, lon, radius_km)` — construct one.
- `to_ee_geometry(aoi)` — convert to an `ee.Geometry` for GEE calls.
- `read_local_vector(path, columns, aoi, source_crs)` — read a local
  GeoJSON/shapefile clipped to the AOI.

**`orchestrator.py`** — runs every stage, fault-isolated.
- `run_analysis(lat, lon, radius_km, ...)` — the pipeline entry point (see
  "The pipeline" above).
- `class RunCancelled` — raised when a run is cooperatively stopped mid-way.
- `_call_with_hard_timeout` / `_start_hard_timeout_call` /
  `_join_hard_timeout_call` — run a function with a hard wall-clock cutoff,
  optionally on a background thread (used for the `on_assembled` hook and
  the two overlapped OSM stages).
- `_safe_call` / `_start_safe_call` / `_finish_safe_call` — wrap one `get_*`
  stage call, catching any exception into a `FAILURE_CONFIDENCE` envelope.
- `_failure_envelope(dataset, source, params, error)` — build that envelope.
- `_combine_osm_results(roads_env, rail_env, water_env)` — merge the three
  separately-fetched OSM stages back into one `osm` section.
- `_configure_run_logger(log_path)` — per-run `run.log` file handler.

**`assemble_summary.py`** — builds the final `site_summary.json`.
- `assemble_summary(aoi, results, run_success)` — the only public function;
  flattens every stage's envelope into the documented schema.
- `_section_summary`, `_unavailable`, `_dict_to_list` — internal shaping
  helpers.

**`runs.py`** — pure (no Streamlit) run-directory helpers, shared by
`frontend` and `aI_agents` (see the cycle-avoidance note above).
- `list_previous_runs()` — every completed run under `RUNS_DIR`, newest
  first, with a display label built from its own `site_summary.json`.
- `load_default_site()` — load the bundled `runs/manual/` fast-path site.
- `resolve_run_output_dir(lat, lon, radius_km, label)` /
  `make_run_dir_name(...)` / `slugify_run_label(label)` — naming a fresh
  run's output directory.
- `unique_dir(path)` — append `_2`, `_3`, ... if `path` already exists.
- `rename_run_dir(old_dir, new_label)` — rename a run directory in place.
- `failed_sections(site_summary)` — list section names with
  `available: false`, for the partial-failure warning banner.

**`build_map.py`** — the interactive Folium map + its summary panel.
- `build_map(aoi, results, site_summary)` — the map builder: layered
  roads/rail/water/hydrology/groundwater/terrain/NDVI/land-cover/heat/
  nearby-places, plus the collapsible HTML panel (`_build_panel_html`).
- `build_reference_map(site_lat, site_lon, ref_lat, ref_lon, ...)` — a
  standalone route map between the site and a custom-facts reference point.
- `_build_panel_html(site_summary)` — the ~15-block HTML summary panel
  embedded in the map.
- `_section`, `_row`, `_meter`, `_fmt`, `_fmt_distance`,
  `_guideline_rates_html`, `_seasonal_table_html`, `_esc` — internal HTML
  fragment builders for that panel.

**`custom_facts.py`** — "Custom Facts" (reference location, plot
price/area), addable to any existing run after the fact.
- `merge_custom_facts(site_summary, reference_location, plot_price,
  plot_area, area_unit)` — the only public entry point; writes
  `reference_location`/`land_price` into a copy of `site_summary`.
- `haversine_km(lat1, lon1, lat2, lon2)` — straight-line distance.
- `fetch_road_travel_time(...)` — best-effort driving time/distance via the
  free public OSRM router.
- `fetch_transit_time(...)` — best-effort public-transit itinerary via
  Google Maps Directions.
- `price_per_acre(price, area, area_unit)` / `area_to_sqft(area, area_unit)`
  — unit normalization for the price/area inputs.

**`location_search.py`** — resolving what the user typed into a lat/lon.
- `parse_latlon_query(query)` — parse raw `"23.04, 76.21"`-style text.
- `search_google_maps(query, limit)` — live Google Maps place search (via
  SerpApi) for the location picker's search box.
- `search_places(query, limit)` — offline MP-only fuzzy search over the
  cached village/tehsil/district index (`data_analysis_pipeline/mp_location_index.parquet`,
  built by `scripts/build_location_search_index.py`); no network call.
- `_load_index()` — lazily load and cache that parquet index.

**`mp_boundary.py`** — `is_in_mp(lat, lon)`: point-in-polygon check against
the cached MP outline (`data_analysis_pipeline/mp_boundary.geojson`,
`scripts/build_mp_boundary_cache.py`); `_load_boundary()` loads/caches it.

**`main.py`** — CLI entrypoint (`python -m data_analysis_pipeline.main
--lat --lon --radius --out-dir`).
- `main()` — parse args, run the pipeline, write `site_summary.json`/
  `map.html`/`run.log`.
- `_json_safe(value)` — `json.dump(..., default=_json_safe)` handler for
  numpy/pandas scalar types.

**`get_*.py`** — one module per data source, each returning one
self-contained `dict` section of `site_summary.json` (see
[DATA_SOURCES.md](DATA_SOURCES.md) for what each one actually measures):
- `get_admin_context.py` → `get_admin_context(aoi)` — village/tehsil/
  district, census, govt guideline land rates (`_lookup_govt_guideline_rates`,
  `_attach_village_guideline_rates`, `_district_guideline_table`, and
  smaller text/number-cleaning helpers).
- `get_groundwater.py` → `get_groundwater(aoi)` — CGWB well levels, latest +
  5-year mean + seasonal pre-monsoon/monsoon breakdown (`_seasonal_levels`,
  `_season_level`, `_observed_months`).
- `get_hydrology.py` → `get_hydrology(aoi)` — named water bodies/rivers,
  floodplain proximity.
- `get_land_cover.py` → `get_land_cover(aoi)` — ESA WorldCover class shares.
- `get_land_surface_temperature.py` → `get_land_surface_temperature(aoi)`
  (Landsat annual mean/min/max) and `get_air_temperature(aoi)` (ERA5-Land
  annual mean + coldest/warmest month); `_default_lst_date_range`,
  `_default_air_temp_years` pick the query window.
- `get_nearby_places.py` → `get_nearby_places(aoi, radius_km)` — SerpApi
  category search (schools/hospitals/markets/...); `_serpapi_search` is the
  raw API call (reused directly by `aI_agents.qa_agent`'s ad-hoc chat tool);
  `google_maps_link`, `place_popup` build per-place display strings.
- `get_osm_features.py` → `get_roads(aoi)`, `get_railways(aoi)`,
  `get_water(aoi)`, and the combined `get_osm_features(aoi)`; `_fetch_osm_layer`
  wraps one Overpass query, `_nearest_feature`/`_find_nearest_major_road_beyond_aoi`
  compute proximity.
- `get_rainfall.py` → `get_rainfall(aoi, years)` — CHIRPS + IMD annual/
  monsoon totals; `_imd_year_metrics`, `_metrics_by_year`, `_difference_pct`
  reconcile the two sources.
- `get_sentinel_features.py` → `get_sentinel_features(aoi)` — NDVI stats +
  peak-period detection (`_find_peak_period`, `_ndvi_composite_stats`).
- `get_terrain.py` → `get_terrain(aoi)` — SRTM elevation/slope stats.

### `aI_agents/`

**`groq_client.py`** — the Groq client core (all that survives from the
legacy notebook's `decision_app/chat.py`).
- `class ChatUnavailable` — raised when no key/package/persistent rate
  limit; every AI-facing UI surface catches this specifically.
- `_get_client()` / `_require_client()` — lazily construct (and cache) the
  Groq client.
- `_create_completion(client, **kwargs)` — one retry on a transient
  rate-limit error.

**`qa.py`** — the one grounded Q&A function and its canned-question
wrappers (see "One grounded Q&A function" above).
- `answer_question(site_summary, question, purpose, preferences, history,
  additional_sites, ...)` — free-form grounded Q&A, single or multi-site.
- `generate_strengths_concerns(site_summary, purpose, preferences)` — View
  Analysis's AI Insights.
- `generate_report_interpretations(site_summary, purpose, preferences)` —
  the report's one consolidated 7-field Groq call.
- `rank_sites(sites, purpose, preferences, extra_context)` — the ranked
  comparison judgement (verbal reasoning, then a 0-100 score/verdict).
- `_curate_site_context(site_summary, compact)` — the token-budget-aware
  grounding excerpt every call above is built from; `compact=True` drops
  addresses/full guideline tables/named lists for multi-site calls.
- `_settlement_context`, `_nearby_category_excerpt`, `_nearest_named`,
  `_trim_seasonal`, `_section`, `_round`, `_drop_empty` — internal excerpt
  builders feeding `_curate_site_context`.

**`qa_agent.py`** — tool-routing layer for free-form chat.
- `route_and_answer(site_summary, question, purpose, preferences, history,
  site_label, additional_sites, summary_path)` — decide whether the
  question needs a live tool, fetch it if so, then call `qa.answer_question`.
- `rank_sites_with_tools(sites, purpose, preferences)` — `qa.rank_sites`,
  after one optional live tool fetch grounded on the primary site.
- `_decide_and_fetch_tool(...)` — the routing Groq call + dispatch to
  whichever tool it picked.
- `_fetch_onefivenine_tehsil_info`, `_nearby_category_search_text`
  (wraps `get_nearby_places._serpapi_search` for an ad-hoc category),
  `_wikipedia_summary_text` — the three live tools.
- `_routing_system_prompt(...)` — builds that routing call's prompt.
- `_other_run_choices(exclude_summary_path, limit)` — other saved runs the
  user might be naming by mention (uses `data_analysis_pipeline.runs.list_previous_runs`).

**`compare.py`** — thin wrapper pairing a ranking with its fact table.
- `compare(sites, purpose, preferences)` — `(ranking_result, facts_table)`.
- `side_by_side_facts(sites)` — the N-column comparison table, drawn from
  the same curated context the ranking was grounded in.
- `_round(value, decimals)` — display rounding (fixes raw floats like
  `650.37498650989`).

**`report_assets.py`** — static chart/map generation for reports, run once
per pipeline run (see the `on_assembled` hook above).
- `generate_report_assets(aoi, results, site_summary, assets_dir)` — the
  entry point; calls every generator below, fault-isolated, and writes
  `manifest.json`.
- Maps: `_site_context_map`, `_groundwater_map`, `_water_bodies_map`,
  `_reference_location_map`, `_nearby_places_map`, `_ndvi_map`,
  `_land_cover_map`.
- Charts: `_groundwater_chart`, `_slope_class_chart`.
- Shared plumbing: `_fetch_satellite_labels_basemap`/`_fetch_street_basemap`
  (Esri tiles), `_new_map_fig`/`_finish_map`/`_draw_aoi_boundary`/
  `_draw_site_marker`/`_set_aoi_extent`/`_project_bounds`/`_draw_water_basemap`
  (matplotlib scaffolding), `_ee_region`/`_fetch_ee_thumbnail`/
  `_draw_site_star_on_thumb` (GEE raster thumbnails), `_well_2025_median`,
  `_haversine_km`.

**`run_worker.py`** — `run_analysis_worker(lat, lon, radius_km, log_path,
progress_queue, cancel_event, result)`: the background-thread entry point
`frontend`'s "Run full analysis" button starts, binding
`report_assets.generate_report_assets` as `run_analysis`'s `on_assembled`.

**`build_report.py`** — the 6-chapter report assembler (see
[REPORT_METHODOLOGY.md](REPORT_METHODOLOGY.md) for the full chapter-by-chapter
breakdown).
- `generate_report(site_summary, summary_path)` — assemble the full report
  HTML: reads `report_assets/manifest.json`, makes the one Groq call, embeds
  every chart/map.
- `report_to_pdf_bytes(html)` — render that HTML to a PDF via `xhtml2pdf`.
- `_cover`, `_chapter_location_admin`, `_chapter_water`,
  `_chapter_climate_land`, `_chapter_locality`, `_chapter_summary`,
  `_sources_note_html` — one builder per chapter.
- `_card_open`, `_fact`, `_table`, `_bullets`, `_ai_box`, `_esc`,
  `_fmt_num`, `_month_name`, `_img_tag`, `_load_manifest`,
  `_guideline_rates_html`, `_named_water_features_table`,
  `_seasonal_table`, `_rainfall_table`, `_land_cover_table`,
  `_nearby_categories_table` — shared HTML-fragment builders (also imported
  directly by `build_compare_report.py`).

**`build_compare_report.py`** — the comparison report's HTML, reusing
`build_report.py`'s card/table primitives.
- `generate_compare_html(sites, ranking_result, facts, purpose)` — the only
  public entry point.
- `_site_facts_card`, `_facts_table`, `_rank_badge` — per-site/summary
  fragment builders.

### `frontend/`

**`streamlit_app.py`** — the single app entry point
(`streamlit run frontend/streamlit_app.py`); see "The actual user workflow"
above for how these fit together.
- `main()` — page setup, the 3-way source picker, and dispatch to whichever
  branch is active.
- `_render_results(site_summary, map_html, summary_path)` — the shared
  post-analysis view (site header, action buttons, chat) for all 3 sources.
- `_site_header`, `_edit_purpose_dialog` — the identity line + purpose
  editor popup.
- `_action_buttons`, `_secondary_actions` — View Analysis/Generate
  Report/Compare/Pick-a-different-site buttons and their dialogs.
- `_view_analysis_dialog`, `_generate_report_dialog`, `_compare_dialog` —
  the three `st.dialog` modals.
- `_custom_facts_section`, `_key_points_section` — the two sections nested
  inside View Analysis.
- `_render_chat` — the free-flowing chat box.
- `_run_progress_dialog` — polls a fresh run's progress queue until done.
- `_render_sidebar` — thin wrapper that renders `_render_api_key_settings`
  inside `st.sidebar`.
- `_render_api_key_settings` — the **⚙️ Settings** panel: `GROQ_API_KEY`,
  `SERPAPI_KEY`, `GEE_PROJECT_ID` fields; "Run directory" and "Data
  directory" fields; the "📥 Download source data" button (enabled only
  when `config.REFERENCE_DATA_BUNDLE_URL` is configured); calls
  `config.set_env_key`/`config.reset_data_dir`/`runs_module.reset_runs_dir`
  and, after a key change, `groq_client.reset_client()` /
  `config.reset_earth_engine()` so it takes effect without a restart.
- `_key_status(value)` — "configured (…last 4 chars)" vs. "not set", used by
  the settings panel's placeholder text.
- `_data_download_success_dialog(path)` — the `st.dialog` popup confirming a
  finished source-data download, with the extracted path to copy.
- `_init_session_state`, `_sync_active_site` — session-state defaults and
  the "reset any open dialog when the active site changes" guard.
- `_map_html_with_default_layers` — post-processes the Folium map HTML to
  tick a few layers on by default and disable scroll-wheel zoom.
- `_fmt_hm(minutes)` — "H:MM hr" duration formatting.

**`location_picker.py`** — the search box + pin-droppable map widget.
- `render_location_picker_with_satellite(key_prefix, heading, default_lat,
  default_lon, secondary_marker)` — the only public entry point; used for
  both the main site picker and the reference-location picker.
- `parse_google_maps_url(url)` / `resolve_google_maps_url(url)` — extract
  `(lat, lon)` from a pasted (possibly shortened) Google Maps link.
- `_looks_like_maps_url(text)` — decide whether typed text is a URL.
