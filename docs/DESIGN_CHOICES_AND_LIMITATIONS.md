# Design Choices & Limitations

## The product is built in two stages

**Stage 1 — Data Analysis.** For a given site (lat/lon + radius), real data
is gathered and processed from satellite imagery, groundwater records,
terrain, hydrology, roads/rail, nearby places, and administration/census
sources. This stage produces three artifacts, all written to that run's
folder under `runs/`: a structured **`site_summary.json`** (every measured
fact, with its source and a plain "unavailable" marker for anything that
couldn't be fetched), an **interactive Folium map**, and a set of
**pre-rendered static images** (the charts/maps the PDF/HTML report
embeds). These three artifacts are the only things Stage 2 ever sees — no
live satellite/OSM query happens past this point.

**Stage 2 — Reasoning Agent.** Given Stage 1's `site_summary.json` and a
user-stated buying purpose (farming, housing, resort, warehouse, ...), an
LLM reasons about the site's suitability, optionally reaching for one of
three live tools when the pipeline's own data isn't enough: a Wikipedia
summary, a onefivenine.com tehsil-level page (village/population/rivers/
schools not in the pipeline's own data), or an ad-hoc SerpApi search for a
nearby-place category the site's fixed 8 categories don't cover. This one
reasoning layer powers three product surfaces: the **chat box** (free-form
Q&A), the **generated report**'s AI-interpretation text, and **Compare**'s
ranked, reasoned shortlist across sites.

The two stages map onto the codebase mostly, but not exactly, along package
lines: Stage 1 is `data_analysis_pipeline/` plus one module physically
living in `aI_agents/` for import reasons — `report_assets.py` (the static
report images) and `run_worker.py` (which binds it to the pipeline via the
`on_assembled` hook) are Stage 1 work even though they sit in the AI
package. Stage 2 is the rest of `aI_agents/` — `qa.py`, `qa_agent.py`,
`compare.py`, `build_report.py`, `build_compare_report.py`,
`groq_client.py`. `frontend/` is neither stage — it's the UI that triggers
Stage 1 (via `run_worker`) and calls into Stage 2 (via `qa`/`qa_agent`/
`compare`) for every AI-facing feature.

```
 STAGE 1 — Data Analysis
 data_analysis_pipeline/ (orchestrator + one get_*.py per source)
 + aI_agents/report_assets.py, run_worker.py
                        │
                        ▼
 site_summary.json  +  map.html  +  report_assets/
 (the ONLY inputs Stage 2 ever sees — no live satellite/OSM
  call happens past this point)
                        │
                        ▼
 STAGE 2 — Reasoning Agent
 aI_agents/qa.py, qa_agent.py, compare.py,
 build_report.py, build_compare_report.py, groq_client.py
                        │
                        ▼
 chat Q&A   ·   report interpretation   ·   ranked Compare
```

## Stage 1: Data Analysis — architecture

`data_analysis_pipeline/orchestrator.py::run_analysis(lat, lon, radius_km)`
builds an AOI (area-of-interest) polygon, then runs one stage per data
source — Sentinel-2 NDVI, ESA WorldCover land cover, Landsat land-surface
temperature, ERA5-Land air temperature, SRTM terrain, CHIRPS+IMD rainfall,
and CGWB groundwater from Google Earth Engine and local reference files;
roads, railways, water bodies, and settlements from OpenStreetMap; nearby
places from SerpApi; village/tehsil/district/census/guideline-rates from
Survey-of-India boundaries — before handing everything to
`assemble_summary.py`, which flattens every stage's result into the final
`site_summary.json`.

Two decisions specific to this stage:

- **Fault isolation.** Every stage runs through a hard-timeout wrapper that
  catches any exception or timeout and records a `available: false`
  envelope for that one section instead of aborting the run. A run can
  complete successfully with some sections unavailable — that's a normal
  outcome here, not an error state, and it's exactly why Stage 2's
  guardrails treat a missing section as "not fetched," never as "confirmed
  absent" (see Stage 2 below).
- **Two stages overlapped on background threads.** OSM railway and
  water-body queries (the pipeline's slowest, least reliable dependency,
  via the public Overpass API) are `start()`ed on background threads
  immediately and only `join()`ed right before `assemble_summary()` needs
  them — overlapping their network wait with unrelated GEE calls instead of
  adding to total wall time.

Two more artifacts come out of this same stage, both built from the exact
same raw per-module results (GeoDataFrames, `ee.Image`s) so neither is
derived from the other or re-fetches anything:

- **The interactive Folium map** (`data_analysis_pipeline/build_map.py`) —
  the map `frontend`'s View Analysis dialog embeds directly.
- **Static report images** (`aI_agents/report_assets.py`) — every chart and
  map the PDF/HTML report needs, rendered once via a small `on_assembled`
  callback hook the orchestrator calls right after assembling
  `site_summary.json`, and recorded in that run's
  `report_assets/manifest.json`. This means "Generate Report" (Stage 2)
  never triggers a second GEE/OSM/SerpApi call — it only ever reads what
  Stage 1 already produced.

## Stage 2: Reasoning Agent — architecture

Every AI call in this stage funnels through **one grounded Q&A function**
(`aI_agents/qa.py::answer_question`), which builds a token-budget-aware
excerpt of `site_summary.json` (`_curate_site_context` — headline values
per section plus the *nearest* place per category, not the full
`nearby_places` dump, which alone can be 35-110 KB) and asks an LLM to
reason about it in plain language for the user's stated purpose, citing the
real numbers it used, rather than reducing suitability to a weighted
numeric score. Three sibling functions reuse the same grounding + guardrails
for the app's other AI-facing needs: `generate_strengths_concerns` (View
Analysis's AI Insights), `generate_report_interpretations` (the report's
one consolidated 7-field call), and `rank_sites` (Compare's per-site
reasoning + score/verdict).

**Tool routing** (`aI_agents/qa_agent.py::route_and_answer`) sits in front
of chat and Compare specifically: given a question, it first decides
whether `site_summary.json` alone already answers it (true for most
questions — the curated context covers the whole pipeline output), and only
reaches for one live tool when it doesn't:

- **`wikipedia_summary`** — general background on a named place, not in
  the pipeline's own data at all.
- **`onefivenine_tehsil_info`** — a public onefivenine.com page with
  tehsil-level detail the pipeline doesn't fetch (full village list,
  population, rivers, nearby railway stations, registered companies).
- **`nearby_category_search`** — an ad-hoc SerpApi search (reusing
  `data_analysis_pipeline.get_nearby_places._serpapi_search` directly) for
  a place category outside the site's fixed 8 (e.g. "petrol pump", "ATM").

**Guardrails every call in this stage is held to** (the same system-prompt
rules across chat, AI Insights, report interpretation, and Compare):

- **Never invent a measurement, place name, or figure** not present in the
  curated data it was given.
- **A missing or empty section means "not found in our data," never a
  confirmed real-world fact.** Caught live during development: an early
  report interpretation, given an empty `nearby_places` section, wrote "no
  schools, markets, or restaurants nearby" as though that were an observed
  fact, when the real cause was that a live places search had never run for
  that run (SerpApi disabled/unavailable). Every prompt now checks a
  `nearby_places_search_was_performed` flag (and the equivalent for other
  optional sections) and is instructed to say **"N/A"** instead of
  asserting an absence.
- **Groundwater depth is described by its 5-year mean, not the single
  latest reading** — a well reading can be several metres off from typical
  conditions depending on the monsoon cycle (confirmed on a real run: 2 m
  "latest" vs. a 6 m five-year mean for the same well); the latest reading
  is only used for explicit "as of `<date>`" framing.
- **Physical suitability is distinguished from legal suitability** — the
  data covers terrain, water, climate, and access; it says nothing about
  land title, zoning, or permits, and every prompt is told to say so rather
  than imply otherwise.
- **No price-trend or appreciation claims** — a supplied land price is a
  single point-in-time snapshot with no historical series behind it;
  proximity to towns/roads/industry may be offered only as an explicitly
  labeled rough proxy, never as a measurement of appreciation.

**Why one reasoning path pays off in practice**: an early version, asked
about a site's "financial growth potential," answered as if the area had
"no well-developed town with schools and restaurants nearby" — when a town
5.3 km away, with real named schools and restaurants, was already sitting
in `site_summary.json`. It just wasn't being surfaced into what the model
saw. Because there's one grounding function (`_curate_site_context`) behind
every surface, fixing that one context-building step fixed chat, the
report, and Compare all at once — they could not have drifted apart the
way a separate scoring path and a separate explanation path might have.

## Other cross-cutting design choices

**The report: pre-generated assets, one consolidated Groq call, ~75/25
split.** Building on Stage 1's pre-rendered images, `build_report.py` is a
pure assembler at report-click time — it reads `report_assets/manifest.json`,
embeds each listed PNG as base64, and makes exactly **one** consolidated
Groq call requesting all seven AI-written fields the report needs (four
per-chapter interpretations plus strengths/concerns/trade-offs) in a single
JSON-mode response. The result is a report that's ~75% deterministic
content (every number, chart, and map traces to a specific
`site_summary.json` field) and one Groq call total — both to respect the
free tier's token budget and because a report whose numbers come straight
from the pipeline's own output is more trustworthy than one an LLM
re-states from a prompt. Full chapter-by-chapter structure:
[REPORT_METHODOLOGY.md](REPORT_METHODOLOGY.md).

**Bundled demo runs need no API keys at all.** `runs/` ships 7 real,
complete Stage 1 outputs. Every API-dependent capability sits behind a
specific, avoidable action: `GEE_PROJECT_ID` is only touched by running a
**fresh** Stage 1 analysis (never by browsing an existing run, viewing its
map, or generating its report — those read entirely from disk), and
`SERPAPI_KEY` is only touched by a fresh run's nearby-places fetch or the
ad-hoc "search another category" chat tool. `GROQ_API_KEY` is the one key
genuinely needed for any Stage 2 surface (chat, AI Insights, report
interpretation, Compare) — everything else is pre-computed and viewable
with zero configuration. All three keys, the reference-data directory, and
a one-click download of that reference data (from a Drive link the
deployer configures once via `REFERENCE_DATA_BUNDLE_URL`, never something
an end user pastes in) can all be set from the app's own sidebar at
runtime — no `.env` edit or restart required, though the sidebar does
write its changes back into `.env` so they persist across restarts too.

**One cohesive package with a strict, one-way dependency direction.**
`data_analysis_pipeline` (Stage 1), `aI_agents` (Stage 2), and `frontend`
(UI) are three plain top-level packages: `frontend` depends on `aI_agents`,
both depend on `data_analysis_pipeline`, and nothing ever imports back the
other way. This keeps Stage 1 testable and runnable completely on its own
(no LLM, no Streamlit — `python -m data_analysis_pipeline.main` is a full
pipeline run by itself), keeps Stage 2 swappable behind one model client
(`aI_agents/groq_client.py`) without touching pipeline or UI code, and
means a shared helper (like listing previous runs, needed by both the chat
tool-router and the UI) has one obvious home — the lowest layer that needs
it — rather than either layer reaching into the other.

## Known limitations

- **Groq free-tier limits** — requests-per-minute/day and tokens-per-minute
  caps are real constraints; multi-site comparison uses a compacted
  grounding context (`_curate_site_context(..., compact=True)`)
  specifically to stay under the TPM budget with more than one site.
- **Madhya Pradesh coverage only** — the reference dataset (village
  boundaries, guideline rates, census) is MP-specific; the pipeline will
  not produce administration/census data for a location outside MP, and
  `data_analysis_pipeline.mp_boundary.is_in_mp` blocks running a fresh
  analysis outside the state entirely.
- **SerpApi usage limits** — nearby-places search depends on a metered
  third-party API; the app degrades cleanly (the AI says "N/A" rather than
  assert a false absence — see Stage 2's guardrails above) when it's
  unavailable or disabled via `DISABLE_SERPAPI`.
- **Overpass (OSM) reliability** — railway/water-body/settlement queries
  hit the public Overpass API, which can be slow or rate-limited; these
  stages run fault-isolated so a slow/failed OSM query doesn't fail the
  whole run, but the affected section is simply marked unavailable rather
  than retried indefinitely.
- **No true monthly rainfall or monthly land-surface temperature** —
  rainfall and LST are reported as annual statistics; the underlying
  daily/16-day-composite source data isn't currently aggregated to a
  monthly series.
- **Session-only saved sites and chat history** — no persistence layer
  yet; closing the browser tab loses bookmarked sites and chat sessions,
  since `st.session_state` is the only place either is stored.
- **Streamlit `st.dialog` quirks** — dialogs are gated by a
  `st.session_state.view_mode` flag rather than Streamlit's own dialog
  lifecycle (there's no built-in "on close" callback), so opening one
  reliably requires an explicit `st.rerun()` right after setting that flag,
  and anything that should implicitly dismiss an open dialog (asking a chat
  question, switching the site-source tab) has to clear the flag itself.
