# Report Methodology

The generated report is deliberately split **~75% deterministic, ~25% AI**:
almost everything in it — every chart, map, table, and stat — is real,
pre-computed data pulled straight from that run's `site_summary.json` and
pre-rendered image assets. Exactly **one** Groq call per report produces the
AI-written interpretation text. This split exists because Groq's free tier
has real per-minute/per-day token budgets, and because a report whose numbers
come straight from the pipeline's own output is more trustworthy than one an
LLM re-states from a prompt.

## Assets are generated once, at analysis time — not at report-click time

Earlier versions of report generation built every chart fresh (and, briefly,
re-fetched a live GEE query) on every "Generate Report" click. The current
design generates every chart and map exactly once, **during the pipeline run
itself**, via a callback (`on_assembled`, see
[ARCHITECTURE.md](ARCHITECTURE.md)) that hands `aI_agents/report_assets.py`
the same raw `results` (GeoDataFrames, `ee.Image`s) the pipeline already
computed — no second GEE/OSM/SerpApi call. Each chart/map generator is
wrapped individually so one failing asset (e.g. a basemap tile fetch) never
blocks the others; successes are recorded in that run's
`report_assets/manifest.json`, failures go to a `warnings` list and are
simply omitted. `build_report.py` is a pure assembler at report-click time:
it reads the manifest, embeds each listed PNG as base64, and makes the one
Groq call — no live data-fetching happens when a user clicks "Generate
Report."

Basemap choice varies by map:
- Water/overview maps (site context, groundwater, water bodies): satellite +
  place-labels.
- Reference-location and nearby-places maps: street-view.
- NDVI and land-cover maps: no basemap — a raw GEE thumbnail with a color
  legend, since a basemap would visually clash with a full-AOI color raster.

## One Groq call, seven fields

`aI_agents.qa.generate_report_interpretations(site_summary, purpose, preferences)`
makes one JSON-mode Groq call requesting seven fields: four short (2-3
sentence) per-chapter interpretations (`location_admin`, `water_hydrology`,
`climate_land`, `locality_context`) plus `strengths`/`concerns`/`trade_offs`
(3-6 bullets each) for the overall-summary chapter. Same guardrail system
prompt as every other Q&A surface in the app (see
[ARCHITECTURE.md](ARCHITECTURE.md)): never invent a figure, disclose missing
data instead of asserting a false absence, no price-trend claims without
trend data. If a data category (e.g. nearby places) has
`serpapi_enabled: false` for that run, the guardrail prompt explicitly
instructs the model to say **"N/A"** for that category rather than assert
"there are no schools/markets nearby" — a real bug caught during
development, where an early version answered from an empty category as if
it were a confirmed real-world absence.

## Chapter structure

**Cover** — site name/coordinates, stated buying purpose, land price (if
supplied), `site_context_map`.

1. **Location & Administration** — village/tehsil/district, roads and rail
   connectivity, reference-location travel details (if set), government
   guideline land-rate table. Ends with a short AI interpretation.
2. **Water & Hydrology** — groundwater map + seasonal (pre-monsoon/monsoon)
   table + chart, 3-year rainfall table, named water-bodies map + table,
   floodplain proximity. Ends with a short AI interpretation.
3. **Climate & Land Use** — annual temperature stats (air + land-surface),
   NDVI stats + map, land-cover class table + map, elevation/slope stats +
   chart. Ends with a short AI interpretation.
4. **Human / Locality Context** — population/census figures, nearest
   settlement of any type *and* nearest developed city/town (shown
   separately — the fix for the town-distance bug described in
   [ARCHITECTURE.md](ARCHITECTURE.md)), built-up land share, nearby-place
   categories table + map. Ends with a short AI interpretation.
5. **Overall Summary & Suitability** — fully AI: strengths / concerns /
   trade-offs for the stated purpose, followed immediately (same chapter, no
   new heading or page break) by a short deterministic **Note:** paragraph
   naming each section's data source and any run-specific
   warnings/limitations — kept terse and factual, no AI involved in that
   part.

Every chart/map, deterministic table, and interpretation sentence traces
back to a specific field in that run's `site_summary.json` or
`report_assets/manifest.json` — nothing in the report is generated purely
from the AI's own knowledge.
