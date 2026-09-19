# Data Sources

Every dataset the pipeline uses, what it feeds in `site_summary.json`, and
its known limitations. All satellite/reanalysis datasets are queried live via
Google Earth Engine (GEE); "local" datasets are static files shipped in the
shared `data/` folder (`LAND_INTEL_DATA_DIR`).

| Dataset | Type | Feeds | Limitations |
|---|---|---|---|
| **Sentinel-2 SR Harmonized** (`COPERNICUS/S2_SR_HARMONIZED`) | GEE, live | `ndvi` — vegetation index (mean/min/max, area above 0.30/0.50 thresholds), whole-year + peak-period composites | A single median composite for the year; doesn't capture seasonal NDVI swings within the year. |
| **ESA WorldCover v200** (10 m, `ESA/WorldCover/v200`) | GEE, live | `land_cover` — per-class area/percentage breakdown (built-up, cropland, forest, etc.) | 2021 vintage; land cover may have changed since. |
| **SRTM 1 arc-second** (`USGS/SRTMGL1_003`) | GEE, live | `terrain` — elevation range/mean, slope mean/max/90th-percentile, slope-class shares | ~30 m resolution; not fine enough for lot-level grading decisions. |
| **Landsat 8 C2 L2 ST_B10** (`LANDSAT/LC08/C02/T1_L2`) | GEE, live | `land_surface_temperature` — annual mean/min/max | A single annual median composite; no monthly signal (16-day revisit + cloud-cover filtering makes a reliable monthly series out of scope for v1). |
| **ERA5-Land monthly aggregated** (`ECMWF/ERA5_LAND/MONTHLY_AGGR`) | GEE, live | `air_temperature` — annual mean, coldest/warmest month | Reanalysis data (modeled), not a ground station reading; ~9 km grid. |
| **CHIRPS v3 daily** (`UCSB-CHC/CHIRPS/V3/DAILY_RNL`) + **IMD 0.25° gridded daily** | GEE (CHIRPS) + local NetCDF (IMD), live/local | `rainfall` — annual + monsoon-season totals, rain days, wettest day, 3-year (2023-2025) history | Two independent estimates shown side by side (they can disagree); no true monthly breakdown yet. |
| **CGWB manual monthly groundwater monitoring** | Local CSV (`ground_water_level_manual_monthly_madhya_pradesh_1974_2025.csv`) | `groundwater` — nearest well depth, 5-year range/mean, pre-monsoon/monsoon seasonal table, trend | Sparse well network — the "nearest well" can be several km away; manual (not telemetered) readings. |
| **OpenStreetMap via OSMnx** | Live Overpass API query | `roads`, `railways`, `water_resources` (waterways/waterbodies from OSM tags), `settlements` (nearby villages/towns by name+type+distance) | Crowd-sourced — coverage/tagging quality varies by area; the public Overpass instance can be slow or rate-limited (these stages run fault-isolated and time out gracefully). |
| **SAC water bodies** (`wb_sac_mp.GeoJSON`) + **river polygons** (`river_polygon.GeoJSON`) | Local GeoJSON | `hydrology` — named water bodies and rivers within the search window, with area/distance; floodplain proxy | River floodplain extent is approximated from polygon coverage, not a modeled flood-risk layer. |
| **Survey of India village boundaries** (`vb_soi_mp.GeoJSON`) + **census_derieved.csv** | Local GeoJSON + CSV | `administration` — village/tehsil/district/state, village count in AOI; census — population, literacy, SC/ST%, land-use shares | Census figures are from the most recent available Census year, not live-updated. |
| **MP Govt district guideline land rates** (FY2026-27) | Local CSV/PDF-derived tables | `administration.govt_guideline_rates` | Guideline (registration/stamp-duty) rates, not market transaction prices — the app explicitly never presents these as a market-price estimate. |
| **SerpApi / Google Maps local results** | Live API, metered | `nearby_places` — schools, hospitals, markets, restaurants, hotels, tourist attractions, industrial businesses, warehouses/logistics, within a 10 km search radius | Metered third-party API with a usage cap; the app degrades to "unavailable" (and instructs the AI to say "N/A" rather than assert a false absence) when the key is missing, the quota is hit, or `DISABLE_SERPAPI` is set. |

## Reference tile providers (report maps only)

Static maps embedded in generated reports use free, keyless basemap tiles —
**Esri World Imagery**, **Esri World Street Map**, and **Esri Reference /
World Boundaries and Places** (via `contextily`) — chosen after confirming
empirically that both the raw OSM tile server and CartoDB's Voyager basemap
now block or require a key for this kind of scripted/bulk use.

## Licensing & attribution

None of the datasets above are bundled or redistributed as part of this
repository — the reference dataset is downloaded separately by whoever
deploys the app, and each source remains subject to its own license and
terms of use, independent of this project's [MIT license](../LICENSE).
This is a best-effort summary, not legal advice — verify current terms
directly with each source before any commercial use or redistribution,
especially for the Government of India datasets noted below.

**Open, well-established licenses:**
- **Copernicus (Sentinel-2, ERA5-Land)** — free and open under the EU's
  Copernicus data policy. Required attribution: *"Contains modified
  Copernicus Sentinel/Climate Change Service information [year]"*, with the
  standard disclaimer that neither the European Commission nor ECMWF is
  responsible for any use made of it.
- **ESA WorldCover v200** — CC BY 4.0. Cite as *Zanaga, D. et al. (2022).
  ESA WorldCover 10 m 2021 v200*, https://doi.org/10.5281/zenodo.7254221.
- **SRTM / Landsat 8 (USGS, NASA)** — U.S. Government work, public domain;
  no legal attribution requirement, though *"Courtesy of the U.S.
  Geological Survey"* is customary.
- **CHIRPS (UCSB Climate Hazards Center)** — freely available for any use;
  suggested citation: Funk, C. et al. (2015), *"The climate hazards
  infrared precipitation with stations — a new environmental record for
  monitoring extremes."* Scientific Data 2, 150066.
- **OpenStreetMap** — © OpenStreetMap contributors, [ODbL
  1.0](https://opendatacommons.org/licenses/odbl/) — attribution required;
  see [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright).
- **Esri basemap tiles** — used under Esri's free/keyless tile-service
  terms (display/attribution only, no bulk download or redistribution of
  the tiles themselves); attribution ("Source: Esri...") is baked directly
  into every generated map image.

**Government of India open data — verify current terms before reuse:**
CGWB groundwater levels, IMD gridded rainfall, Survey of India village
boundaries, Census of India derived figures, and MP Govt district
guideline land rates are all sourced from Indian government agencies,
generally under India's National Data Sharing and Accessibility Policy
(NDSAP) framework, which typically permits reuse with attribution to the
originating agency (Central Ground Water Board / Ministry of Jal Shakti;
India Meteorological Department; Survey of India; Office of the Registrar
General & Census Commissioner; Government of Madhya Pradesh). **Survey of
India boundary data in particular can carry additional usage
restrictions** (per India's National Map Policy) — confirm the exact terms
attached to your specific copy of this data before any commercial use,
redistribution, or public display beyond this kind of research/
demonstration context.

**Commercial third-party services (not open data):**
- **SerpApi / Google Maps** — metered commercial API; usage is bound by
  both [SerpApi's Terms of Service](https://serpapi.com/legal) and
  [Google Maps Platform's Terms of Service](https://cloud.google.com/maps-platform/terms),
  which apply to Google Maps content even when accessed through a
  third-party proxy. This app displays results live within a session and
  does not cache, store, or redistribute them beyond that.
- **Groq** — commercial LLM inference API; usage is bound by
  [Groq's Terms of Service](https://groq.com/terms-of-use/). Generated text
  (chat answers, report interpretations, comparisons) is grounded in this
  app's own pipeline data under explicit guardrails, but remains
  AI-generated content and should be independently verified.
