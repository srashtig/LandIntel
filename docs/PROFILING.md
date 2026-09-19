# Pipeline Profiling Report

Generated 2026-09-17 16:03:52 for lat=23.0401972, lon=76.2086806, radius=5.0km.

Based on a single representative run (`manual`/Budasa). The tool supports any lat/lon and pairing any two existing runs for Compare — a fuller multi-site sweep would sharpen this further; treat the numbers below as directionally correct, not exact for every possible site.

## Startup / import cost

> **⚠️ Observed anomaly, worth flagging for Cloud Run**: the very first
> time `import aI_agents.build_report` ran on this dev machine in this
> session, it took **902 seconds** (~15 minutes) — not the ~3s shown
> below. Every subsequent import (including the numbers below) was fast.
> This was not reproducible on retry, and it wasn't matplotlib's font
> cache (already built days earlier) — the exact cause wasn't isolated
> further. This strongly suggests *some* dependency in the chain
> (osmnx/scikit-learn/pyproj/contextily are the candidates that showed up
> in the import graph) does one-time setup work on first use in a fresh
> environment — e.g. a lazily-downloaded data file, a compiled cache, or a
> slow first DNS/network check. **Practical implication**: a brand-new
> Cloud Run container's first request could stall far longer than these
> numbers suggest. Recommend either a warm-up step baked into the
> container image build (import everything once at build time, not at
> first request) or a generous startup probe timeout, and re-measuring
> this specifically against a genuinely fresh container before finalizing
> that timeout.

**data_analysis_pipeline (pipeline only)**: 3.32s cold import (warm-cache
retry; see anomaly above for the true first-ever cold number observed)

**aI_agents.build_report (+ matplotlib/contextily/PIL)**: 3.02s cold
import (same caveat)

## Full pipeline run

Total wall time: **105.5s** (1.8 min). Peak RSS: **728 MB**. Avg CPU: 50%.

| Stage | Time | Peak RSS | Avg CPU | Overlaps with |
|---|---|---|---|---|
| AOI geometry | 0.1s | 0 MB | 0% | - |
| OSM roads | 0.1s | 0 MB | 0% | - |
| Sentinel-2 NDVI | 6.6s | 313 MB | 9% | OSM railways |
| OSM railways | 6.6s | 313 MB | 9% | Sentinel-2 NDVI |
| ESA WorldCover land cover | 2.1s | 308 MB | 1% | - |
| Landsat land-surface temperature | 4.8s | 288 MB | 0% | - |
| ERA5-Land air temperature | 0.5s | 103 MB | 2% | - |
| SRTM terrain (elevation/slope) | 4.2s | 104 MB | 1% | - |
| CGWB groundwater levels | 2.1s | 728 MB | 86% | OSM water |
| OSM water | 2.1s | 728 MB | 86% | CGWB groundwater levels |
| River floodplain and SAC water bodies | 31.3s | 613 MB | 99% | - |
| CHIRPS + IMD rainfall metrics | 1.8s | 185 MB | 10% | - |
| Nearby settlements and places (SerpApi) | 3.6s | 187 MB | 6% | - |
| Administrative context (village/tehsil/district) | 8.1s | 556 MB | 96% | - |
| Site summary | 0.0s | 0 MB | 0% | - |
| Report assets | 34.4s | 515 MB | 15% | - |
| Map | 5.8s | 244 MB | 35% | - |

> Memory/CPU are sampled every ~0.25s from outside the process — a stage
> shorter than that (AOI geometry, OSM roads here) can show 0 MB/0% simply
> because no sample landed inside its window, not because it used none.

**External calls made** (this run):

- SerpApi: 8 call(s)
- server.arcgisonline.com: 44 call(s)
- earthengine.googleapis.com: 2 call(s)

**Report assets sub-stage breakdown** (part of the "Report assets" row above):

| Asset | Time |
|---|---|
| ndvi_map | 14.9s |
| fetch_satellite_labels_basemap | 14.1s |
| fetch_street_basemap | 3.3s |
| land_cover_map | 1.4s |
| water_bodies_map | 0.2s |
| nearby_places_map | 0.2s |
| groundwater_map | 0.1s |
| site_context_map | 0.1s |
| groundwater_chart | 0.0s |
| slope_class_chart | 0.0s |
| reference_location_map | 0.0s |

## Generate Report add-on cost

Time: **4.9s**. Peak RSS: 325 MB.

- `generate_report_interpretations`: 2866 prompt + 341 completion = 3207 total tokens (requested max 1400)

## AI chat add-on cost

- "What is the groundwater situation and dominant land cover here?" — 2.1s (tool used: none)
- "Are there any petrol pumps nearby?" — 31.9s (tool used: nearby_category_search)

- `_decide_and_fetch_tool`: 1419 prompt + 81 completion = 1500 total tokens (requested max 300)
- `answer_question`: 2757 prompt + 316 completion = 3073 total tokens (requested max 900)
- `_decide_and_fetch_tool`: 1569 prompt + 70 completion = 1639 total tokens (requested max 300)
- `answer_question`: 3177 prompt + 100 completion = 3277 total tokens (requested max 900)

## Compare add-on cost

Paired with existing run `betma` (no fresh pipeline run needed). Time: **31.3s**. Peak RSS: 278 MB.

- `_decide_and_fetch_tool`: 1589 prompt + 112 completion = 1701 total tokens (requested max 300)
- `rank_sites`: 2425 prompt + 482 completion = 2907 total tokens (requested max 900)

## Total Groq token usage (this profiling run)

The full pipeline run itself makes zero Groq calls — it's entirely
deterministic. All Groq usage comes from the three add-ons above:

| Phase | Groq calls | Total tokens |
|---|---|---|
| Generate Report | 1 | 3,207 |
| AI chat (2 questions) | 4 | 9,489 |
| Compare | 2 | 4,608 |
| **Total** | **7** | **17,304** |

Per-call breakdown:
- Report: `generate_report_interpretations` — 3,207
- Chat Q1 ("groundwater/land cover"): `_decide_and_fetch_tool` 1,500 + `answer_question` 3,073 = 4,573
- Chat Q2 ("petrol pumps", triggered a live tool): `_decide_and_fetch_tool` 1,639 + `answer_question` 3,277 = 4,916
- Compare: `_decide_and_fetch_tool` 1,701 + `rank_sites` 2,907 = 4,608

At Groq's free-tier 200,000 tokens/day limit, that's roughly **11-12**
similar "full report + 2 chat questions + 1 compare" sessions per day
before hitting the daily cap — a real, already-hit constraint this session
(see the rate-limit error reproduced earlier while testing chat) and worth
weighing alongside memory/CPU when planning for concurrent users.

## Cloud Run sizing

Observed peak RSS during the full pipeline run: **728 MB**. With headroom for concurrent requests and GC slack, a reasonable starting point is **1 GB memory**, revisited once a multi-site sweep confirms this isn't the lightest case. CPU is mostly I/O-bound waiting on GEE/Overpass/SerpApi — 1-2 vCPU should be sufficient unless report-asset chart/map rendering (matplotlib) becomes a bottleneck under concurrent load.

**Startup probe / min-instances**: given the observed 902s cold-import anomaly above, do not size Cloud Run's startup probe timeout off the ~3s warm numbers in this report alone — either keep at least one warm instance (`min-instances=1`) so real user requests never hit a truly cold container, or explicitly re-measure this import chain against a fresh container image before setting a startup timeout.
