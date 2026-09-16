"""Village / tehsil / district context + Census join.

Ports the village-boundary portion of notebook cell 27 (village layer +
Census join) and cell 55 (administrative-location assembly: host village,
AOI overlap, fallbacks).
"""

from __future__ import annotations

import functools
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from ._util import clean_text
from .aoi import AOI, read_local_vector

VILLAGE_COLUMNS = [
    "village", "vlcode", "subdistric", "block", "district", "state",
    "dtcode", "sdcode", "total_urban_rural\n",
    "total_population_village\n", "total_geographical_area\n",
    "forest_area\n", "area_under_non_agricultural_use\n",
    "barren_uncultivable_land\n", "permanent_pastures_grazing\n",
    "land_under_miscellaneous\n", "culturable_waste_land\n",
    "fallows_land_other_than_current\n", "current_fallows_area\n",
    "net_area_sown\n", "total_unirrigated_land\n",
    "area_irrigated_by_source\n",
]

# source column (with the source file's stray trailing "\n") -> clean key
VILLAGE_FIELDS = {
    "village": "village",
    "subdistric": "tehsil",
    "block": "block",
    "district": "district",
    "state": "state",
    "vlcode": "boundary_village_code",
}
LAND_USE_FIELDS = {
    "total_population_village\n": "population",
    "total_geographical_area\n": "area_ha",
    "forest_area\n": "forest_area_ha",
    "area_under_non_agricultural_use\n": "non_agricultural_area_ha",
    "barren_uncultivable_land\n": "barren_land_ha",
    "permanent_pastures_grazing\n": "pastures_grazing_ha",
    "land_under_miscellaneous\n": "miscellaneous_land_ha",
    "culturable_waste_land\n": "culturable_waste_ha",
    "fallows_land_other_than_current\n": "other_fallows_ha",
    "current_fallows_area\n": "current_fallows_ha",
    "net_area_sown\n": "net_area_sown_ha",
    "total_unirrigated_land\n": "unirrigated_land_ha",
    "area_irrigated_by_source\n": "irrigated_land_ha",
}
CENSUS_FEATURE_COLUMNS = [
    "population_total", "households", "cultivation_share",
    "agricultural_labour_share", "household_industry_share",
    "other_worker_share", "literacy_percent", "sc_percent", "st_percent",
]

# The "headline" rate categories from the MP Govt guideline-rate tables
# (see scripts/build_district_guideline_tables.py) -- building
# construction rates and the two agri-plot sub-clause rates are left out
# as too granular/obscure for a general-purpose land-price signal.
GUIDELINE_RATE_COLUMNS = [
    "plot_residential_sqm", "plot_commercial_sqm", "plot_industrial_sqm",
    "agri_land_irrigated_per_ha", "agri_land_unirrigated_per_ha",
]

# vb_soi_mp.GeoJSON's `district` field and the guideline CSVs' filenames
# (named after the source PDF) disagree for these two districts only;
# everything else matches after stripping spaces/case (e.g. vb's
# "Ashoknagar" vs. the file "Ashok Nagar.csv").
_VB_TO_GUIDELINE_DISTRICT_OVERRIDES = {
    "EAST NIMAR": "Khandwa",
    "SINGRAULI": "Singroli",
}


@functools.lru_cache(maxsize=1)
def _guideline_csv_paths() -> dict[str, Path]:
    """{normalized district name -> CSV path}, built once from whatever
    per-district guideline-rate files actually exist on disk."""

    if not config.GUIDELINE_RATES_DIR.exists():
        return {}
    return {
        re.sub(r"[^A-Z0-9]", "", path.stem.upper()): path
        for path in config.GUIDELINE_RATES_DIR.glob("*.csv")
        if not path.stem.startswith("_")
    }


@functools.lru_cache(maxsize=8)
def _load_guideline_table(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _district_guideline_table(district: str | None) -> pd.DataFrame | None:
    """Resolve ``district`` (vb_soi_mp naming) to its guideline-rate table,
    or ``None`` if no such file exists."""

    if not district:
        return None
    override = _VB_TO_GUIDELINE_DISTRICT_OVERRIDES.get(district.upper())
    key = re.sub(r"[^A-Z0-9]", "", (override or district).upper())
    path = _guideline_csv_paths().get(key)
    return _load_guideline_table(str(path)) if path is not None else None


def _match_urban_range_row(table: pd.DataFrame, tehsil: str) -> pd.Series | None:
    """The best ``row_kind == "urban_range"`` row for ``tehsil`` in
    ``table``: an exact (case-insensitive) match, else a tehsil whose name
    starts with it (covers the "<Tehsil> Nagar" naming variant used for a
    tehsil's urban-adjacent rows) -- same tehsil-resolution heuristic
    :mod:`scripts.build_district_guideline_tables` itself relies on."""

    urban = table[table["row_kind"] == "urban_range"]
    tehsil_upper = tehsil.upper().strip()
    exact = urban[urban["tehsil"].str.upper().str.strip() == tehsil_upper]
    candidate = exact if not exact.empty else urban[urban["tehsil"].str.upper().str.strip().str.startswith(tehsil_upper)]
    return candidate.iloc[0] if not candidate.empty else None


def _lookup_govt_guideline_rates(
    district: str | None, vlcode: str | None, tehsil: str | None
) -> tuple[dict | None, str | None]:
    """Look up this site's MP Govt guideline (circle) rates: a specific
    village's rate if the host village matched one in the source PDF, else
    a (district, tehsil) urban rate range if the host village is itself
    urban (no ward-boundary shapefile exists to go any finer than that).

    Args:
        district: ``administration["district"]`` (vb_soi_mp naming).
        vlcode: ``administration["boundary_village_code"]``.
        tehsil: ``administration["tehsil"]``.

    Returns:
        ``(govt_guideline_rates, warning)`` -- the dict is ``None`` (with a
        warning) if no district file exists or no row matched.
    """

    if not district:
        return None, None

    table = _district_guideline_table(district)
    if table is None:
        return None, f"No guideline-rate table found for district {district!r}"

    if vlcode is not None:
        try:
            vlcode_int = int(float(vlcode))
        except (TypeError, ValueError):
            vlcode_int = None
        if vlcode_int is not None:
            rural = table[(table["row_kind"] == "rural_village") & (table["vlcode"] == vlcode_int)]
            if not rural.empty:
                row = rural.iloc[0]
                return {
                    "row_kind": "rural_village",
                    "village": clean_text(row.get("village")),
                    "vlcode": vlcode_int,
                    "tehsil": clean_text(row.get("tehsil")),
                    "frontage_type": clean_text(row.get("frontage_type")),
                    "match_score": _to_number(row.get("match_score")),
                    **{col: _to_number(row.get(col)) for col in GUIDELINE_RATE_COLUMNS},
                    "source": "MP Govt district guideline rates, FY2026-27",
                }, None

    if tehsil:
        row = _match_urban_range_row(table, tehsil)
        if row is not None:
            rates = {}
            for col in GUIDELINE_RATE_COLUMNS:
                rates[f"{col}_min"] = _to_number(row.get(f"{col}_min"))
                rates[f"{col}_max"] = _to_number(row.get(f"{col}_max"))
            return {
                "row_kind": "urban_range",
                "tehsil": clean_text(row.get("tehsil")),
                "n_wards": _to_number(row.get("n_wards")),
                "n_rows": _to_number(row.get("n_rows")),
                **rates,
                "source": "MP Govt district guideline rates, FY2026-27",
            }, None

    return None, f"No matching village/tehsil row in {district!r}'s guideline-rate table"


# Kept intentionally small -- this is a map-hover tooltip, not the full
# admin-context lookup's detail panel.
_VILLAGE_TOOLTIP_RATE_COLUMNS = [
    "plot_residential_sqm", "agri_land_irrigated_per_ha", "agri_land_unirrigated_per_ha",
]


def _attach_village_guideline_rates(villages_gdf: pd.DataFrame) -> pd.DataFrame:
    """Add a guideline-rate column per village (for the "Village
    boundaries" map-layer tooltip) -- unlike
    :func:`_lookup_govt_guideline_rates` (one lookup for the single host
    village), this enriches every village in ``villages_gdf`` so hovering
    any of them shows its own rate, not just the site's.

    An AOI centered inside a town/city typically has few or even zero
    *rural* villages in it -- the site's own "village" there is really the
    single polygon covering the whole urban area (see
    ``total_urban_rural`` -- e.g. all of Nagar Nigam Dewas is one such
    polygon), which has no per-village guideline row of its own. Urban
    villages are handled separately: they show that (district, tehsil)'s
    urban rate RANGE instead (formatted as "lo-hi", the same fallback
    :func:`_lookup_govt_guideline_rates` uses for an urban host village),
    rather than being left blank the way a genuinely unmatched rural
    village is.

    Args:
        villages_gdf: A village GeoDataFrame with ``district``/``vlcode``/
            ``total_urban_rural\\n`` columns (e.g. ``villages_in_aoi_gdf``).
            Mutated in place and returned.
    """

    for col in _VILLAGE_TOOLTIP_RATE_COLUMNS:
        villages_gdf[f"guideline_{col}"] = None

    is_urban = (
        villages_gdf.get("total_urban_rural\n", pd.Series("", index=villages_gdf.index))
        .astype("string").fillna("").str.strip().str.upper() == "URBAN"
    )

    # --- Rural villages: a direct per-village vlcode match -----------
    for district, group in villages_gdf.loc[~is_urban].groupby("district"):
        table = _district_guideline_table(district)
        if table is None:
            continue

        rural = table.loc[table["row_kind"] == "rural_village", ["vlcode"] + _VILLAGE_TOOLTIP_RATE_COLUMNS].copy()
        rural = rural.dropna(subset=["vlcode"]).drop_duplicates(subset=["vlcode"])
        rural["vlcode"] = rural["vlcode"].astype(int)

        group_vlcode = pd.to_numeric(group["vlcode"], errors="coerce")
        merged = group_vlcode.to_frame("vlcode").merge(rural, on="vlcode", how="left")
        for col in _VILLAGE_TOOLTIP_RATE_COLUMNS:
            values = merged[col].to_numpy()
            villages_gdf.loc[group.index, f"guideline_{col}"] = pd.array(values, dtype="object")

    # --- Urban villages: that tehsil's rate range, as a "lo-hi" string ---
    for idx in villages_gdf.index[is_urban]:
        district, tehsil = villages_gdf.at[idx, "district"], villages_gdf.at[idx, "subdistric"]
        table = _district_guideline_table(district) if district else None
        row = _match_urban_range_row(table, tehsil) if table is not None and tehsil else None
        if row is None:
            continue
        for col in _VILLAGE_TOOLTIP_RATE_COLUMNS:
            lo, hi = _to_number(row.get(f"{col}_min")), _to_number(row.get(f"{col}_max"))
            if lo is not None and hi is not None:
                villages_gdf.at[idx, f"guideline_{col}"] = f"{lo:,.0f}–{hi:,.0f} (tehsil range)"

    # NaN isn't valid JSON (villages_in_aoi_gdf is serialized via .to_json()
    # for the map's GeoJsonTooltip) -- use None so an unmatched village
    # shows a blank tooltip field instead of a literal "NaN"/serialization
    # error.
    for col in _VILLAGE_TOOLTIP_RATE_COLUMNS:
        target = f"guideline_{col}"
        villages_gdf[target] = villages_gdf[target].where(villages_gdf[target].notna(), None)

    return villages_gdf


def _to_number(value):
    """Coerce ``value`` to a float, or None if it is null/non-numeric."""

    number = pd.to_numeric(clean_text(value), errors="coerce")
    return None if pd.isna(number) else float(number)


def _normalize_join_name(values: pd.Series) -> pd.Series:
    """Uppercase and strip non-alphanumeric characters, for a robust
    village-name join key between the boundary and Census files."""

    return values.astype("string").fillna("").str.upper().str.replace(r"[^A-Z0-9]+", "", regex=True)


def get_admin_context(aoi: AOI) -> dict:
    """Resolve the host village (containing, or nearest to, the site),
    every village/tehsil/district overlapping the AOI, and the Census join,
    from Survey-of-India village boundaries.

    Args:
        aoi: The area of interest, from :func:`data_analysis_pipeline.aoi.build_aoi`.

    Returns:
        A standard envelope dict. ``observations`` = {"villages_in_aoi_gdf":
        GeoDataFrame, "summary": {"administration": {village, tehsil, block,
        district, state, boundary_village_code, point_inside_village_polygon,
        govt_guideline_rates, source}, "land_use": {population, area_ha,
        forest_area_ha, ..., forest/barren/net_sown_share_percent (0-100),
        source}, "census": {population_total, households, literacy_percent,
        sc_percent, st_percent, ...}, "census_join": {method,
        boundary_villages, matched_villages, unmatched_villages, source},
        "villages_in_aoi": [name, ...], "tehsils_in_aoi": [...],
        "districts_in_aoi": [...]}}.

        ``administration["govt_guideline_rates"]`` is the MP Govt district
        guideline (circle) rate for the host village -- a specific
        village's rate (row_kind "rural_village") if it matched one in the
        source PDF, else a (district, tehsil) rate range (row_kind
        "urban_range") if the host village is itself urban (no
        ward-boundary shapefile exists to resolve any finer than that);
        ``None`` if no district guideline-rate table or matching row was
        found (see :func:`_lookup_govt_guideline_rates`).
    """

    warnings: list[str] = []
    villages = read_local_vector(config.VILLAGE_BOUNDARY_FILE, VILLAGE_COLUMNS, aoi)

    if villages.empty:
        warnings.append(f"Village boundary file not found or empty in AOI: {config.VILLAGE_BOUNDARY_FILE}")
        administration = {key: None for key in VILLAGE_FIELDS.values()}
        administration.update({
            "boundary_village_code": None,
            "point_inside_village_polygon": None,
            "govt_guideline_rates": None,
            "source": "vb_soi_mp.GeoJSON (not available)",
            "land_use": None,
            "census": None,
        })
        return {
            "dataset": "admin_context",
            "source": "Survey of India village boundaries (vb_soi_mp) + census_derieved.csv",
            "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
            "observations": {
                "villages_in_aoi_gdf": villages,
                "summary": {
                    "administration": administration,
                    "villages_in_aoi": [], "tehsils_in_aoi": [], "districts_in_aoi": [],
                    "census_join": {
                        "method": "not available", "boundary_villages": 0,
                        "matched_villages": 0, "unmatched_villages": 0,
                        "source": "vb_soi_mp.GeoJSON not found",
                    },
                },
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": None,
            "confidence": "low",
            "warnings": warnings,
            "limitations": ["MP-only source."],
        }

    # --- Census join (by normalized village name) -----------------------
    if config.CENSUS_FEATURES_FILE.exists():
        census_features = pd.read_csv(
            config.CENSUS_FEATURES_FILE,
            usecols=["State", "District", "Subdistt", "Name", "village_code"] + CENSUS_FEATURE_COLUMNS,
            dtype={"State": "string", "District": "string", "Subdistt": "string", "Name": "string", "village_code": "string"},
        )
        census_features["census_name_key"] = _normalize_join_name(census_features["Name"])
        census_lookup = census_features.rename(
            columns={"District": "census_district_code", "Subdistt": "census_subdistrict_code"}
        )[["census_district_code", "census_subdistrict_code", "census_name_key"] + CENSUS_FEATURE_COLUMNS]
        census_lookup = census_lookup.drop_duplicates(
            subset=["census_district_code", "census_subdistrict_code", "census_name_key"]
        )

        villages["boundary_name_key"] = _normalize_join_name(villages["village"])
        villages = villages.merge(
            census_lookup.drop_duplicates(subset=["census_name_key"]),
            left_on="boundary_name_key", right_on="census_name_key", how="left",
        )
        villages["census_match"] = villages[CENSUS_FEATURE_COLUMNS[0]].notna()
        census_join_summary = {
            "method": "normalized village name",
            "boundary_villages": int(len(villages)),
            "matched_villages": int(villages["census_match"].sum()),
            "unmatched_villages": int((~villages["census_match"]).sum()),
            "source": "census_derieved.csv",
        }
    else:
        warnings.append(f"Census features file not found: {config.CENSUS_FEATURES_FILE}")
        census_join_summary = {
            "method": "not available", "boundary_villages": int(len(villages)),
            "matched_villages": 0, "unmatched_villages": int(len(villages)),
            "source": "census_derieved.csv not found",
        }
        villages["census_match"] = False
        for column in CENSUS_FEATURE_COLUMNS:
            villages[column] = pd.NA

    # --- Host village (containment, fallback to nearest) -----------------
    site_point = aoi.center_4326
    containing = villages[villages.contains(site_point)]

    if not containing.empty:
        host = containing.iloc[0]
        point_inside = True
    else:
        villages_utm = villages.to_crs(aoi.utm_crs)
        idx = villages_utm.geometry.distance(aoi.center_utm).idxmin()
        host = villages.loc[idx]
        point_inside = False

    administration = {key: clean_text(host.get(column)) for column, key in VILLAGE_FIELDS.items()}
    administration["point_inside_village_polygon"] = point_inside
    administration["source"] = "Survey of India village boundaries (vb_soi_mp)"

    govt_guideline_rates, guideline_warning = _lookup_govt_guideline_rates(
        administration.get("district"), administration.get("boundary_village_code"), administration.get("tehsil"),
    )
    administration["govt_guideline_rates"] = govt_guideline_rates
    if guideline_warning:
        warnings.append(guideline_warning)

    land_use_raw = {key: _to_number(host.get(column)) for column, key in LAND_USE_FIELDS.items()}
    total_area = land_use_raw.get("area_ha")

    def share(part_key):
        part = land_use_raw.get(part_key)
        return (part / total_area * 100) if total_area not in (None, 0) and part is not None else None

    land_use = {
        **land_use_raw,
        "forest_share_percent": share("forest_area_ha"),
        "barren_share_percent": share("barren_land_ha"),
        "net_sown_share_percent": share("net_area_sown_ha"),
        "source": "Survey of India village boundaries (vb_soi_mp)",
    }

    census = {key: _to_number(host.get(key)) for key in CENSUS_FEATURE_COLUMNS}

    # --- AOI overlap (villages/tehsils/districts intersecting the AOI) ---
    villages_utm = villages.to_crs(aoi.utm_crs)
    villages_in_aoi_gdf = villages.loc[villages_utm.geometry.intersects(aoi.polygon_utm)].copy()
    villages_in_aoi_gdf = _attach_village_guideline_rates(villages_in_aoi_gdf)

    def unique_sorted(series):
        return sorted({name for name in (clean_text(v) for v in series) if name})

    villages_in_aoi = unique_sorted(villages_in_aoi_gdf["village"])
    tehsils_in_aoi = unique_sorted(villages_in_aoi_gdf["subdistric"])
    districts_in_aoi = unique_sorted(villages_in_aoi_gdf["district"])

    summary = {
        "administration": administration,
        "land_use": land_use,
        "census": census,
        "census_join": census_join_summary,
        "villages_in_aoi": villages_in_aoi,
        "tehsils_in_aoi": tehsils_in_aoi,
        "districts_in_aoi": districts_in_aoi,
    }

    return {
        "dataset": "admin_context",
        "source": "Survey of India village boundaries (vb_soi_mp) + census_derieved.csv",
        "params": {"lat": aoi.lat, "lon": aoi.lon, "radius_km": aoi.radius_km},
        "observations": {"villages_in_aoi_gdf": villages_in_aoi_gdf, "summary": summary},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resolution": None,
        "confidence": "high" if point_inside else "medium",
        "warnings": warnings,
        "limitations": ["MP-only source.", "Census join is by normalized village name; a small share of villages do not match."],
    }
