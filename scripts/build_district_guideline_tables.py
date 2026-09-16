"""Build the final per-district guideline-rate feature tables.

For every `data/2026_guidelines_MP/2026-2027-<District>-en.pdf`:

1. Parse it with `parse_guideline_rate_pdfs.parse_pdf` (already verified on
   Dewas: 2746/2746 rows, 0 warnings).
2. Save the raw parsed rows to
   `cache/guideline_rates/raw/<District>.csv`.
3. Rural rows (`row_kind == "rural_patwari_halka"`) are fuzzy-matched by
   name to a specific village + `vlcode` in `data/vb_soi_mp.GeoJSON`,
   scoped to that district and tehsil, then deduplicated to one row per
   village -- preferring an "andar" (interior, off-road) rate over a
   "road_par" (on-road frontage) rate when a village has more than one
   guideline entry, since a road-frontage rate is a poor stand-in for a
   typical plot.
4. Urban rows (`row_kind == "urban_ward"`) have no ward-boundary shapefile
   to geocode against, so they're summarized instead: one row per
   (district, tehsil) with the min/max of each rate category across all
   its ward entries.
5. Both tables are combined into one sparse CSV -- `row_kind` tells you
   which shape a given row has -- and written to
   `data/2026_guidelines_MP_dfs/<District>.csv`.

Run standalone:
    python -m scripts.build_district_guideline_tables
    python -m scripts.build_district_guideline_tables --district Dewas
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_guideline_rate_pdfs import _RATE_COLUMNS, parse_pdf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_pipeline import config  # noqa: E402

PDF_DIR = config.DATA_DIR / "2026_guidelines_MP"
VB_PATH = config.VILLAGE_BOUNDARY_FILE
RAW_OUT_DIR = config.CACHE_DIR / "guideline_rates" / "raw"
FINAL_OUT_DIR = config.GUIDELINE_RATES_DIR

# The PDF's district name and vb_soi_mp.GeoJSON's `district` field agree
# for 50 of 52 districts after stripping spaces/case; these two use
# genuinely different names for the same district.
DISTRICT_NAME_OVERRIDES = {
    "KHANDWA": "East Nimar",
    "SINGROLI": "Singrauli",
}

DESCRIPTOR_RE = re.compile(
    r"\b(ROAD|MARG|PAR$|ANDAR|ANDER|BYPASS|SEEM|TAK|SE\b|PANCHAYAT|NAGAR PANCHAYAT|MAJARA|MAJRA)\b",
    re.IGNORECASE,
)

_SUFFIX_RE = re.compile(
    r"\s*[/,]?\s*\b(?:"
    r"(?:PRADHANMANTRI|RASHATRIYA|RASHATIYA|RAJAY|RAJYA|RAJKIYA|DISTRICT|NH|S\.?H\.?)\s+"
    r"(?:RAJ\s+)?MARG(?:A)?\b(?:\s+(?:PAR|PER|SE\s+ANDA?R))?"
    r"|PRADHANMANTRI\s+ROAD\b"
    r"|(?:NH|S\.?H\.?)\s+ROAD\b(?:\s+(?:PAR|PER))?"
    r"|ROAD\s*(?:PAR|PER|SE\s+ANDA?R|PR)\b"
    r"|BY\s*-?\s*PASS(?:\s+ROAD)?(?:\s+(?:PAR|PER|SE\s+ANDA?R))?"
    r").*$",
    re.IGNORECASE,
)

# A road/highway/frontage phrase, possibly qualified (PM/CM/PAKKI SADAK,
# RAJAY MARG, N.H. ...), that some districts phrase with the Hindi word
# "Sadak" instead of "Road" -- e.g. Harda's "PM/CM/PAKKI SADAK SE ANDAR" or
# "PRADHANMANTRI/MUKHYAMANTRI/PA KKI SADAK PAR" (the mid-word space in
# "PA KKI" is a PDF line-wrap artifact). Everything from the first such
# qualifier or trigger word onward is stripped.
_QUALIFIER = r"(?:PRADHANMANTRI|MUKHYAMANTRI|RASHATRIYA|RASHATIYA|RAJAY|RAJYA|RAJKIYA|DISTRICT|JILA|PAKKI|PA\s*KKI|PM|CM|NH|S\.?H\.?)"
_SADAK_RE = re.compile(
    r"(?:\s*/?\s*" + _QUALIFIER + r")*\s*\b(SADAK|HIGHWAY)\b.*$",
    re.IGNORECASE,
)
# "N.H. 59A PAR" / "NH - 59 (A) SE ANDAR" -- a highway number stands in for
# the usual ROAD/MARG/SADAK trigger word.
_HIGHWAY_NUMBER_RE = re.compile(
    r"\bN\.?H\.?\s*[-.]?\s*\d+[A-Z]?\s*(?:\(\w+\))?\s*(?:PAR|PER|SE\s+ANDA?R)?\b.*$",
    re.IGNORECASE,
)
# A bare "(1)"/"(2)" road-category tag -- never part of a village name --
# can appear mid-string, not just trailing, e.g. "CHARKHEDA (1) N.H. 59A
# PAR"; the end-anchored multiplier-tag regex in parse_guideline_rate_pdfs
# only strips this shape when it's the very last thing in the string.
_MIDSTRING_NUMERIC_TAG_RE = re.compile(r"\(\s*\d+\s*\)")

# "Gram <village> -<Tehsil/town>" (Burhanpur's convention -- "Gram" is
# Hindi for "village") -- the leading word is never part of the name, and
# the trailing "-<place>" is typically the containing tehsil/town's name,
# not the village's.
_GRAM_PREFIX_RE = re.compile(r"^\s*GRA?A?M\s+", re.IGNORECASE)

_ANDAR_RE = re.compile(r"\b(ANDAR|ANDER)\b", re.IGNORECASE)
_ROAD_PAR_RE = re.compile(
    r"\b(ROAD\s*(PAR|PER|PR)|MARG\s*(PAR|PER)|SADAK\s*(PAR|PER)|BY\s*-?\s*PASS)\b", re.IGNORECASE
)
_FRONTAGE_RANK = {"andar": 0, "plain": 1, "road_par": 2}


def classify_frontage(raw_text: str) -> str:
    """"Andar" (interior) beats "plain" beats "road_par" (frontage)."""

    if _ANDAR_RE.search(raw_text):
        return "andar"
    if _ROAD_PAR_RE.search(raw_text):
        return "road_par"
    return "plain"


def strip_suffix(name: str) -> str:
    stripped = _MIDSTRING_NUMERIC_TAG_RE.sub(" ", name)
    stripped = _HIGHWAY_NUMBER_RE.sub("", stripped)
    stripped = _SADAK_RE.sub("", stripped)
    stripped = _SUFFIX_RE.sub("", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" -/,")
    return stripped if stripped else name


def candidates_for(clean_name: str, tehsil: str | None = None) -> list[str]:
    """Guess which substring of a guideline place name is the village.

    A trailing parenthetical is the real village name for a named colony
    (e.g. "PARAS RESIDENCY (NAGUKEDI)") but is just a road-frontage
    descriptor for others (e.g. "BHMORI (ROAD SE ANDAR)") -- distinguish
    by checking whether that parenthetical itself looks like a descriptor.
    Different districts phrase the non-village part quite differently (a
    mid-string "(1)"/"(2)" road-category tag, a leading "Gram" ["village"]
    prefix, a trailing "-<tehsil name>", Hindi "Sadak" instead of "Road"),
    so on top of the regex-based cleanup, the first word (and first two
    words, for genuinely multi-word village names) are always tried too as
    a cheap, convention-agnostic fallback -- `best_match`'s thresholds and
    prefix-tier check keep a short/common leading word from mismatching.
    """

    text = _GRAM_PREFIX_RE.sub("", clean_name).strip()
    if tehsil:
        tehsil_variants = {tehsil.strip()}
        tehsil_variants.add(re.sub(r"\s*\([^()]*\)\s*", " ", tehsil).strip())
        for variant in tehsil_variants:
            if not variant:
                continue
            stripped = re.sub(r"\s*-\s*" + re.escape(variant) + r"\s*$", "", text, flags=re.IGNORECASE).strip()
            if stripped != text:
                text = stripped
                break

    cands = []
    m = re.search(r"\(([^()]+)\)\s*$", text)
    leading = re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()
    if m:
        trailing = m.group(1).strip()
        if DESCRIPTOR_RE.search(trailing):
            cands.append(leading)
        else:
            cands.append(trailing)
            cands.append(leading)
    else:
        cands.append(text.strip())

    extra = [strip_suffix(c) for c in cands]
    cands.extend(c for c in extra if c not in cands)

    words = re.split(r"\s+", cands[0]) if cands and cands[0] else []
    if words:
        cands.append(words[0])
        if len(words) > 1:
            cands.append(" ".join(words[:2]))

    seen = set()
    result = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def best_match(
    candidates: list[str],
    choices_norm: list[str],
    choices_nospace: list[str],
    threshold: int = 70,
) -> tuple[int, float] | None:
    """Three tiers, checked in order of how much they can be trusted.

    0. Exact match, or the candidate is exactly the leading word(s) of a
       vb village name (many villages share a base name and are
       disambiguated by an appended word, e.g. "DHATURIYA" vs. vb's
       "DHATURIYA TONK"). Plain fuzzy scoring penalizes that trailing word
       so heavily that an unrelated village can outscore the true match --
       this tier exists specifically to avoid that failure mode, and wins
       outright over anything from the two fuzzy passes below.
    1. `token_sort_ratio` on the normal (spaced) names -- word-order/typo
       tolerant.
    2. `fuzz.ratio` on space-stripped names -- recovers pairs like vb's
       "Pipalyabaks" (one word) vs. the guideline's "PIPALYA BAKS" (two
       words), which (1) scores poorly since "one token" vs "two tokens"
       looks structurally different even though the spelling barely
       differs.
    """

    prefix_hits = []
    for c in candidates:
        c_upper = c.upper()
        for i, choice in enumerate(choices_norm):
            if choice == c_upper or choice.startswith(c_upper + " "):
                prefix_hits.append((i, len(choice)))
    if prefix_hits:
        best_idx, _ = min(prefix_hits, key=lambda x: x[1])
        return (best_idx, 99.0)

    best = None
    for c in candidates:
        c_upper = c.upper()
        r1 = process.extractOne(c_upper, choices_norm, scorer=fuzz.token_sort_ratio)
        if r1 and r1[1] >= threshold and (best is None or r1[1] > best[1]):
            best = (r1[2], r1[1])
        r2 = process.extractOne(c_upper.replace(" ", ""), choices_nospace, scorer=fuzz.ratio)
        if r2 and r2[1] >= threshold and (best is None or r2[1] > best[1]):
            best = (r2[2], r2[1])
    return best


def resolve_district_name(pdf_district: str, vb_district_names: set[str]) -> str | None:
    override = DISTRICT_NAME_OVERRIDES.get(pdf_district.upper())
    if override:
        return override
    norm = lambda s: s.replace(" ", "").replace("-", "").upper()
    target = norm(pdf_district)
    for vb_name in vb_district_names:
        if norm(vb_name) == target:
            return vb_name
    return None


def build_tehsil_map(rural_tehsils: list[str], vb_villages: pd.DataFrame) -> dict[str, str]:
    """Map each guideline "tehsil" value to the matching vb subdistrict.

    A guideline PDF's tehsil names and vb's `subdistric`/`block` names for
    the same district usually agree exactly (case aside), with the one
    recurring exception being a "<Tehsil> Nagar" variant used for a
    tehsil's urban-adjacent rural rows. Fuzzy matching absorbs everything
    else (typos, minor spelling drift) since each district only has a
    handful of tehsils to disambiguate among -- collision risk is low.
    """

    subdistric_names = set(vb_villages["subdistric"].dropna().unique())
    block_names = set(vb_villages["block"].dropna().unique())
    all_names = subdistric_names | block_names
    all_names_upper = {n.upper(): n for n in all_names}

    mapping: dict[str, str] = {}
    for tehsil in rural_tehsils:
        key = tehsil.upper().strip()

        # A tehsil name is sometimes given as "OUTER (INNER)" -- e.g.
        # "SINGRAULI (BAIDHAN)" where the parenthetical is vb's actual
        # `block` name -- so try both parts, inner first (usually the more
        # specific/reliable one).
        variants = [key]
        paren = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", key)
        if paren:
            variants = [paren.group(2).strip(), paren.group(1).strip()]

        matched_name = None
        for variant in variants:
            if variant in all_names_upper:
                matched_name = all_names_upper[variant]
                break
            stripped = re.sub(r"\s+NAGAR$", "", variant).strip()
            if stripped in all_names_upper:
                matched_name = all_names_upper[stripped]
                break
            fuzzy = process.extractOne(variant, list(all_names_upper.keys()), scorer=fuzz.token_sort_ratio)
            if fuzzy and fuzzy[1] >= 80:
                matched_name = all_names_upper[fuzzy[0]]
                break
        if matched_name:
            mapping[key] = matched_name
    return mapping


def match_rural_rows(rural: pd.DataFrame, vb_villages: pd.DataFrame) -> pd.DataFrame:
    """One row per rural guideline entry, with its best-match village."""

    tehsil_map = build_tehsil_map(rural["tehsil"].dropna().unique().tolist(), vb_villages)

    vb_villages = vb_villages.copy()
    vb_villages["village_norm"] = vb_villages["village"].str.upper().str.strip()
    vb_villages["village_nospace"] = vb_villages["village_norm"].str.replace(" ", "", regex=False)

    records = []
    for _, row in rural.iterrows():
        tehsil_key = str(row["tehsil"]).upper().strip()
        subdistric_name = tehsil_map.get(tehsil_key)
        used_fallback_scope = False
        if subdistric_name:
            scoped = vb_villages[
                (vb_villages["subdistric"].str.upper() == subdistric_name.upper())
                | (vb_villages["block"].str.upper() == subdistric_name.upper())
            ].reset_index(drop=True)
        else:
            # This tehsil name has no match anywhere in vb's subdistric/block
            # columns for this district -- most often a newly carved-out
            # tehsil the (older) boundary dataset doesn't know about yet, so
            # its villages are still filed under whichever tehsil they
            # belonged to before the split. Falling back to searching the
            # whole district (instead of returning zero matches for every
            # row in this tehsil) trades a small amount of cross-tehsil
            # false-match risk for recovering otherwise-total misses; the
            # match-quality tiers in `best_match` keep that risk in check.
            used_fallback_scope = True
            scoped = vb_villages.reset_index(drop=True)

        cands = candidates_for(row["guideline_place_clean"], tehsil=row["tehsil"])
        match = None
        if len(scoped) > 0:
            match = best_match(cands, scoped["village_norm"].tolist(), scoped["village_nospace"].tolist())

        frontage_type = classify_frontage(row["guideline_place_raw"])
        record = {
            "tehsil": row["tehsil"],
            "frontage_type": frontage_type,
            "used_fallback_scope": used_fallback_scope,
            **{col: row[col] for col in _RATE_COLUMNS},
        }
        if match:
            matched_village = scoped.iloc[match[0]]
            record.update(
                village=matched_village["village"],
                vlcode=matched_village["vlcode"],
                match_score=match[1],
            )
        else:
            record.update(village=None, vlcode=None, match_score=None)
        records.append(record)

    return pd.DataFrame(records)


def dedup_to_village_level(matched: pd.DataFrame) -> pd.DataFrame:
    """One row per matched village, preferring andar > plain > road_par."""

    hit = matched[matched["vlcode"].notna()].copy()
    if hit.empty:
        return hit
    hit["frontage_rank"] = hit["frontage_type"].map(_FRONTAGE_RANK)
    hit = hit.sort_values(["vlcode", "frontage_rank", "match_score"], ascending=[True, True, False])
    village_level = hit.groupby("vlcode", as_index=False).first()
    village_level = village_level.drop(columns=["frontage_rank"])
    village_level.insert(0, "row_kind", "rural_village")
    return village_level


def build_urban_ranges(urban: pd.DataFrame) -> pd.DataFrame:
    """One row per (tehsil), with min/max of each rate category."""

    if urban.empty:
        return pd.DataFrame()

    records = []
    for tehsil, group in urban.groupby("tehsil"):
        record = {
            "row_kind": "urban_range",
            "tehsil": tehsil,
            "n_wards": group["ward"].nunique(),
            "n_rows": len(group),
        }
        for col in _RATE_COLUMNS:
            record[f"{col}_min"] = group[col].min()
            record[f"{col}_max"] = group[col].max()
        records.append(record)
    return pd.DataFrame(records)


def process_district(pdf_path: Path, vb_gdf: gpd.GeoDataFrame, vb_district_names: set[str]) -> dict:
    pdf_district = pdf_path.stem.replace("2026-2027-", "").replace("-en", "")
    vb_district = resolve_district_name(pdf_district, vb_district_names)
    if vb_district is None:
        return {"district": pdf_district, "error": "no matching vb district name found"}

    rows, warnings = parse_pdf(pdf_path)
    if not rows:
        return {"district": pdf_district, "error": "parse_pdf returned 0 rows"}

    field_names = [f.name for f in __import__("dataclasses").fields(rows[0])]
    df = pd.DataFrame([{name: getattr(r, name) for name in field_names} for r in rows])

    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT_DIR / f"{pdf_district}.csv", index=False)

    vb_villages = vb_gdf[vb_gdf["district"] == vb_district][
        ["village", "vlcode", "subdistric", "block", "total_urban_rural\n"]
    ].copy()
    vb_villages = vb_villages[vb_villages["total_urban_rural\n"].str.strip() == "Rural"]

    rural = df[df["row_kind"] == "rural_patwari_halka"].copy()
    urban = df[df["row_kind"] == "urban_ward"].copy()
    unknown_count = int((df["row_kind"] == "unknown").sum())

    rural_matched = match_rural_rows(rural, vb_villages) if len(rural) else pd.DataFrame()
    village_level = dedup_to_village_level(rural_matched) if len(rural_matched) else pd.DataFrame()
    urban_ranges = build_urban_ranges(urban)

    combined = pd.concat([village_level, urban_ranges], ignore_index=True)
    combined.insert(0, "district", pdf_district)

    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(FINAL_OUT_DIR / f"{pdf_district}.csv", index=False)

    n_rural_matched = int(rural_matched["vlcode"].notna().sum()) if len(rural_matched) else 0
    return {
        "district": pdf_district,
        "total_rows": len(df),
        "parse_warnings": len(warnings),
        "unknown_row_kind": unknown_count,
        "rural_rows": len(rural),
        "rural_matched_rows": n_rural_matched,
        "rural_match_rate": round(100 * n_rural_matched / len(rural), 1) if len(rural) else None,
        "rural_villages_out": len(village_level),
        "urban_rows": len(urban),
        "urban_tehsils_out": len(urban_ranges),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--district", type=str, default=None, help="Only process this one district (PDF stem name)")
    args = parser.parse_args()

    vb_gdf = gpd.read_file(VB_PATH)
    vb_district_names = set(vb_gdf["district"].unique())

    pdf_paths = sorted(PDF_DIR.glob("2026-2027-*-en.pdf"))
    if args.district:
        pdf_paths = [p for p in pdf_paths if args.district.upper() in p.stem.upper()]

    summaries = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        try:
            summary = process_district(pdf_path, vb_gdf, vb_district_names)
        except Exception as e:
            summary = {"district": pdf_path.stem, "error": f"{type(e).__name__}: {e}"}
        summaries.append(summary)
        print(f"[{i}/{len(pdf_paths)}] {summary}")

    summary_df = pd.DataFrame(summaries)
    summary_path = FINAL_OUT_DIR / "_build_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    if "error" in summary_df.columns:
        errors = summary_df[summary_df["error"].notna()]
        if len(errors):
            print(f"\n{len(errors)} district(s) failed:")
            print(errors[["district", "error"]].to_string())


if __name__ == "__main__":
    main()
