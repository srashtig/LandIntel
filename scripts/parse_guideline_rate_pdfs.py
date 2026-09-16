"""Extract MP Government district land-guideline-rate PDFs (Collector
guideline / circle rate tables, format year 2026-2027) into one
structured table.

Each PDF (`data/2026_guidelines_MP/2026-2027-<District>-en.pdf`) is a
repeating 18-column table of "Guideline Place" rows, grouped under a
context header line that appears once per section and holds until the
next one:

    Urban Local Body : X, Sub-Area : Y, Ward : Z, Tehsil : W   (urban)
    Tehsil : X, Sub-Area : NON-PLANNING AREA, Patwari Halka : NNNNN  (rural)
    Tehsil : X, Sub-Area : PLANNING AREA, Patwari Halka : NNNNN      (peri-urban, still village-indexed)

Row extraction uses PyMuPDF's `page.find_tables()` (reliable for the
16 numeric columns + S.No. + place name). The context header is a
separate, isolated text block that appears exactly when the page
reprints the 3-row column header (i.e. exactly at each section
boundary) — so "does this page have a 3-row header table" doubles as
"does a new context line need to be read on this page."

Run standalone:
    python -m scripts.parse_guideline_rate_pdfs --out-dir cache/guideline_rates

Or import `parse_pdf(path)` / `parse_all(pdf_dir, out_dir)` directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from pathlib import Path

import fitz  # PyMuPDF

_CONTEXT_LINE_RE = re.compile(r"Sub-Area\s*:")
_KV_RE = re.compile(r"([A-Za-z][A-Za-z \-]*?)\s*:\s*(.+)")

# Column order matches the PDF's own (1)-(18) numbering.
_RATE_COLUMNS = [
    "plot_residential_sqm", "plot_commercial_sqm", "plot_industrial_sqm",
    "bldg_res_rcc", "bldg_res_rbc", "bldg_res_tinshade", "bldg_res_kacchakabelu",
    "bldg_comm_shop", "bldg_comm_office", "bldg_comm_godown",
    "bldg_multi_res", "bldg_multi_comm",
    "agri_land_irrigated_per_ha", "agri_land_unirrigated_per_ha",
    "agri_plot_subclause1_sqm", "agri_plot_subclause2_sqm",
]

_MULTIPLIER_RE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*$")
_VISHISHT_GRAM_RE = re.compile(r"\*\s*(?:\(\s*\d+(?:\.\d+)?\s*\))?\s*$")
_TRAILING_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")


@dataclass
class GuidelineRow:
    source_district: str
    source_page: int
    urban_local_body: str | None
    tehsil: str | None
    sub_area: str | None
    ward: str | None
    patwari_halka: str | None
    row_kind: str  # "urban_ward" | "rural_patwari_halka" | "unknown"
    s_no: int
    guideline_place_raw: str
    guideline_place_clean: str
    village_candidate: str | None
    is_vishisht_gram: bool
    multiplier_tag: float | None
    plot_residential_sqm: float | None = None
    plot_commercial_sqm: float | None = None
    plot_industrial_sqm: float | None = None
    bldg_res_rcc: float | None = None
    bldg_res_rbc: float | None = None
    bldg_res_tinshade: float | None = None
    bldg_res_kacchakabelu: float | None = None
    bldg_comm_shop: float | None = None
    bldg_comm_office: float | None = None
    bldg_comm_godown: float | None = None
    bldg_multi_res: float | None = None
    bldg_multi_comm: float | None = None
    agri_land_irrigated_per_ha: float | None = None
    agri_land_unirrigated_per_ha: float | None = None
    agri_plot_subclause1_sqm: float | None = None
    agri_plot_subclause2_sqm: float | None = None
    parse_warning: str | None = None


def _parse_context_line(text: str) -> dict[str, str]:
    """Parse a "Key : Value, Key : Value, ..." context header line.

    Args:
        text: The isolated context-line block text, e.g.
            "Urban Local Body : DEWAS, Sub-Area : NAGAR NIGAM DEWAS, "
            "Ward : WARD NUMBER-01, Tehsil : Dewas nagar".

    Returns:
        A dict keyed by lowercase, underscore-joined field name (e.g.
        "urban_local_body", "sub_area", "ward", "tehsil",
        "patwari_halka"), stripped values.
    """

    result: dict[str, str] = {}
    for segment in text.replace("\n", " ").split(","):
        match = _KV_RE.match(segment.strip())
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        result[key] = match.group(2).strip()
    return result


def _clean_number(raw: str | None) -> tuple[float | None, str | None]:
    """Parse one rate-table cell into a float.

    Args:
        raw: The raw cell string, e.g. "7,15,00,000" (Indian digit
            grouping), "1,00,58,40,00\\n0" (a single number whose
            trailing digits wrapped onto their own line — newlines
            within a cell are always mid-number line-wraps, never a
            separator between two distinct values, so they're removed
            before splitting), or "13,00,00,000 13,00,00,000"
            (occasionally two genuinely distinct space-separated
            values in one merged cell).

    Returns:
        `(value, warning)` — `value` is the first parseable number
        (commas stripped, newlines rejoined), or `None` if the cell is
        blank/unparseable. `warning` is set when more than one
        space-separated number remains after rejoining line-wraps, so
        the caller can flag the row rather than silently dropping data.
    """

    if raw is None:
        return None, None
    text = raw.strip()
    if not text:
        return None, None
    tokens = text.replace("\n", "").split()
    try:
        value = float(tokens[0].replace(",", ""))
    except ValueError:
        return None, f"unparseable cell {raw!r}"
    if len(tokens) > 1:
        return value, f"multiple values in one cell, kept first: {raw!r}"
    return value, None


def _clean_place_name(raw: str) -> tuple[str, str | None, bool, float | None]:
    """Split a raw "Guideline Place" cell into its parts.

    Args:
        raw: The cell text, e.g. "DEWAS LIFE STYLE PART-2 (KARANAKHEDI)
            * (1.2)" or plain "AVANTI NAGAR (BRAHMANKHEDA)".

    Returns:
        `(clean_name, village_candidate, is_vishisht_gram,
        multiplier_tag)` — `clean_name` has the trailing "*"/"(N.N)"
        tags removed; `village_candidate` is the last parenthesised
        segment (often, not always, an actual village name — the
        caller's join step must still fuzzy-match it, this is only a
        head start); `multiplier_tag` is the trailing "(N.N)" factor,
        undocumented here beyond the source PDF's own "* - Vishist
        Gram" footnote — treat it as informational until a legend page
        is available, not as a rate multiplier to apply automatically.
    """

    text = " ".join(raw.split())
    is_vishisht_gram = bool(_VISHISHT_GRAM_RE.search(text))

    multiplier_tag = None
    match = _MULTIPLIER_RE.search(text)
    if match:
        multiplier_tag = float(match.group(1))
        text = text[: match.start()].strip()
    text = text.rstrip("*").strip()

    village_candidate = None
    match = _TRAILING_PAREN_RE.search(text)
    if match:
        village_candidate = match.group(1).strip()

    return text, village_candidate, is_vishisht_gram, multiplier_tag


def parse_pdf(pdf_path: Path) -> tuple[list[GuidelineRow], list[str]]:
    """Extract every guideline-rate row from one district PDF.

    Args:
        pdf_path: Path to a `2026-2027-<District>-en.pdf` file.

    Returns:
        `(rows, warnings)` — `rows` is one `GuidelineRow` per data row
        found (S.No. + place name + up to 16 rate values); `warnings`
        is a list of page-level notes (e.g. a context line that didn't
        parse, a page with no data table) for manual spot-checking.
    """

    district = pdf_path.stem.replace("2026-2027-", "").replace("-en", "")
    doc = fitz.open(pdf_path)
    rows: list[GuidelineRow] = []
    warnings: list[str] = []
    current_context: dict[str, str] = {}

    for page_idx, page in enumerate(doc):
        tabs = page.find_tables()
        if not tabs.tables:
            warnings.append(f"page {page_idx}: no tables detected")
            continue

        # A page can contain more than one context switch (e.g. a short
        # ward ends and the next begins on the same page) — matching only
        # "the first" context line silently mis-attributes every row
        # after the second switch. Instead, every context line and every
        # data row is located by its character offset in the page's full
        # text stream (blocks joined in their natural reading order), and
        # a row's context is whichever switch's offset is the latest one
        # still <= that row's own offset.
        blocks = page.get_text("blocks")
        concat = "\n".join(b[4] for b in blocks)

        context_events: list[tuple[int, dict[str, str]]] = []
        for b in blocks:
            if _CONTEXT_LINE_RE.search(b[4]):
                offset = concat.find(b[4])
                if offset != -1:
                    context_events.append((offset, _parse_context_line(b[4])))
        context_events.sort(key=lambda e: e[0])

        search_cursor = 0

        for table in tabs.tables:
            extracted = table.extract()
            for raw_row in extracted:
                if not raw_row or raw_row[0] is None:
                    continue
                s_no_text = str(raw_row[0]).strip()
                if not s_no_text.isdigit():
                    continue  # header/spacer row, not a data row
                if raw_row[1] is None:
                    continue

                # Locate this row's position in the text stream so it can
                # be compared against context_events' offsets. The name's
                # first word is included so the bare number "168" doesn't
                # match some unrelated "168" inside a rate value instead.
                name_first_word = str(raw_row[1]).split()[0] if str(raw_row[1]).split() else ""
                row_pattern = re.compile(re.escape(s_no_text) + r"\s+" + re.escape(name_first_word))
                match = row_pattern.search(concat, search_cursor)
                row_offset = match.start() if match else None
                if row_offset is not None:
                    search_cursor = row_offset
                else:
                    warnings.append(
                        f"page {page_idx}: could not locate row {s_no_text} in text "
                        "stream for context resolution; using last-known context"
                    )

                if row_offset is not None:
                    for offset, ctx in context_events:
                        if offset <= row_offset:
                            current_context = ctx
                        else:
                            break

                ward = current_context.get("ward")
                patwari_halka = current_context.get("patwari_halka")
                if patwari_halka:
                    row_kind = "rural_patwari_halka"
                elif ward:
                    row_kind = "urban_ward"
                else:
                    row_kind = "unknown"

                clean_name, village_candidate, is_vishisht_gram, multiplier_tag = (
                    _clean_place_name(str(raw_row[1]))
                )

                row_warnings: list[str] = []
                rate_values: dict[str, float | None] = {}
                for i, col_name in enumerate(_RATE_COLUMNS):
                    cell = raw_row[2 + i] if 2 + i < len(raw_row) else None
                    value, warn = _clean_number(cell)
                    rate_values[col_name] = value
                    if warn:
                        row_warnings.append(f"{col_name}: {warn}")

                rows.append(
                    GuidelineRow(
                        source_district=district,
                        source_page=page_idx,
                        urban_local_body=current_context.get("urban_local_body"),
                        tehsil=current_context.get("tehsil"),
                        sub_area=current_context.get("sub_area"),
                        ward=ward,
                        patwari_halka=patwari_halka,
                        row_kind=row_kind,
                        s_no=int(s_no_text),
                        guideline_place_raw=" ".join(str(raw_row[1]).split()),
                        guideline_place_clean=clean_name,
                        village_candidate=village_candidate,
                        is_vishisht_gram=is_vishisht_gram,
                        multiplier_tag=multiplier_tag,
                        parse_warning="; ".join(row_warnings) or None,
                        **rate_values,
                    )
                )

    doc.close()
    return rows, warnings


def parse_all(pdf_dir: Path, out_dir: Path) -> None:
    """Parse every district PDF in `pdf_dir` and write combined output.

    Args:
        pdf_dir: Directory of `2026-2027-<District>-en.pdf` files.
        out_dir: Directory to write `guideline_rates.parquet` (one row
            per guideline place, all districts combined) and
            `guideline_rates_warnings.csv` (per-district parse notes)
            into. Created if missing.
    """

    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[GuidelineRow] = []
    all_warnings: list[dict] = []

    pdf_paths = sorted(pdf_dir.glob("2026-2027-*-en.pdf"))
    for i, pdf_path in enumerate(pdf_paths, 1):
        rows, warnings = parse_pdf(pdf_path)
        all_rows.extend(rows)
        for w in warnings:
            all_warnings.append({"district": pdf_path.stem, "warning": w})
        print(f"[{i}/{len(pdf_paths)}] {pdf_path.name}: {len(rows)} rows, {len(warnings)} warnings")

    field_names = [f.name for f in fields(GuidelineRow)]
    df = pd.DataFrame([{name: getattr(r, name) for name in field_names} for r in all_rows])
    df.to_parquet(out_dir / "guideline_rates.parquet", index=False)
    pd.DataFrame(all_warnings).to_csv(out_dir / "guideline_rates_warnings.csv", index=False)

    print(f"\nTotal: {len(df)} rows across {len(pdf_paths)} districts")
    print(df["row_kind"].value_counts())
    print(f"Rows with a parse_warning: {df['parse_warning'].notna().sum()}")
    print(f"Wrote {out_dir / 'guideline_rates.parquet'}")


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_analysis_pipeline import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=config.DATA_DIR / "2026_guidelines_MP",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.CACHE_DIR / "guideline_rates",
    )
    args = parser.parse_args()
    parse_all(args.pdf_dir, args.out_dir)
