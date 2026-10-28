"""Phase 2 ETL: clean the raw Realtor.com listing data into a modelling dataset.

Reads  data/raw/realtor-data.zip.csv.zip   (never modified; SHA-256 recorded before and after)
Writes data/processed/sold_clean.parquet   primary modelling dataset (status == "sold")
       data/processed/quarantine.parquet   suspicious rows held for review, with reasons
       reports/cleaning_*.csv|json|md      row counts, exclusions, missingness, outliers, coverage

Run from the project root:  python src/data_cleaning.py
The pipeline is deterministic and overwrites its own outputs, so reruns never duplicate records.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "realtor-data.zip.csv.zip"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# ---------------------------------------------------------------------------
# Validity thresholds (documented in reports/cleaning_report.md)
# ---------------------------------------------------------------------------
MIN_PRICE = 1_000                 # below this a "sold" price is more likely a data error than a market sale
INT32_SENTINEL = 2_147_483_647    # 2**31 - 1, an overflow / placeholder value
MAX_PRICE_REVIEW = 100_000_000    # above this, held for manual review (plausible mega-estates stay below)
MAX_BED = 20
MAX_BATH = 20
MIN_HOUSE_SQFT = 200
MAX_HOUSE_SQFT = 50_000
MAX_ACRES = 1_000                 # land-dominated sales are outside a house-price model's scope
MIN_PPSF, MAX_PPSF = 5, 10_000    # price per sqft plausibility band
ZIP_STATE_MIN_ROWS = 5            # zip->state consistency check needs this many rows ...
ZIP_STATE_MIN_SHARE = 0.90        # ... and this share agreeing on one state

VALID_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "Puerto Rico", "Guam", "Virgin Islands",
}

# Ordered rules. disposition: "exclude" = unusable (missing data), "quarantine" = present but suspicious.
# stage: "all" rules run on every status; "sold" rules run after the status filter.
RULES = [
    # code,                       stage,  disposition,  description
    ("missing_location",          "all",  "exclude",    "state, city or zip_code is missing, or zip_code is not a valid 1-5 digit number"),
    ("invalid_state",             "all",  "quarantine", "state is not a U.S. state, DC, Puerto Rico, Guam or the U.S. Virgin Islands"),
    ("zip_state_mismatch",        "all",  "quarantine", f"state disagrees with the state {ZIP_STATE_MIN_SHARE:.0%}+ of rows in the same zip report (zips with >= {ZIP_STATE_MIN_ROWS} rows)"),
    ("status_not_sold",           "all",  "scope",      "status is not 'sold' (kept in raw data; outside the primary modelling dataset)"),
    ("missing_price",             "sold", "exclude",    "price is missing (target is never imputed)"),
    ("nonpositive_price",         "sold", "quarantine", "price <= 0"),
    ("price_int32_sentinel",      "sold", "quarantine", f"price >= {INT32_SENTINEL:,} (integer overflow / placeholder)"),
    ("price_below_min",           "sold", "quarantine", f"0 < price < ${MIN_PRICE:,}"),
    ("price_above_review_cap",    "sold", "quarantine", f"price > ${MAX_PRICE_REVIEW:,} (manual review)"),
    ("missing_bed",               "sold", "exclude",    "bed is missing (required baseline feature)"),
    ("missing_bath",              "sold", "exclude",    "bath is missing (required baseline feature)"),
    ("missing_house_size",        "sold", "exclude",    "house_size is missing (required baseline feature)"),
    ("implausible_bed",           "sold", "quarantine", f"bed > {MAX_BED}"),
    ("implausible_bath",          "sold", "quarantine", f"bath > {MAX_BATH}"),
    ("implausible_house_size",    "sold", "quarantine", f"house_size < {MIN_HOUSE_SQFT} or > {MAX_HOUSE_SQFT:,} sqft"),
    ("implausible_price_per_sqft", "sold", "quarantine", f"price / house_size outside ${MIN_PPSF}-${MAX_PPSF:,} per sqft"),
    ("implausible_acre_lot",      "sold", "quarantine", f"acre_lot > {MAX_ACRES:,} acres"),
    ("exact_duplicate_sale",      "sold", "exclude",    "same property key, price, sale date, bed, bath, house_size and acre_lot as an earlier kept row"),
]
RULE_INFO = {code: (stage, disp, desc) for code, stage, disp, desc in RULES}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    """Read the single CSV inside the zip. Identifier-like columns are kept as nullable ints."""
    df = pd.read_csv(RAW, dtype={"status": "string", "city": "string", "state": "string",
                                 "prev_sold_date": "string"})
    df.insert(0, "raw_row_id", np.arange(len(df), dtype="int64"))  # stable id back to the raw file
    return df


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise text and identifier columns. Original numeric values are not altered here."""
    df = df.copy()
    df["status"] = df["status"].str.strip().str.lower()
    df["state"] = df["state"].str.strip().str.replace(r"\s+", " ", regex=True)
    city = df["city"].str.strip().str.replace(r"\s+", " ", regex=True)
    # Spelling variants differing only by case/spacing map to the most frequent spelling in that state.
    key = pd.DataFrame({"state": df["state"], "ck": city.str.casefold(), "city": city}).dropna()
    canon = (key.groupby(["state", "ck", "city"]).size().rename("n").reset_index()
                .sort_values(["state", "ck", "n", "city"], ascending=[True, True, False, True])
                .drop_duplicates(["state", "ck"]).set_index(["state", "ck"])["city"])
    idx = pd.MultiIndex.from_arrays([df["state"], city.str.casefold()])
    df["city"] = pd.Series(canon.reindex(idx).to_numpy(), index=df.index, dtype="string")
    # Zip codes were read as floats, so leading zeros were lost (e.g. 601 -> "00601").
    z = df["zip_code"]
    valid_zip = z.notna() & (z >= 1) & (z <= 99999) & (z % 1 == 0)
    df["zip_code"] = pd.Series(np.where(valid_zip, z.fillna(0).astype("int64").astype(str).str.zfill(5), None),
                               index=df.index, dtype="string")
    # (street, zip_code) provisional property key. `street` is an anonymised numeric code in this dataset.
    has_key = df["street"].notna() & df["zip_code"].notna()
    df["property_key"] = pd.Series(np.where(has_key, df["street"].fillna(0).astype("int64").astype(str) + "|" + df["zip_code"].fillna(""), None),
                                   index=df.index, dtype="string")
    # Lot size of exactly 0 acres is not a physical value for a property: treat as unknown, not as a quarantine.
    df["acre_lot_was_zero"] = df["acre_lot"].eq(0)
    df.loc[df["acre_lot"].eq(0), "acre_lot"] = np.nan
    # Dates: parse, keep unparsable as missing. Meaning of prev_sold_date for 'sold' rows is unverified.
    df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors="coerce")
    return df


def location_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    missing = df["state"].isna() | df["city"].isna() | df["zip_code"].isna()
    invalid_state = df["state"].notna() & ~df["state"].isin(VALID_STATES)
    # Zip -> dominant state, computed over rows with a valid state.
    ok = df[df["zip_code"].notna() & df["state"].isin(VALID_STATES)]
    counts = ok.groupby(["zip_code", "state"]).size().rename("n").reset_index()
    tot = counts.groupby("zip_code")["n"].transform("sum")
    counts["share"] = counts["n"] / tot
    dom = counts[(tot >= ZIP_STATE_MIN_ROWS) & (counts["share"] >= ZIP_STATE_MIN_SHARE)].set_index("zip_code")["state"]
    dom_state = df["zip_code"].map(dom)
    mismatch = dom_state.notna() & df["state"].notna() & (dom_state != df["state"]) & df["state"].isin(VALID_STATES)
    # The invalid state row is also reported by zip dominance only if its state is valid, so no double counting.
    return {"missing_location": missing, "invalid_state": invalid_state & ~missing,
            "zip_state_mismatch": mismatch & ~missing}


def sold_flags(df: pd.DataFrame) -> dict[str, pd.Series]:
    p, hs = df["price"], df["house_size"]
    ppsf = p / hs.where(hs > 0)
    return {
        "missing_price": p.isna(),
        "nonpositive_price": p.le(0),
        "price_int32_sentinel": p.ge(INT32_SENTINEL),
        "price_below_min": p.gt(0) & p.lt(MIN_PRICE),
        "price_above_review_cap": p.gt(MAX_PRICE_REVIEW) & p.lt(INT32_SENTINEL),
        "missing_bed": df["bed"].isna(),
        "missing_bath": df["bath"].isna(),
        "missing_house_size": hs.isna(),
        "implausible_bed": df["bed"].gt(MAX_BED),
        "implausible_bath": df["bath"].gt(MAX_BATH),
        "implausible_house_size": hs.lt(MIN_HOUSE_SQFT) | hs.gt(MAX_HOUSE_SQFT),
        "implausible_price_per_sqft": ppsf.notna() & (ppsf.lt(MIN_PPSF) | ppsf.gt(MAX_PPSF)),
        "implausible_acre_lot": df["acre_lot"].gt(MAX_ACRES),
    }


def first_failing(df: pd.DataFrame, flags: dict[str, pd.Series], order: list[str]) -> pd.Series:
    """Assign each row the first rule in `order` that it fails (None = passes all)."""
    reason = pd.Series(pd.NA, index=df.index, dtype="string")
    for code in order:
        hit = flags[code].fillna(False) & reason.isna()
        reason[hit] = code
    return reason


def dup_cols_incomplete() -> list[str]:
    return ["property_key", "price", "prev_sold_date", "bed", "bath", "house_size", "acre_lot"]


def run() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    raw_hash_before = sha256(RAW)
    raw = load_raw()
    n_raw = len(raw)
    df = standardise(raw)
    waterfall = [{"step": "raw rows read", "rule": "", "removed": 0, "remaining": n_raw}]
    independent = []   # each rule's hits regardless of order
    exclusion_rows = []  # first-failing attribution: (rule, disposition, status_scope, rows)

    def record(code: str, flags: pd.Series, remaining_before: int, first_hits: int):
        stage, disp, desc = RULE_INFO[code]
        independent.append({"rule": code, "stage": stage, "disposition": disp, "description": desc,
                            "rows_hit_independently": int(flags.fillna(False).sum()),
                            "rows_removed_first_failing": first_hits})
        waterfall.append({"step": f"{stage}: {code}", "rule": code, "removed": first_hits,
                          "remaining": remaining_before - first_hits})

    # ---- Stage A: location validity on every status -------------------------------------------
    loc = location_flags(df)
    loc_order = ["missing_location", "invalid_state", "zip_state_mismatch"]
    loc_reason = first_failing(df, loc, loc_order)
    remaining = n_raw
    for code in loc_order:
        n = int((loc_reason == code).sum())
        record(code, loc[code], remaining, n)
        remaining -= n
    df["reason"] = loc_reason
    quarantine_parts = [df[df["reason"].isin(["invalid_state", "zip_state_mismatch"])]]
    excluded_missing_loc = int((df["reason"] == "missing_location").sum())
    df = df[df["reason"].isna()].drop(columns="reason")

    # ---- Stage B: scope to sold ----------------------------------------------------------------
    status_counts_after_location = df["status"].value_counts(dropna=False).to_dict()
    not_sold = df["status"].ne("sold")
    record("status_not_sold", not_sold, len(df), int(not_sold.sum()))
    df = df[~not_sold].copy()
    n_sold_scope = len(df)

    # ---- Stage C: sold-row validity and completeness -------------------------------------------
    sf = sold_flags(df)
    sold_order = [c for c, st, *_ in RULES if st == "sold" and c != "exact_duplicate_sale"]
    reason = first_failing(df, sf, sold_order)
    remaining = len(df)
    for code in sold_order:
        n = int((reason == code).sum())
        record(code, sf[code], remaining, n)
        remaining -= n
    df["reason"] = reason
    # Independent flag list per quarantined row, for review.
    qmask = df["reason"].notna() & df["reason"].map(lambda r: RULE_INFO.get(r, ("", ""))[1] == "quarantine").fillna(False)
    q = df[qmask].copy()
    q["all_flags"] = pd.concat({c: sf[c].reindex(q.index).fillna(False) for c in sold_order}, axis=1).apply(
        lambda r: ";".join(r.index[r.to_numpy()]), axis=1)
    quarantine_parts.append(q)
    clean = df[df["reason"].isna()].drop(columns="reason").copy()

    # ---- Side output for the Phase 4 missing-data experiment ----------------------------------------
    # Sold rows whose ONLY problem is a missing bed / bath / house_size: every value that is present passes
    # all validity rules. They are not part of the primary modelling set; target prices are never imputed.
    quarantine_codes = [c for c, st, disp, _ in RULES if st == "sold" and disp == "quarantine"]
    other_problem = pd.concat([sf[c].fillna(False) for c in quarantine_codes], axis=1).any(axis=1)
    incomplete = df[df["reason"].isin(["missing_bed", "missing_bath", "missing_house_size"]) & ~other_problem].copy()
    incomplete = incomplete[~(incomplete["property_key"].notna() & incomplete.duplicated(subset=dup_cols_incomplete(), keep="first"))]
    incomplete["property_group"] = incomplete["property_key"].fillna("row_" + incomplete["raw_row_id"].astype(str))
    incomplete = incomplete[["raw_row_id", "property_key", "property_group", "status", "price", "bed", "bath",
                             "house_size", "acre_lot", "city", "state", "zip_code"]].sort_values("raw_row_id")
    incomplete.to_parquet(PROCESSED / "sold_incomplete_features.parquet", index=False)

    # ---- Stage D: exact duplicate sales ----------------------------------------------------------
    dup_cols = ["property_key", "price", "prev_sold_date", "bed", "bath", "house_size", "acre_lot"]
    # Rows lacking a property key are never treated as duplicates (they cannot be identified).
    dup = clean["property_key"].notna() & clean.duplicated(subset=dup_cols, keep="first")
    record("exact_duplicate_sale", dup, len(clean), int(dup.sum()))
    clean = clean[~dup].copy()

    # ---- Derived columns and ordering ------------------------------------------------------------
    grp_size = clean.groupby("property_key")["raw_row_id"].transform("size")
    clean["repeated_property"] = grp_size.fillna(1).gt(1)
    # Grouping id for leakage-safe splits: rows without a key are singletons.
    clean["property_group"] = clean["property_key"].fillna("row_" + clean["raw_row_id"].astype(str))
    clean["price_per_sqft"] = (clean["price"] / clean["house_size"]).round(2)
    clean["sold_year"] = clean["prev_sold_date"].dt.year.astype("Int16")
    clean["bed"] = clean["bed"].astype("int16"); clean["bath"] = clean["bath"].astype("float32")
    clean = clean[["raw_row_id", "property_key", "property_group", "repeated_property", "status", "price",
                   "bed", "bath", "house_size", "acre_lot", "acre_lot_was_zero", "city", "state", "zip_code",
                   "prev_sold_date", "sold_year", "price_per_sqft", "brokered_by", "street"]].sort_values("raw_row_id")

    quarantine = pd.concat(quarantine_parts, ignore_index=True)
    quarantine = quarantine.rename(columns={"reason": "quarantine_reason"})
    quarantine["quarantine_reason"] = quarantine["quarantine_reason"].fillna(
        pd.Series(np.where(quarantine["state"].isin(VALID_STATES), "zip_state_mismatch", "invalid_state"), index=quarantine.index))
    quarantine = quarantine.sort_values("raw_row_id")

    # ---- Integrity checks ------------------------------------------------------------------------
    removed = sum(w["removed"] for w in waterfall)
    assert n_raw - removed == len(clean), "waterfall does not reconcile with clean row count"
    assert clean["raw_row_id"].is_unique
    assert clean["price"].notna().all() and (clean["price"] > 0).all()
    assert clean["state"].isin(VALID_STATES).all()
    assert set(clean["status"]) == {"sold"}
    assert RAW.stat().st_size > 0

    # ---- Write outputs ---------------------------------------------------------------------------
    clean.to_parquet(PROCESSED / "sold_clean.parquet", index=False)
    quarantine.to_parquet(PROCESSED / "quarantine.parquet", index=False)

    wf = pd.DataFrame(waterfall)
    wf.to_csv(REPORTS / "cleaning_waterfall.csv", index=False)
    ind = pd.DataFrame(independent)
    ind.to_csv(REPORTS / "cleaning_exclusions_by_reason.csv", index=False)

    miss_raw = raw.drop(columns="raw_row_id").isna().mean().mul(100).round(2)
    sold_raw = raw[raw["status"].str.strip().str.lower().eq("sold")].drop(columns="raw_row_id")
    miss = pd.DataFrame({"missing_pct_all_raw": miss_raw,
                         "missing_pct_sold_raw": sold_raw.isna().mean().mul(100).round(2)})
    miss["missing_pct_sold_clean"] = clean.reindex(columns=miss.index).isna().mean().mul(100).round(2)
    miss.to_csv(REPORTS / "cleaning_missingness.csv", index_label="column")

    # Outlier review: per-rule counts + up to 25 sample rows each, plus the priciest retained rows.
    sample_cols = ["raw_row_id", "quarantine_reason", "status", "price", "bed", "bath", "house_size", "acre_lot", "city", "state", "zip_code"]
    samples = (quarantine.sort_values("price", ascending=False).groupby("quarantine_reason").head(25)
               .sort_values(["quarantine_reason", "price"], ascending=[True, False])[sample_cols])
    samples.to_csv(REPORTS / "cleaning_quarantine_samples.csv", index=False)
    top = clean.sort_values("price", ascending=False).head(50)[
        ["raw_row_id", "price", "bed", "bath", "house_size", "acre_lot", "price_per_sqft", "city", "state", "zip_code"]]
    top.to_csv(REPORTS / "cleaning_top_priced_retained.csv", index=False)
    dist = clean[["price", "bed", "bath", "house_size", "acre_lot", "price_per_sqft"]].describe(
        percentiles=[.001, .01, .05, .5, .95, .99, .999]).T
    dist.to_csv(REPORTS / "cleaning_clean_numeric_summary.csv")

    # Geographic coverage of the clean set.
    st = clean.groupby("state").agg(rows=("price", "size"), cities=("city", "nunique"),
                                    median_price=("price", "median")).sort_values("rows", ascending=False)
    st.to_csv(REPORTS / "cleaning_state_coverage.csv")
    cc = clean.groupby(["state", "city"]).agg(rows=("price", "size"), median_price=("price", "median"),
                                              properties=("property_group", "nunique")).reset_index()
    cc.sort_values("rows", ascending=False).to_csv(REPORTS / "cleaning_city_coverage.csv", index=False)
    cover = pd.DataFrame([{"min_rows_per_city": t, "n_cities": int((cc["rows"] >= t).sum()),
                           "rows_covered": int(cc.loc[cc["rows"] >= t, "rows"].sum()),
                           "pct_of_clean_rows": round(100 * cc.loc[cc["rows"] >= t, "rows"].sum() / len(clean), 1)}
                          for t in (1, 10, 30, 100, 500)])
    cover.to_csv(REPORTS / "cleaning_city_coverage_thresholds.csv", index=False)

    # Repeated-property and sale-date review.
    rep = {"clean_rows": len(clean),
           "rows_without_property_key": int(clean["property_key"].isna().sum()),
           "distinct_property_groups": int(clean["property_group"].nunique()),
           "properties_with_multiple_sales": int(clean.loc[clean["repeated_property"], "property_group"].nunique()),
           "rows_in_repeated_properties": int(clean["repeated_property"].sum()),
           "max_rows_one_property": int(clean.groupby("property_group").size().max())}
    yrs = clean["sold_year"].dropna().astype(int)
    sale_dates = {"min": str(clean["prev_sold_date"].min().date()), "max": str(clean["prev_sold_date"].max().date()),
                  "rows_missing_date": int(clean["prev_sold_date"].isna().sum()),
                  "rows_before_2000": int((yrs < 2000).sum()),
                  "rows_by_year_2015_plus": {int(k): int(v) for k, v in yrs[yrs >= 2015].value_counts().sort_index().items()}}

    raw_hash_after = sha256(RAW)
    assert raw_hash_before == raw_hash_after, "raw file changed during cleaning"

    summary = {"raw_file": str(RAW.relative_to(ROOT)), "raw_sha256": raw_hash_before, "raw_unchanged": True,
               "raw_rows": n_raw, "rows_in_scope_sold_after_location_checks": n_sold_scope,
               "clean_rows": len(clean), "quarantined_rows": len(quarantine),
               "incomplete_feature_rows": len(incomplete),
               "quarantined_by_reason": quarantine["quarantine_reason"].value_counts().to_dict(),
               "status_counts_after_location_checks": status_counts_after_location,
               "repeated_properties": rep, "sale_dates_in_clean": sale_dates,
               "acre_lot_set_to_missing_because_zero": int(clean["acre_lot_was_zero"].sum()),
               "acre_lot_missing_in_clean": int(clean["acre_lot"].isna().sum())}
    (REPORTS / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, wf, ind, cover)
    return summary


def write_markdown(summary: dict, wf: pd.DataFrame, ind: pd.DataFrame, cover: pd.DataFrame) -> None:
    def table(d: pd.DataFrame) -> str:
        return d.to_markdown(index=False) if hasattr(d, "to_markdown") and _has_tabulate() else d.to_string(index=False)
    lines = ["# Phase 2 cleaning report (generated by src/data_cleaning.py)", "",
             f"Raw file: `{summary['raw_file']}` (SHA-256 `{summary['raw_sha256'][:16]}...`, verified unchanged after the run).", "",
             "**Price caveat:** the source's definition of `price` for `sold` rows has not been verified. "
             "These rows are *sold-status records*, not confirmed transaction prices.", "",
             f"- Raw rows: {summary['raw_rows']:,}", f"- Clean sold rows: {summary['clean_rows']:,}",
             f"- Quarantined rows (all statuses): {summary['quarantined_rows']:,}", "",
             "## Row-count waterfall (each row counted under the first rule it fails)", "", "```", wf.to_string(index=False), "```", "",
             "## Rules and independent hit counts", "", "```", ind.drop(columns="stage").to_string(index=False), "```", "",
             "## City coverage of clean rows", "", "```", cover.to_string(index=False), "```", ""]
    (REPORTS / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    out = run()
    json.dump({k: out[k] for k in ("raw_rows", "clean_rows", "quarantined_rows")}, sys.stdout, indent=2)
    print()
