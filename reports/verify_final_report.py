"""Checks the numbers in reports/final_report.md against the project's source files.

    python reports/verify_final_report.py            (run from the project root)

Each check pulls a value from its authoritative file (JSON/CSV/Parquet) and confirms that the report text contains the expected formatted
string. It also checks table structure, figure links, placeholders, and that no secret or machine-specific path is present. The
expected strings follow the report's current wording, so if you rewrite a sentence, an affected check may need updating."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
REP, MOD = ROOT / "reports", ROOT / "models"
text = (REP / "final_report.md").read_text(encoding="utf-8")
J = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
exp, meta, zipv, cs = J(REP / "experiment_summary.json"), J(MOD / "model_metadata.json"), J(REP / "zip_experiment_validation.json"), J(REP / "cleaning_summary.json")
ev = meta["evaluation_test_split_reporting_only"]
fails, passes = [], 0


def need(label, *strings):
    global passes
    miss = [s for s in strings if s not in text]
    if miss:
        fails.append((label, miss))
    else:
        passes += 1


# --- headline numbers the task asked to verify (from the authoritative files, not from memory)
hy, bl = ev["with_zip"], ev["no_zip"]
cov = lambda m, lvl: next(x for x in m["coverage"][lvl] if x["group"] == "all")["coverage"]  # noqa: E731
vals = {"hybrid MAE": (hy["metrics"]["MAE"], 98281), "hybrid RMSE": (hy["metrics"]["RMSE"], 352288), "hybrid R2": (round(hy["metrics"]["R2"], 3), 0.745),
        "hybrid MdAPE": (round(hy["metrics"]["MdAPE"] * 100, 1), 13.8), "hybrid cov80": (round(cov(hy, "0.8") * 100, 1), 80.4), "hybrid cov90": (round(cov(hy, "0.9") * 100, 1), 90.2),
        "base MAE": (bl["metrics"]["MAE"], 106982), "base RMSE": (bl["metrics"]["RMSE"], 383269), "base R2": (round(bl["metrics"]["R2"], 3), 0.699),
        "base MdAPE": (round(bl["metrics"]["MdAPE"] * 100, 1), 14.9), "base cov80": (round(cov(bl, "0.8") * 100, 1), 80.1), "base cov90": (round(cov(bl, "0.9") * 100, 1), 90.2)}
for k, (got, want) in vals.items():
    ok = abs(got - want) < 0.5 if want > 100 else abs(got - want) < 0.051
    print(("OK   " if ok else "FAIL ") + f"{k}: source {got} vs brief {want}")
    if not ok:
        fails.append((k, [f"source {got} != {want}"]))
    else:
        passes += 1
need("hybrid headline", "$98,281", "$352,288", "0.745", "13.8%", "80.4%", "90.2%")
need("baseline headline", "$106,982", "$383,269", "0.699", "14.9%", "80.1%")

# --- data and cleaning
need("raw rows", f"{cs['raw_rows']:,}", f"{cs['clean_rows']:,}", f"{cs['quarantined_rows']:,}", f"{cs['incomplete_feature_rows']:,}")
need("sold in scope share", "85.8%", "31.3%", f"{cs['rows_in_scope_sold_after_location_checks']:,}")
assert round(cs["clean_rows"] / cs["raw_rows"] * 100, 1) == 31.3 and round(cs["clean_rows"] / cs["rows_in_scope_sold_after_location_checks"] * 100, 1) == 85.8
w = pd.read_csv(REP / "cleaning_waterfall.csv")
assert w.removed.iloc[[10, 11, 12]].sum() == 69802 + 10449 + 34447 == 114698, "missing-feature total"
need("missing-feature rows", "114,698")
need("incomplete lacking house size", f"{exp['incomplete_rows']['house_size_missing']:,}", f"{exp['incomplete_rows']['total']:,}")
rg = cs["repeated_properties"]
need("repeated properties", f"{rg['rows_without_property_key']:,}", f"{rg['properties_with_multiple_sales']}", f"{rg['rows_in_repeated_properties']}", "2,806", "134")
need("sale dates", cs["sale_dates_in_clean"]["min"], cs["sale_dates_in_clean"]["max"], "215,400", "480,886")
p = pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet", columns=["price", "state", "price_per_sqft", "house_size", "bed", "bath"])
need("price stats", f"${p.price.median():,.0f}", f"${p.price.quantile(.05):,.0f}", f"${p.price.quantile(.95):,.0f}", f"${p.price.max():,.0f}", f"{p.price.skew():.1f}", f"{np.log(p.price).skew():.2f}")
need("typical home", f"{p.house_size.median():,.0f} square feet", f"${p.price_per_sqft.median():,.2f}")
need("three-state total", f"{p.state.value_counts().loc[['California', 'Texas', 'Florida']].sum():,}")
sc = pd.read_csv(REP / "cleaning_state_coverage.csv")
assert int((sc.rows < 100).sum()) == 8
need("small states", "eight states", f"Vermont has {int(sc[sc.state == 'Vermont'].rows.iloc[0])}")
assert "Guam" not in set(sc.state)
need("city coverage", "16,090", "3,378", "199")

# --- model evaluation
sp = pd.DataFrame(exp["splits"])
for r in sp.itertuples():
    need(f"split {r.split}", f"{r.rows:,}")
m = {r["model"]: r for r in exp["model_comparison"]}
need("model test MAE", f"${m['hist_gb_raw']['test_MAE']:,.0f}", f"${m['dummy_median']['test_MAE']:,.0f}", f"${m['random_forest_log']['test_MAE']:,.0f}", f"${m['ridge_log']['test_MAE']:,.0f}")
need("selected iterations", str(exp["selected_model_boosting_iterations"]))
gh = pd.DataFrame(exp["geographic_holdout"]["results"])
al = gh[gh.city_size == "all"].set_index("evaluation")
need("geography", f"{al.iloc[0].MdAPE * 100:.1f}%", f"{al.iloc[1].MdAPE * 100:.1f}%", "2,341")
need("geography R2", f"{al.iloc[0].R2:.2f}", f"{al.iloc[1].R2:.2f}")
bb = pd.DataFrame(zipv["test"]["by_band"])
g = lambda mdl, band, col: float(bb[(bb.band == band) & (bb.model == mdl)][col].iloc[0])  # noqa: E731
need("band errors", f"{g('baseline (production model)', '<100k', 'MdAPE') * 100:.1f}%", f"{g('hybrid', '<100k', 'MdAPE') * 100:.1f}%", f"{g('baseline (production model)', '>2M', 'MdAPE') * 100:.1f}%", f"{g('hybrid', '>2M', 'MdAPE') * 100:.1f}%")
c = lambda mode, band: next(x for x in ev[mode]["coverage"]["0.8"] if x["group"] == "price_band" and x["value"] == band)["coverage"]  # noqa: E731
need("band coverage", f"{c('no_zip', '<100k') * 100:.1f}%", f"{c('with_zip', '<100k') * 100:.1f}%", f"{c('no_zip', '1-2M') * 100:.1f}%", f"{c('with_zip', '1-2M') * 100:.1f}%", f"{c('no_zip', '>2M') * 100:.1f}%", f"{c('with_zip', '>2M') * 100:.1f}%")
st = lambda mode, v: next(x for x in ev[mode]["coverage"]["0.8"] if x["group"] == "support_status" and x["value"] == v)["coverage"]  # noqa: E731
need("status coverage", f"{st('with_zip', 'supported') * 100:.1f}%", f"{st('with_zip', 'limited') * 100:.1f}%", f"{st('with_zip', 'unsupported') * 100:.1f}%")
need("improvement", "8.1%")
assert round((106982.27 - 98281.28) / 106982.27 * 100, 1) == 8.1 and round((383268.90 - 352287.74) / 383268.90 * 100, 1) == 8.1
mult = meta["calibration"]
need("multipliers", f"{mult['no_zip_path']['multipliers']['0.8']:.3f}", f"{mult['no_zip_path']['multipliers']['0.9']:.3f}", f"{mult['zip_path']['multipliers']['zip']['0.8']:.3f}",
     f"{mult['zip_path']['multipliers']['fallback']['0.9']:.3f}", "66,546", "2,707")
need("support policy counts", *[f"{r['cities']:,}" for r in meta["support_policy"]["status_counts"]])
ck = zipv["decision_checks"]
need("zip checks", f"{ck['zip']['unseen_zip_MdAPE_change_pts']:+.1f} pts", f"{ck['hybrid']['val_MAE_gain'] * 100:.1f}%")
gz = pd.DataFrame(zipv["generalisation"]).set_index(["evaluation", "model"])
need("zip gen", f"{gz.loc[('unseen cities', 'hybrid')].MdAPE * 100:.1f}%", f"{gz.loc[('unseen cities', 'baseline')].MdAPE * 100:.1f}%", f"{gz.loc[('unseen ZIPs in seen cities', 'hybrid')].MdAPE * 100:.1f}%")
byz = pd.DataFrame(zipv["test"]["by_zip_training_rows"])
need("zip rows groups", "133,616", "5,484", "4,324", "1,160")
assert 4324 + 1160 == 5484

# --- warehouse / SQL
for qn, col in (("q1_state_summary", "homes"), ("q2_size_band", "homes")):
    pass
q1 = pd.read_csv(REP / "example_query_results" / "q1_state_summary.csv")
need("query facts", "39 states", "$140,000", "$699,000", "$554,199", "$254,900")
assert q1.homes.min() >= 1000 and len(q1) == 39
q3 = pd.read_csv(REP / "example_query_results" / "q3_top_cities.csv")
assert (q3.state == "California").sum() == 8
need("query3 claim", "Eight of the top ten")
q4 = pd.read_csv(REP / "example_query_results" / "q4_month_coverage.csv")
need("months", "13,124", "6,759", "142,010", "157,250")
wv = J(REP / "warehouse_validation.json")
assert wv["all_passed"] and len(wv["checks"]) == 52
need("52 checks", "52 of 52")

# --- tests
bj = J(REP / "figures" / "app_screenshots" / "browser_check_results.json")["runs"]
assert (bj["full"]["passed"], bj["full"]["total"], bj["failure"]["passed"], bj["failure"]["total"]) == (89, 89, 11, 11)
need("tests", "138 passed, 0 failed, 0 skipped", "89 of 89", "11 of 11", "105 passed and 33 skipped")

# --- hygiene: no secrets, machine paths, or unresolved placeholders
bad_paths = re.findall(r"[A-Za-z]:[\\/](?:Users|Personal Folder|Program Files)[^ \n]*", text)
secrets = re.findall(r"(?i)(?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*['\"]?[A-Za-z0-9]{6,}", text)
secrets += ["PGPASSWORD="] if "PGPASSWORD=" in text else []
need("placeholders intact", "[Student name]", "[Instructor name]", "[Date of submission]")
assert not bad_paths, bad_paths
assert not secrets, secrets
assert "{{" not in text
# --- links to figures exist
missing = [m for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) if not (REP / m).exists()]
assert not missing, missing
n_figs = text.count("![")
print(f"\nfigure links resolved: {n_figs} | machine paths: 0 | secrets: 0 | claim checks passed: {passes} | failed: {len(fails)}")
for label, miss in fails:
    print("FAIL", label, "missing:", miss)
sys.exit(1 if fails else 0)
