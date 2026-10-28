"""Integrate the approved hybrid ZIP model into production (model version 2).

    python src/integrate_zip_model.py

What it does (and the data each step may use):
  1. Backs up the v1 production artifacts to models/backups/v1_no_zip/ and verifies them against the hashes recorded
     by the ZIP experiment. Nothing proceeds if the backup cannot be verified.
  2. Loads the experiment's hybrid candidate (no retraining) and checks it reproduces its recorded validation result.
  3. ZIP-path prediction ranges: spread model fitted on the VALIDATION split, conformal multipliers set on the
     CALIBRATION split, separately for the two ZIP routes (ZIP model / city-level fallback).
     The no-ZIP path keeps the v1 model and the v1 calibration unchanged.
  4. Location support rebuilt from VALIDATION-split errors of both paths (same thresholds as v1).
  5. TEST split: reporting only (coverage, metrics). No value computed on it feeds back into any step.
  6. Writes the v2 artifacts; models/house_price_model.joblib (the v1 model, still used without ZIP) is not modified.

Rollback: see reports/zip_feature_experiment.md ("Rollback").
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import hashlib
import json
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS, PROCESSED = ROOT / "models", ROOT / "reports", ROOT / "data" / "processed"
V1_FILES = ["house_price_model.joblib", "location_support.csv", "feature_schema.json", "model_metadata.json"]
BACKUP = MODELS / "backups" / "v1_no_zip"
CANDIDATE = MODELS / "experiments" / "zip" / "zip_candidates.joblib"
EXP_JSON = REPORTS / "zip_experiment_validation.json"
VERSION = "2.0.0-zip-hybrid"
SCALE_FLOOR = 0.03
MIN_ROUTE_CAL_ROWS = 500          # below this a ZIP route uses the pooled calibration multiplier
LEVELS = {"0.8": 0.20, "0.9": 0.10}
BANDS = ([0, 100e3, 200e3, 400e3, 700e3, 1e6, 2e6, 1e12], ["<100k", "100-200k", "200-400k", "400-700k", "700k-1M", "1-2M", ">2M"])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------------------------------------
# 1. backup
# ------------------------------------------------------------------------------------------------
def backup_v1() -> dict:
    recorded = json.loads(EXP_JSON.read_text(encoding="utf-8"))["production_hashes"]
    manifest = BACKUP / "MANIFEST.json"
    if manifest.exists():
        for f in V1_FILES:
            if sha256(BACKUP / f) != recorded[f]:
                raise SystemExit(f"backup file {f} does not match the recorded v1 hash; aborting")
        log("v1 backup present and verified")
    else:
        for f in V1_FILES:
            if sha256(MODELS / f) != recorded[f]:
                raise SystemExit(f"models/{f} is not the recorded v1 file and no backup exists; aborting")
        BACKUP.mkdir(parents=True, exist_ok=True)
        for f in V1_FILES:
            shutil.copy2(MODELS / f, BACKUP / f)
            assert sha256(BACKUP / f) == recorded[f]
        manifest.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        "version": "1.0.0-no-zip", "sha256": recorded}, indent=2), encoding="utf-8")
        log(f"v1 backed up to {BACKUP.relative_to(ROOT)} and verified")
    if sha256(MODELS / "house_price_model.joblib") != recorded["house_price_model.joblib"]:
        raise SystemExit("the v1 model file in models/ differs from v1; aborting")
    return recorded


# ------------------------------------------------------------------------------------------------
# data and predictions
# ------------------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = mc.to_model_frame(pd.read_parquet(PROCESSED / "sold_clean.parquet", columns=[
        "raw_row_id", "price", "bed", "bath", "house_size", "acre_lot", "state", "city", "zip_code", "property_group"]))
    df = df.sort_values("raw_row_id").reset_index(drop=True)
    df["city_key"] = df.state + "|" + df.city
    df["split"] = mc.assign_split(df.property_group)
    prod = json.loads((REPORTS / "experiment_summary.json").read_text(encoding="utf-8"))["splits"]
    counts = df.split.value_counts()
    assert all(counts[r["split"]] == r["rows"] for r in prod), "split differs from production"
    assert (df.groupby("property_group").split.nunique() == 1).all(), "a property group crosses splits"
    df["band"] = pd.cut(df.price, bins=BANDS[0], labels=BANDS[1]).astype(str)
    return df


class Predictor:
    """Batch version of the production prediction paths (predict.py uses the same artifacts row by row)."""

    def __init__(self, v1: dict, cand: dict):
        self.v1, self.cand = v1, cand
        self.pre = v1["model"].regressor_.named_steps["pre"]

    def paths(self, d: pd.DataFrame) -> dict:
        X = mc.build_features(d)
        p_prod = mc.predict_dollars(self.v1["model"], X)
        p_zip = mc.predict_dollars(self.cand["zip_model"], mc.build_zip_features(d))
        n = d.zip_code.map(self.cand["train_zip_counts"]).fillna(0).to_numpy()
        route = n >= self.cand["hybrid_min_zip_rows"]
        Z_pre = self.pre.transform(X)
        return {"p_prod": p_prod, "p_zip": p_zip, "route_zip": route, "p_hybrid": np.where(route, p_zip, p_prod), "Z_pre": Z_pre}

    def v1_sigma(self, P) -> np.ndarray:  # exactly the v1 no-ZIP spread
        return np.maximum(self.v1["scale_model"].predict(np.column_stack([P["Z_pre"], np.log(P["p_prod"])])),
                          self.v1["scale_floor"])


def zip_scale_features(P) -> np.ndarray:
    return np.column_stack([P["Z_pre"], np.log(P["p_hybrid"]), P["route_zip"].astype(float)])


def coverage_rows(y, est, lo, hi, groups: dict) -> list:
    rows = []
    d = pd.DataFrame({"y": y, "est": est, "lo": lo, "hi": hi, **groups})
    d["covered"] = (d.y >= d.lo) & (d.y <= d.hi)
    d["w"] = d.hi - d.lo
    rows.append({"group": "all", "value": "all", "n": len(d), "coverage": float(d.covered.mean()),
                 "median_width": float(d.w.median()), "mean_width": float(d.w.mean())})
    for g in groups:
        for v, s in d.groupby(g, observed=True):
            rows.append({"group": g, "value": str(v), "n": len(s), "coverage": float(s.covered.mean()),
                         "median_width": float(s.w.median()), "mean_width": float(s.w.mean())})
    return rows


# ------------------------------------------------------------------------------------------------
def main() -> None:
    recorded = backup_v1()
    v1 = joblib.load(MODELS / "house_price_model.joblib")
    cand = joblib.load(CANDIDATE)
    exp = json.loads(EXP_JSON.read_text(encoding="utf-8"))
    df = load()
    S = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "validation", "calibration", "test")}
    pr = Predictor(v1, cand)
    P = {s: pr.paths(S[s]) for s in ("validation", "calibration", "test")}

    # 2. reproduction check against the experiment's recorded validation result
    rec = next(r for r in exp["validation"] if r["model"] == "hybrid")
    got = mc.metrics(S["validation"].price, P["validation"]["p_hybrid"])
    assert abs(got["MAE"] - rec["MAE"]) <= 1e-6 * rec["MAE"] and abs(got["MdAPE"] - rec["MdAPE"]) < 1e-9, \
        f"candidate does not reproduce its recorded validation result ({got['MAE']} vs {rec['MAE']})"
    log(f"reproduction check passed: validation MAE {got['MAE']:,.2f} == recorded {rec['MAE']:,.2f}")

    # 3. ZIP-path ranges: spread model on validation, multipliers on calibration (per route)
    yv, yc = S["validation"].price.to_numpy(), S["calibration"].price.to_numpy()
    rv = np.abs(np.log(yv) - np.log(P["validation"]["p_hybrid"]))
    scale = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                          random_state=mc.CONFIG["seed"]).fit(zip_scale_features(P["validation"]), rv)
    sig_c = np.maximum(scale.predict(zip_scale_features(P["calibration"])), SCALE_FLOOR)
    score_c = np.abs(np.log(yc) - np.log(P["calibration"]["p_hybrid"])) / sig_c
    route_c = P["calibration"]["route_zip"]
    q, cal_rows = {}, {}
    for name, mask in (("zip", route_c), ("fallback", ~route_c)):
        n = int(mask.sum())
        use = mask if n >= MIN_ROUTE_CAL_ROWS else np.ones_like(mask)
        q[name] = {lvl: mc.conformal_quantile(score_c[use], a) for lvl, a in LEVELS.items()}
        cal_rows[name] = {"rows": n, "pooled": bool(n < MIN_ROUTE_CAL_ROWS)}
    log(f"ZIP-path calibration: {cal_rows}; multipliers {q}")

    def zip_interval(Pp, lvl):
        sig = np.maximum(scale.predict(zip_scale_features(Pp)), SCALE_FLOOR)
        half = np.where(Pp["route_zip"], q["zip"][lvl], q["fallback"][lvl]) * sig
        return Pp["p_hybrid"] * np.exp(-half), Pp["p_hybrid"] * np.exp(half)

    def v1_interval(Pp, lvl):
        half = v1["conformal_quantiles"][lvl] * pr.v1_sigma(Pp)
        return Pp["p_prod"] * np.exp(-half), Pp["p_prod"] * np.exp(half)

    # 4. location support from VALIDATION rows only (both paths), v1 thresholds
    sp = mc.CONFIG["support_policy"]
    v = S["validation"].assign(ape_no_zip=np.abs(P["validation"]["p_prod"] - yv) / yv,
                               ape_zip=np.abs(P["validation"]["p_hybrid"] - yv) / yv)
    city = v.groupby("city_key").agg(heldout_rows=("price", "size"), heldout_MdAPE_no_zip=("ape_no_zip", "median"),
                                     heldout_MdAPE_zip=("ape_zip", "median"))
    loc = (df.groupby(["state", "city", "city_key"]).agg(n_total=("price", "size"), median_price=("price", "median")).reset_index()
           .merge(S["train"].groupby("city_key").size().rename("n_train").reset_index(), on="city_key", how="left")
           .merge(city.reset_index(), on="city_key", how="left"))
    loc["n_train"] = loc.n_train.fillna(0).astype(int)
    loc["heldout_rows"] = loc.heldout_rows.fillna(0).astype(int)
    loc["heldout_MdAPE"] = loc[["heldout_MdAPE_no_zip", "heldout_MdAPE_zip"]].max(axis=1, skipna=False)  # conservative

    def classify(r):
        if r.n_train >= sp["supported_min_train_rows"] and r.heldout_rows >= sp["min_heldout_rows"] \
                and r.heldout_MdAPE <= sp["max_heldout_mdape"]:
            return "supported", "enough training rows and acceptable validation error with and without ZIP"
        if r.n_train >= sp["limited_min_train_rows"]:
            why = []
            if r.n_train < sp["supported_min_train_rows"]:
                why.append(f"only {r.n_train} training rows")
            if r.heldout_rows < sp["min_heldout_rows"]:
                why.append("too few validation rows to verify accuracy")
            elif r.heldout_MdAPE > sp["max_heldout_mdape"]:
                why.append(f"validation median error {r.heldout_MdAPE:.0%} exceeds {sp['max_heldout_mdape']:.0%}")
            return "limited", "; ".join(why)
        return "unsupported", f"only {r.n_train} training rows (< {sp['limited_min_train_rows']})"
    cls = loc.apply(classify, axis=1, result_type="expand")
    loc["status"], loc["reason"] = cls[0], cls[1]
    v1_support = pd.read_csv(BACKUP / "location_support.csv", keep_default_na=False)[["state", "city", "status"]]
    trans = (loc[["state", "city", "status"]].merge(v1_support.rename(columns={"status": "v1_status"}), on=["state", "city"], how="left")
             .groupby(["v1_status", "status"]).size().rename("cities").reset_index())
    log("support transitions v1 -> v2:\n" + trans.to_string(index=False))
    moved = loc.merge(v1_support.rename(columns={"status": "v1_status"}), on=["state", "city"])
    moved = moved[(moved.v1_status == "supported") & (moved.status == "limited")]
    reason_kind = np.where(moved.reason.str.contains("exceeds"), "validation error above the 25% limit",
                           np.where(moved.reason.str.contains("too few validation rows"), "too few validation rows", "other"))
    down_reasons = "; ".join(f"{n} {k}" for k, n in pd.Series(reason_kind).value_counts().items()) or "none"
    status_of = loc.set_index("city_key").status

    # 5. TEST split: reporting only
    t, Pt = S["test"], P["test"]
    yt = t.price.to_numpy()
    t_status = t.city_key.map(status_of).fillna("unsupported").to_numpy()
    groups = {"price_band": t.band.to_numpy(), "support_status": t_status}
    ev = {"no_zip": {"metrics": mc.metrics(yt, Pt["p_prod"]), "coverage": {}},
          "with_zip": {"metrics": mc.metrics(yt, Pt["p_hybrid"]), "coverage": {}}}
    for lvl in LEVELS:
        ev["no_zip"]["coverage"][lvl] = coverage_rows(yt, Pt["p_prod"], *v1_interval(Pt, lvl), groups)
        ev["with_zip"]["coverage"][lvl] = coverage_rows(
            yt, Pt["p_hybrid"], *zip_interval(Pt, lvl),
            {**groups, "zip_route": np.where(Pt["route_zip"], "ZIP model", "city-level fallback")})
    for mode in ev:
        for k in ("n", "MAE", "RMSE", "R2", "MedAE", "MdAPE"):
            ev[mode]["metrics"][k] = float(ev[mode]["metrics"][k])
        log(f"TEST (reporting) {mode}: MAE {ev[mode]['metrics']['MAE']:,.0f} MdAPE {ev[mode]['metrics']['MdAPE']:.3f} "
            f"coverage 80% {ev[mode]['coverage']['0.8'][0]['coverage']:.3f} 90% {ev[mode]['coverage']['0.9'][0]['coverage']:.3f}")

    # 6. artifacts
    zip_map = (df.groupby(["state", "city", "zip_code"]).size().rename("rows_in_dataset").reset_index())
    zip_map["zip_train_rows"] = zip_map.zip_code.map(cand["train_zip_counts"]).fillna(0).astype(int)
    zip_map["route"] = np.where(zip_map.zip_train_rows >= cand["hybrid_min_zip_rows"], "zip_model", "city_fallback")
    assert zip_map.zip_code.str.fullmatch(r"\d{5}").all()
    zip_map.sort_values(["state", "city", "zip_code"]).to_csv(MODELS / "zip_city_map.csv", index=False)
    joblib.dump({"version": VERSION, "zip_model": cand["zip_model"], "train_zip_counts": cand["train_zip_counts"],
                 "min_zip_rows": cand["hybrid_min_zip_rows"], "prior_zip": cand["prior_zip"],
                 "scale_model": scale, "scale_floor": SCALE_FLOOR, "conformal_quantiles": q,
                 "calibration_rows": cal_rows, "feature_columns": mc.ZIP_FEATURE_COLS},
                MODELS / "zip_model.joblib", compress=3)
    out_cols = ["state", "city", "status", "reason", "n_train", "heldout_rows", "heldout_MdAPE", "heldout_MdAPE_no_zip",
                "heldout_MdAPE_zip", "median_price"]
    loc[out_cols].sort_values(["state", "city"]).to_csv(MODELS / "location_support.csv", index=False)

    v1_schema = json.loads((BACKUP / "feature_schema.json").read_text(encoding="utf-8"))
    v1_schema["version"] = VERSION
    v1_schema["inputs"]["zip_code"] = ("optional 5-character string (leading zeros kept); must be listed for the chosen "
                                       "state and city in zip_city_map.csv; ZIPs with fewer than "
                                       f"{cand['hybrid_min_zip_rows']} training rows use the city-level model")
    v1_schema["zip_model_features"] = mc.ZIP_FEATURE_COLS
    (MODELS / "feature_schema.json").write_text(json.dumps(v1_schema, indent=2), encoding="utf-8")

    v1_meta = json.loads((BACKUP / "model_metadata.json").read_text(encoding="utf-8"))
    status_counts = loc.groupby("status").agg(cities=("city", "size"), rows=("n_total", "sum")).reset_index()
    meta = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_period": "about Oct 2021 - May 2022 (prev_sold_date of sold-status records)",
            "target": mc.CONFIG["target"],
            "components": {
                "no_zip": {"model": "models/house_price_model.joblib (v1 hist_gb_log, unchanged)",
                           "intervals": "v1 normalised split-conformal calibration, unchanged"},
                "with_zip": {"model": f"models/zip_model.joblib: ZIP model if the ZIP has >= {cand['hybrid_min_zip_rows']} "
                                      "training rows, otherwise the v1 model (city-level fallback)",
                             "intervals": "normalised split conformal; spread model fitted on the validation split, "
                                          "multipliers per route from the calibration split"}},
            "v1_metadata": v1_meta, "seed": mc.CONFIG["seed"],
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
            "calibration": {"zip_path": {"multipliers": q, "rows": cal_rows, "min_route_rows": MIN_ROUTE_CAL_ROWS,
                                         "scale_model_fit_rows": int(len(S["validation"]))},
                            "no_zip_path": {"multipliers": v1["conformal_quantiles"]}},
            "support_policy": {**sp, "evidence": "validation split only; error = worse of the no-ZIP and with-ZIP paths",
                               "status_counts": status_counts.to_dict("records")},
            "evaluation_test_split_reporting_only": ev,
            "backup": {"path": "models/backups/v1_no_zip", "sha256": recorded}}
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    after = sha256(MODELS / "house_price_model.joblib")
    assert after == recorded["house_price_model.joblib"], "v1 model file changed"
    integ = {"version": VERSION, "reproduction_check": {"validation_MAE": got["MAE"], "recorded_MAE": rec["MAE"]},
             "calibration": meta["calibration"], "support_transitions": trans.to_dict("records"),
             "status_counts": status_counts.to_dict("records"), "test_reporting": ev}
    (REPORTS / "zip_integration.json").write_text(json.dumps(integ, indent=2, default=float), encoding="utf-8")
    write_report_section(integ, trans, status_counts, ev, q, cal_rows, cand, down_reasons)
    log("v2 artifacts written: models/zip_model.joblib, zip_city_map.csv, location_support.csv, feature_schema.json, model_metadata.json")


def write_report_section(integ, trans, status_counts, ev, q, cal_rows, cand, down_reasons) -> None:
    def cov_table(mode, lvl, group):
        rows = [r for r in ev[mode]["coverage"][lvl] if r["group"] == group]
        return pd.DataFrame(rows)[["value", "n", "coverage", "median_width"]].rename(columns={"value": group})

    def md(d):
        def f(v):
            if isinstance(v, (int, np.integer)):
                return f"{v:,}"
            if isinstance(v, float):
                return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"
            return str(v)
        return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                         + ["| " + " | ".join(f(x) for x in r) + " |" for r in d.itertuples(index=False)])

    def band_pair(lvl):
        a, b = cov_table("no_zip", lvl, "price_band").set_index("price_band"), cov_table("with_zip", lvl, "price_band").set_index("price_band")
        out = pd.DataFrame({"rows": a.n, "coverage no ZIP": a.coverage, "coverage with ZIP": b.coverage,
                            "median width no ZIP": a.median_width, "median width with ZIP": b.median_width})
        return out.reindex([x for x in BANDS[1] if x in out.index]).reset_index().rename(columns={"index": "price_band"})

    mz, mn = ev["with_zip"]["metrics"], ev["no_zip"]["metrics"]
    L = ["", "## Integration into production (model version 2.0.0-zip-hybrid)", "",
         "Done by `python src/integrate_zip_model.py` after approval. Results: `reports/zip_integration.json`.", "",
         "### Decisions", "",
         "- **Two request modes.** *Without a ZIP* the app uses the v1 model and the v1 calibrated ranges, unchanged, so "
         "existing behaviour is identical. *With a ZIP* the hybrid is used: the ZIP model when the ZIP has at least "
         f"{cand['hybrid_min_zip_rows']} training rows, otherwise the v1 model as a city-level fallback (stated to the user).",
         "- **ZIP validation.** ZIPs are 5-character strings (leading zeros kept). A ZIP must appear with the chosen state and "
         "city in `models/zip_city_map.csv`, built from the cleaned dataset's (state, city, ZIP) combinations (the same "
         "22,128 combinations as the warehouse location dimension). Malformed or mismatched ZIPs are rejected; no other "
         "location is ever substituted. Non-string ZIPs are rejected at the API so leading zeros cannot be lost silently.",
         "- **No retraining.** The experiment's ZIP model was reused; the integration first checks it reproduces the recorded "
         f"validation MAE (${integ['reproduction_check']['validation_MAE']:,.2f}). The encoder code moved to "
         "`src/model_common.py` unchanged so production does not depend on the experiment script.",
         "- **Split discipline.** Spread model: validation split. Multipliers: calibration split. Support policy: validation "
         "split. Test split: reported below only; nothing computed on it was used for any choice.", "",
         "### Recalibrated ranges for the ZIP path", "",
         f"Multipliers from the calibration split, per route (rows: ZIP model {cal_rows['zip']['rows']:,}, fallback "
         f"{cal_rows['fallback']['rows']:,}; a route with fewer than {MIN_ROUTE_CAL_ROWS} rows would use the pooled value): "
         f"ZIP model 80% {q['zip']['0.8']:.3f} / 90% {q['zip']['0.9']:.3f}; fallback 80% {q['fallback']['0.8']:.3f} / 90% {q['fallback']['0.9']:.3f}.", "",
         "Test-split check (reporting only; every test home evaluated with its own ZIP for the 'with ZIP' column):", "",
         md(pd.DataFrame([{"mode": m, "MAE": ev[m]["metrics"]["MAE"], "MdAPE": ev[m]["metrics"]["MdAPE"], "R2": ev[m]["metrics"]["R2"],
                           "coverage 80%": ev[m]["coverage"]["0.8"][0]["coverage"], "median width 80%": ev[m]["coverage"]["0.8"][0]["median_width"],
                           "coverage 90%": ev[m]["coverage"]["0.9"][0]["coverage"], "median width 90%": ev[m]["coverage"]["0.9"][0]["median_width"]}
                          for m in ("no_zip", "with_zip")])), "",
         "80% coverage by recorded-price band:", "", md(band_pair("0.8")), "",
         "90% coverage by recorded-price band:", "", md(band_pair("0.9")), "",
         "With-ZIP coverage by ZIP route:", "", md(cov_table("with_zip", "0.8", "zip_route")), "",
         "Coverage by location support status (v2 statuses):", "",
         md(pd.concat([cov_table(m, "0.8", "support_status").assign(mode=m) for m in ("no_zip", "with_zip")])), "",
         "### Location support policy v2", "",
         "Thresholds unchanged from v1 (supported: at least 100 training rows, at least 10 held-out rows, held-out median "
         "error at most 25%; limited: at least 30 training rows; otherwise unsupported, no estimate). Changes: the held-out "
         "evidence is now the **validation split only** (v1 used validation + calibration), and the error used is the "
         "**worse of the no-ZIP and with-ZIP paths**, so a city is supported only if both request modes are reliable. "
         "Cities not in the table, and unsupported cities, receive no estimate with or without a ZIP.", "",
         md(status_counts), "", "Transitions from v1:", "", md(trans), "",
         f"Supported-to-limited moves by reason: {down_reasons}. Their no-ZIP estimates are unchanged but now carry a "
         "limited-coverage warning.", "",
         "### Limitations after integration", "",
         "- ZIP-path coverage is marginal over test homes that supplied their real ZIP; a user who picks a wrong (but listed) "
         "ZIP in the right city gets a confidently wrong ZIP-level estimate. The app labels the ZIP as optional and shows the route used.",
         "- Coverage still drifts for the cheapest and most expensive recorded prices in both modes (see the tables).",
         "- The ZIP-city mapping reflects this dataset only: real ZIPs missing from it are rejected rather than estimated, and "
         "the source's own ZIP-city pairs are not independently verified (for example, ZIP 01235, in western Massachusetts, "
         "is recorded with Boston).",
         "- The test split has now been looked at three times (Phase 4, ZIP experiment, this check); it was not used for choices "
         "here, but future work should rely on validation data or a new hold-out.",
         "- No real-browser or mobile visual check of the app was performed.", "",
         "### Rollback", "",
         "1. Stop the app.",
         "2. Copy the four files in `models/backups/v1_no_zip/` back into `models/` (`house_price_model.joblib`, "
         "`location_support.csv`, `feature_schema.json`, `model_metadata.json`); check them against `MANIFEST.json`.",
         "3. Delete or move `models/zip_model.joblib` and `models/zip_city_map.csv`. `src/predict.py` then disables ZIP input "
         "automatically and behaves exactly like v1.",
         "4. Run `python -m pytest tests -q` and start the app."]
    p = REPORTS / "zip_feature_experiment.md"
    text = p.read_text(encoding="utf-8")
    marker = "\n## Integration into production"
    if marker in text:
        text = text[:text.index(marker)]
    p.write_text(text.rstrip("\n") + "\n" + "\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
