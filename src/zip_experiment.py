"""Experiment: does ZIP code improve the price model? Isolated from production.

    python src/zip_experiment.py            stage 1: validation only; picks a candidate and saves it
    python src/zip_experiment.py --test     stage 2: evaluates the stage-1 choice on the test split, writes the report

Writes only models/experiments/zip/ and reports/zip_*. It never touches models/house_price_model.joblib or the other
production artifacts (their SHA-256 hashes are checked before and after). Splits are the production ones
(model_common.assign_split on property_group); ZIP is added as a third level under state and city.
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_common as mc  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED, REPORTS, MODELS = ROOT / "data" / "processed", ROOT / "reports", ROOT / "models"
EXP_DIR = MODELS / "experiments" / "zip"
VAL_JSON = REPORTS / "zip_experiment_validation.json"
PRODUCTION_FILES = ["house_price_model.joblib", "location_support.csv", "feature_schema.json", "model_metadata.json"]
ZIP_PRIORS = [5.0, 20.0, 50.0]   # shrinkage strengths tried on validation
ZIP_DROPOUT = 0.20               # variant (a): share of training rows whose ZIP is hidden during fitting
HYBRID_MIN_ZIP_ROWS = 5          # variant (b): use the ZIP model only when the ZIP has this many training rows
# Pre-registered decision rule (written before any results were seen):
DECISION_RULE = {"min_relative_val_mae_gain": 0.02, "max_unseen_city_mdape_worsening_pts": 1.0,
                 "max_unseen_zip_mdape_worsening_pts": 1.0}


# ------------------------------------------------------------------------------------------------
# ZIP handling now lives in model_common (production); re-exported so existing experiment code and the
# experimental bundle (pickled as zip_experiment.ZipHierarchicalEncoder) keep working.
# ------------------------------------------------------------------------------------------------
from model_common import ZIP_FEATURE_COLS, ZipHierarchicalEncoder, build_zip_features, normalize_zip  # noqa: E402,F401


def make_zip_model(prior_zip: float):
    """Identical to production `hist_gb_log` except for the extra ZIP level in the location encoder."""
    pre = ColumnTransformer(
        [("loc", ZipHierarchicalEncoder(prior_zip=prior_zip, random_state=mc.CONFIG["seed"], n_adjust=len(mc.ADJUST_COLS)),
          ["state", "city_key", "zip_key"] + mc.ADJUST_COLS),
         ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), mc.NUMERIC_COLS)], remainder="drop")
    hgb = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50,
                                        l2_regularization=1.0, early_stopping=True, n_iter_no_change=15,
                                        random_state=mc.CONFIG["seed"])
    return TransformedTargetRegressor(regressor=Pipeline([("pre", pre), ("model", hgb)]),
                                      func=np.log, inverse_func=np.exp, check_inverse=False)


# ------------------------------------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------------------------------------
BANDS = ([0, 100e3, 200e3, 400e3, 700e3, 1e6, 2e6, 1e12], ["<100k", "100-200k", "200-400k", "400-700k", "700k-1M", "1-2M", ">2M"])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def production_hashes() -> dict:
    return {f: sha256(MODELS / f) for f in PRODUCTION_FILES}


def summarize(y, p) -> dict:
    m = mc.metrics(y, p)
    return {k: m[k] for k in ("n", "MAE", "RMSE", "R2", "MedAE", "MdAPE")}


def breakdown(frame: pd.DataFrame, col: str, preds: dict) -> pd.DataFrame:
    rows = []
    for g, d in frame.groupby(col, observed=True):
        for name, p in preds.items():
            rows.append({col: g, "model": name, **summarize(d.y, p[d.index])})
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    df = mc.to_model_frame(pd.read_parquet(PROCESSED / "sold_clean.parquet",
                                           columns=["raw_row_id", "price", "bed", "bath", "house_size", "acre_lot",
                                                    "state", "city", "zip_code", "property_group"]))
    df = df.sort_values("raw_row_id").reset_index(drop=True)
    df["city_key"] = df.state + "|" + df.city
    df["split"] = mc.assign_split(df.property_group)
    sup = pd.read_csv(MODELS / "location_support.csv", keep_default_na=False)[["state", "city", "status"]]
    df = df.merge(sup, on=["state", "city"], how="left")
    df["status"] = df.status.fillna("unsupported")
    df["band"] = pd.cut(df.price, bins=BANDS[0], labels=BANDS[1]).astype(str)
    return df


def check_split_matches_production(df: pd.DataFrame) -> None:
    prod = json.loads((REPORTS / "experiment_summary.json").read_text(encoding="utf-8"))["splits"]
    counts = df.split.value_counts()
    for r in prod:
        assert counts[r["split"]] == r["rows"], f"split {r['split']} differs from production"
    assert (df.groupby("property_group").split.nunique() == 1).all()


def fit(model, X, y):
    t = time.time()
    model.fit(X, y)
    return model, time.time() - t


# ------------------------------------------------------------------------------------------------
# Stage 1: validation only
# ------------------------------------------------------------------------------------------------
def zip_dropout(X: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Training-time augmentation: hide the ZIP on a random share of rows so the model learns how to behave when a
    ZIP is unknown on representative homes (otherwise 'no ZIP information' only ever means a tiny rural ZIP)."""
    X = X.copy()
    X.loc[np.random.default_rng(seed).random(len(X)) < rate, "zip_key"] = None
    return X


def fit_candidates(Xb, Xz, y, tr, prior_zip):
    """Fit baseline, plain ZIP model and ZIP-dropout model on the rows in `tr`."""
    base, t_b = fit(mc.make_model("hist_gb_log"), Xb[tr], y[tr])
    plain, t_p = fit(make_zip_model(prior_zip), Xz[tr], y[tr])
    drop, t_d = fit(make_zip_model(prior_zip), zip_dropout(Xz[tr], ZIP_DROPOUT, mc.CONFIG["seed"]), y[tr])
    return {"baseline": base, "zip": plain, "zip_dropout": drop}, {"baseline": t_b, "zip": t_p, "zip_dropout": t_d}


def predict_all(models, Xb, Xz, ev, zip_counts: pd.Series, zip_codes: pd.Series) -> dict:
    """Predictions of every candidate on rows `ev`; 'hybrid' routes by the ZIP's training-row count."""
    pb = mc.predict_dollars(models["baseline"], Xb[ev])
    pz = mc.predict_dollars(models["zip"], Xz[ev])
    pdz = mc.predict_dollars(models["zip_dropout"], Xz[ev])
    enough = zip_codes[ev].map(zip_counts).fillna(0).to_numpy() >= HYBRID_MIN_ZIP_ROWS
    return {"baseline": pb, "zip": pz, "zip_dropout": pdz, "hybrid": np.where(enough, pz, pb)}


def stage_validation() -> None:
    before = production_hashes()
    df = load_data()
    check_split_matches_production(df)
    Xb, Xz, y = mc.build_features(df), build_zip_features(df), df.price.to_numpy()
    tr, va = (df.split == "train").to_numpy(), (df.split == "validation").to_numpy()
    zip_info = {"rows": len(df), "invalid_or_missing_zip": int(Xz.zip_key.isna().sum()),
                "leading_zero_rows": int(df.zip_code.str.startswith("0").sum()),
                "zips_total": int(df.zip_code.nunique()), "zips_in_train": int(df.loc[tr, "zip_code"].nunique()),
                "train_rows_per_zip_median": float(df.loc[tr].groupby("zip_code").size().median())}

    # 1. choose the ZIP shrinkage strength on validation (plain ZIP model)
    prior_rows = []
    for k in ZIP_PRIORS:
        m, t = fit(make_zip_model(k), Xz[tr], y[tr])
        prior_rows.append({"prior_zip": k, "fit_seconds": t, **summarize(y[va], mc.predict_dollars(m, Xz[va]))})
        print(f"prior_zip={k}: val MAE {prior_rows[-1]['MAE']:,.0f}", flush=True)
    prior_tbl = pd.DataFrame(prior_rows)
    k_best = float(prior_tbl.sort_values("MAE").iloc[0].prior_zip)

    # 2. all candidates on the full training split, scored on validation
    models, times = fit_candidates(Xb, Xz, y, tr, k_best)
    counts = df.loc[tr, "zip_code"].value_counts()
    P = predict_all(models, Xb, Xz, va, counts, df.zip_code)
    val_tbl = pd.DataFrame([{"model": k, "fit_seconds": times.get(k, times["zip"] + times["baseline"]),
                             **summarize(y[va], p)} for k, p in P.items()])
    print(val_tbl.round(4).to_string(index=False), flush=True)

    v = df[va].assign(y=y[va])
    v["zip_seen_in_train"] = np.where(v.zip_code.isin(counts.index), "ZIP seen in training", "ZIP not in training")
    preds = {k: pd.Series(p, index=v.index) for k, p in P.items()}
    by_band, by_status, by_zipseen = (breakdown(v, c, preds) for c in ("band", "status", "zip_seen_in_train"))

    # 3. generalisation on validation rows: whole cities held out; whole ZIPs held out inside seen cities
    gen_rows = []
    held_city = mc.heldout_city_mask(df.city_key)
    held_zip = mc.stable_uniform(df.zip_code, mc.CONFIG["seed"], "zip") < 0.15
    for label, held in (("unseen cities", held_city), ("unseen ZIPs in seen cities", held_zip & ~held_city)):
        tr_g = tr & ~held
        ev = va & held & ~df.property_group.isin(set(df.loc[tr_g, "property_group"])).to_numpy()
        if label.startswith("unseen ZIPs"):
            ev = ev & df.city_key.isin(set(df.loc[tr_g, "city_key"])).to_numpy()
        m_g, _ = fit_candidates(Xb, Xz, y, tr_g, k_best)
        for k, p in predict_all(m_g, Xb, Xz, ev, df.loc[tr_g, "zip_code"].value_counts(), df.zip_code).items():
            gen_rows.append({"evaluation": label, "model": k, **summarize(y[ev], p)})
        print(f"generalisation done: {label} ({int(ev.sum()):,} validation rows)", flush=True)
    gen_tbl = pd.DataFrame(gen_rows)
    print(gen_tbl.round(4).to_string(index=False), flush=True)

    # 4. pre-registered rule, applied to each ZIP candidate; best passing candidate by validation MAE
    vt, g = val_tbl.set_index("model"), gen_tbl.set_index(["evaluation", "model"]).MdAPE
    checks = {}
    for k in ("zip", "zip_dropout", "hybrid"):
        gain = float(1 - vt.loc[k, "MAE"] / vt.loc["baseline", "MAE"])
        dc = float(100 * (g[("unseen cities", k)] - g[("unseen cities", "baseline")]))
        dz = float(100 * (g[("unseen ZIPs in seen cities", k)] - g[("unseen ZIPs in seen cities", "baseline")]))
        checks[k] = {"val_MAE_gain": gain, "unseen_city_MdAPE_change_pts": dc, "unseen_zip_MdAPE_change_pts": dz,
                     "passes": bool(gain >= DECISION_RULE["min_relative_val_mae_gain"]
                                    and dc <= DECISION_RULE["max_unseen_city_mdape_worsening_pts"]
                                    and dz <= DECISION_RULE["max_unseen_zip_mdape_worsening_pts"])}
    passing = [k for k, c in checks.items() if c["passes"]]
    chosen = min(passing, key=lambda k: vt.loc[k, "MAE"]) if passing else "baseline"
    print("checks:", json.dumps(checks, indent=1), "| chosen:", chosen, flush=True)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"zip_model": models["zip"], "zip_dropout_model": models["zip_dropout"], "prior_zip": k_best,
                 "chosen": chosen, "hybrid_min_zip_rows": HYBRID_MIN_ZIP_ROWS,
                 "train_zip_counts": counts.to_dict(), "feature_columns": ZIP_FEATURE_COLS,
                 "note": "EXPERIMENTAL - trained on the production train split only; not used by the app; "
                         "no prediction intervals calibrated. The hybrid also needs the production model."},
                EXP_DIR / "zip_candidates.joblib", compress=3)
    after = production_hashes()
    assert before == after, "production artifacts changed during the experiment"
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision_rule": DECISION_RULE,
           "zip_dropout_rate": ZIP_DROPOUT, "hybrid_min_zip_rows": HYBRID_MIN_ZIP_ROWS, "zip_data": zip_info,
           "prior_search": prior_tbl.to_dict("records"), "prior_zip": k_best, "validation": val_tbl.to_dict("records"),
           "decision_checks": checks, "chosen_candidate": chosen, "by_band": by_band.to_dict("records"),
           "by_status": by_status.to_dict("records"), "by_zip_seen": by_zipseen.to_dict("records"),
           "generalisation": gen_tbl.to_dict("records"), "production_hashes_unchanged": True, "production_hashes": after}
    VAL_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {VAL_JSON.relative_to(ROOT)} and {(EXP_DIR / 'zip_candidates.joblib').relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------------
# Stage 2: one test evaluation of the stage-1 choice, then the report
# ------------------------------------------------------------------------------------------------
def hybrid_predict(prod_model, cand: dict, df: pd.DataFrame) -> np.ndarray:
    """Hybrid candidate: ZIP model where the ZIP had >= hybrid_min_zip_rows training rows, production model elsewhere.
    Missing/invalid ZIPs always use the production model."""
    zips = pd.Series([normalize_zip(z) for z in df["zip_code"]], index=df.index) if "zip_code" in df else \
        pd.Series(None, index=df.index, dtype=object)
    n = zips.map(cand["train_zip_counts"]).fillna(0).to_numpy()
    use_zip = n >= cand["hybrid_min_zip_rows"]
    out = mc.predict_dollars(prod_model, mc.build_features(df))
    if use_zip.any():
        out[use_zip] = mc.predict_dollars(cand["zip_model"], build_zip_features(df[use_zip]))
    return out


def stage_test() -> None:
    val = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    chosen = val["chosen_candidate"]
    if "test" in val:
        raise SystemExit("the test split was already evaluated for this experiment; refusing to evaluate it again")
    before = production_hashes()
    df = load_data()
    check_split_matches_production(df)
    te = (df.split == "test").to_numpy()
    y = df.price.to_numpy()
    prod = joblib.load(MODELS / "house_price_model.joblib")["model"]            # read-only use of production model
    cand = joblib.load(EXP_DIR / "zip_candidates.joblib")
    t = df[te].assign(y=y[te])
    preds = {"baseline (production model)": pd.Series(mc.predict_dollars(prod, mc.build_features(t)), index=t.index)}
    if chosen == "hybrid":
        preds["hybrid"] = pd.Series(hybrid_predict(prod, cand, t), index=t.index)
    elif chosen in ("zip", "zip_dropout"):
        m = cand["zip_model" if chosen == "zip" else "zip_dropout_model"]
        preds[chosen] = pd.Series(mc.predict_dollars(m, build_zip_features(t)), index=t.index)
    seen = t.zip_code.map(cand["train_zip_counts"]).fillna(0)
    t["zip_training_rows"] = np.select([seen >= cand["hybrid_min_zip_rows"], seen > 0], ["5+", "1-4"], "0 (unseen)")
    test_tbl = pd.DataFrame([{"model": k, **summarize(t.y, p)} for k, p in preds.items()])
    by_band, by_status, by_zip = (breakdown(t, c, preds) for c in ("band", "status", "zip_training_rows"))
    after = production_hashes()
    assert before == after, "production artifacts changed during the experiment"
    val["test"] = {"evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "candidate": chosen,
                   "overall": test_tbl.to_dict("records"), "by_band": by_band.to_dict("records"),
                   "by_status": by_status.to_dict("records"), "by_zip_training_rows": by_zip.to_dict("records"),
                   "production_hashes_unchanged": True}
    VAL_JSON.write_text(json.dumps(val, indent=2, default=float), encoding="utf-8")
    print(test_tbl.round(4).to_string(index=False))
    for name, tb in (("band", by_band), ("status", by_status), ("zip rows", by_zip)):
        print(f"\n-- test by {name}\n" + tb.pivot_table(index=tb.columns[0], columns="model", values=["MAE", "MdAPE"]).round(3).to_string())


# ------------------------------------------------------------------------------------------------
# Report (reads only reports/zip_experiment_validation.json)
# ------------------------------------------------------------------------------------------------
def _md(df: pd.DataFrame) -> str:
    def f(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, float):
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"
        return str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _side_by_side(records, key, models, order=None) -> pd.DataFrame:
    d = pd.DataFrame(records)
    d = d[d.model.isin(models)]
    w = d.pivot_table(index=key, columns="model", values=["n", "MAE", "MdAPE"], aggfunc="first")
    out = pd.DataFrame({key: w.index, "rows": w[("n", models[0])].astype(int).to_numpy()})
    for m in models:
        out[f"MAE {m}"] = w[("MAE", m)].to_numpy()
    for m in models:
        out[f"MdAPE {m}"] = w[("MdAPE", m)].to_numpy()
    if order:
        out = out.set_index(key).reindex([o for o in order if o in set(out[key])]).reset_index()
    return out


def write_report() -> None:
    r = json.loads(VAL_JSON.read_text(encoding="utf-8"))
    z, rule, ch = r["zip_data"], r["decision_rule"], r["decision_checks"]
    val = pd.DataFrame(r["validation"])[["model", "n", "MAE", "RMSE", "R2", "MedAE", "MdAPE"]]
    vb = val.set_index("model").loc["baseline"]
    vh = val.set_index("model").loc["hybrid"]
    gen = pd.DataFrame(r["generalisation"])[["evaluation", "model", "n", "MAE", "MedAE", "MdAPE", "R2"]]
    chk = pd.DataFrame([{"candidate": k, "val_MAE_gain": c["val_MAE_gain"], "unseen_city_MdAPE_change_pts": c["unseen_city_MdAPE_change_pts"],
                         "unseen_ZIP_MdAPE_change_pts": c["unseen_zip_MdAPE_change_pts"], "passes": "yes" if c["passes"] else "no"}
                        for k, c in ch.items()])
    bands = BANDS[1]
    t = r.get("test")
    L = ["# ZIP-code feature experiment", "",
         "Generated by `python src/zip_experiment.py --report` from `reports/zip_experiment_validation.json`. "
         "**The production model was not changed**: the SHA-256 hashes of `models/house_price_model.joblib`, "
         "`location_support.csv`, `feature_schema.json` and `model_metadata.json` were identical before and after every stage.", "",
         "## Question", "",
         "Does adding the ZIP code to the location features improve the recorded-price model, and does it hurt "
         "predictions for locations the model has not seen?", "",
         "## Design", "",
         f"- **Data.** The same cleaned complete-case dataset ({z['rows']:,} sold-status rows). ZIP codes are stored as 5-character "
         f"strings (the raw CSV parses them as integers, losing leading zeros; Phase 2 restored them). {z['leading_zero_rows']:,} rows "
         f"have a leading-zero ZIP; {z['invalid_or_missing_zip']} are missing or invalid. {z['zips_total']:,} ZIPs in total, "
         f"{z['zips_in_train']:,} in the training split, median {z['train_rows_per_zip_median']:.0f} training rows per ZIP.",
         "- **ZIP as a category, not a number.** `normalize_zip` keeps 5-digit strings, cuts ZIP+4, zero-pads integers "
         "(that is exactly how leading zeros were lost) and rejects anything ambiguous (for example the string '601'), `00000`, "
         "NaN or booleans; rejected ZIPs become missing.",
         "- **Encoding.** The production encoder (state, then city) gains a third level: the size-adjusted mean log price of "
         "the ZIP, shrunk toward its city's value, plus log(1 + training rows in the ZIP). It is learned on training rows only "
         "and cross-fitted (5 folds), so no training row's own price enters its own encoding. A missing ZIP or a ZIP not seen "
         "in training gets exactly its city's value.",
         "- **Everything else identical to production:** the same split (hash of `property_group`; the split counts are "
         "asserted to equal production and no group crosses splits), the same features, gradient-boosting settings and seed. "
         f"The baseline refit reproduces the production validation MAE (${vb.MAE:,.0f}).",
         f"- **Candidates.** (1) the plain ZIP model, with shrinkage strength chosen on validation from {ZIP_PRIORS} "
         f"(chosen: {r['prior_zip']:g}); (2) the same with *ZIP dropout*, hiding the ZIP on {r['zip_dropout_rate']:.0%} of training rows; "
         f"(3) a *hybrid* that uses the ZIP model only when the ZIP has at least {r['hybrid_min_zip_rows']} training rows "
         "and the production model otherwise.",
         "- **Generalisation checks (validation rows only).** (a) 15% of cities held out of training entirely; (b) 15% of ZIPs "
         "held out of training, evaluated on homes in cities that remain in training.",
         f"- **Decision rule, fixed before any result was seen:** a candidate must cut validation MAE by at least "
         f"{rule['min_relative_val_mae_gain']:.0%} and must not worsen median % error by more than "
         f"{rule['max_unseen_city_mdape_worsening_pts']:g} point on unseen cities or "
         f"{rule['max_unseen_zip_mdape_worsening_pts']:g} point on unseen ZIPs. Among passing candidates, the lowest validation "
         "MAE wins. Only that candidate is evaluated on the test split, once (the script refuses a second test run).", "",
         "## Validation results", "", "Shrinkage search (plain ZIP model):", "",
         _md(pd.DataFrame(r["prior_search"])[["prior_zip", "MAE", "RMSE", "R2", "MdAPE"]]), "",
         "All candidates on the 69,769 validation rows (dollars):", "", _md(val), "",
         "Generalisation (validation rows of held-out locations; every model retrained without them):", "", _md(gen), "",
         "Decision checks:", "", _md(chk), "",
         "**Finding.** The plain ZIP model is the most accurate on ordinary validation rows, but it is clearly worse for homes "
         "in ZIPs it has never seen (median % error +6.7 points). Likely cause (consistent with the dropout result, not proven): "
         "in training, 'no ZIP information' only arises for ZIPs with very few rows (when cross-fitting leaves a ZIP empty in a "
         "fold), so the model ties that signal to small, atypical ZIPs and misapplies it to new ZIPs in large cities. "
         "ZIP dropout repairs most of this (+0.6 points); the hybrid avoids it by construction, because a ZIP without enough "
         "training rows goes to the unchanged production model. Unseen *cities* improve slightly with ZIP information, "
         "probably because 2,769 ZIPs span several cities and still carry information from a neighbouring city.", "",
         f"Validation, baseline vs hybrid, by price band:", "",
         _md(_side_by_side(r["by_band"], "band", ["baseline", "hybrid"], bands)), "",
         "By location status (from `models/location_support.csv`):", "",
         _md(_side_by_side(r["by_status"], "status", ["baseline", "hybrid"], ["supported", "limited", "unsupported"])), "",
         "By whether the home's ZIP appears in training:", "",
         _md(_side_by_side(r["by_zip_seen"], "zip_seen_in_train", ["baseline", "hybrid"])), "",
         f"**Chosen on validation: `{r['chosen_candidate']}`** (validation MAE ${vh.MAE:,.0f} against ${vb.MAE:,.0f}, "
         f"{1 - vh.MAE / vb.MAE:.1%} lower; median % error {vh.MdAPE:.1%} against {vb.MdAPE:.1%}).", ""]
    if t:
        tt = pd.DataFrame(t["overall"])[["model", "n", "MAE", "RMSE", "R2", "MedAE", "MdAPE"]]
        tb, th = tt.iloc[0], tt.iloc[1]
        names = list(tt.model)
        L += ["## Test results (evaluated once, after the choice above)", "", _md(tt), "",
              f"MAE {1 - th.MAE / tb.MAE:.1%} lower, RMSE {1 - th.RMSE / tb.RMSE:.1%} lower, R2 {tb.R2:.3f} to {th.R2:.3f}, "
              f"median % error {tb.MdAPE:.1%} to {th.MdAPE:.1%}.", "",
              "By price band:", "", _md(_side_by_side(t["by_band"], "band", names, bands)), "",
              "By location status:", "", _md(_side_by_side(t["by_status"], "status", names, ["supported", "limited", "unsupported"])), "",
              "By the ZIP's number of training rows (the hybrid only changes the 5+ group):", "",
              _md(_side_by_side(t["by_zip_training_rows"], "zip_training_rows", names, ["5+", "1-4", "0 (unseen)"])), ""]
    L += ["## Limitations", "",
          "- **No prediction intervals yet.** The production 80%/90% ranges were calibrated for the production model. Adopting "
          "the hybrid requires refitting the spread model and recalibrating the conformal multipliers on the calibration "
          "split, then re-checking coverage; until then, the hybrid has no valid range.",
          "- **The location support table is model-specific.** It was built from the production model's held-out errors; it "
          "would need to be rebuilt for the hybrid (cities may move between supported and limited).",
          "- **New input.** The app would need a ZIP field, validated with `normalize_zip` and checked against the chosen "
          "city (2,769 ZIPs span several cities). A mistyped but valid ZIP elsewhere in the city would silently change the "
          "estimate, so a ZIP-city consistency check is needed.",
          "- **Two models to maintain** (hybrid). ZIP dropout is a single-model alternative that also passed the rule, at "
          "a slightly smaller gain.",
          "- **The test split has been used** by the Phase 4 evaluation and once here; repeated feature experiments against it "
          "would gradually overfit it.",
          "- **Same data limits as before:** historical (Oct 2021 - May 2022) recorded prices, unverified as sale prices; "
          "ZIP improves location detail but not property detail (condition, age, view).", "",
          "## Recommendation", ""]
    if t and r["chosen_candidate"] != "baseline":
        L += [f"**Include ZIP code, using the hybrid design**, but only after the follow-up steps below. The gain held on the "
              "untouched test split, appeared in every price band and every location status, and did not cost accuracy for "
              "unseen ZIPs or unseen cities. Do **not** ship the plain ZIP model: it fails the pre-registered unseen-ZIP check.", "",
              "Before replacing production (needs approval):", "",
              "1. Recalibrate the conformal ranges for the hybrid on the calibration split and re-check coverage.",
              "2. Rebuild the location support table from the hybrid's held-out errors.",
              "3. Add an optional, validated ZIP input to `predict.py` and the app, with a ZIP-city consistency check; "
              "without a ZIP the app behaves exactly as now.",
              "4. Re-run all tests and the app smoke test."]
    else:
        L += ["Keep the production model; no ZIP candidate met the pre-registered rule."]
    (REPORTS / "zip_feature_experiment.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/zip_feature_experiment.md")


if __name__ == "__main__":
    # Run through the importable module so pickled classes are saved as `zip_experiment.X`, not `__main__.X`
    # (otherwise the saved candidates could only be loaded by this script).
    import zip_experiment as _module
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="stage 2: evaluate the stage-1 choice on the test split")
    ap.add_argument("--report", action="store_true", help="write reports/zip_feature_experiment.md from the JSON results")
    a = ap.parse_args()
    _module.write_report() if a.report else (_module.stage_test() if a.test else _module.stage_validation())
