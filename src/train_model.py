"""Phase 4: train, compare and evaluate price models, build location support and prediction intervals.

    python src/train_model.py              full run (uses the test split once, after model selection)
    python src/train_model.py --no-test    development run: never touches the test split, writes nothing

Data come from the PostgreSQL warehouse (dw.fact_property_listing joined to its dimensions) and are cross-checked
against data/processed/sold_clean.parquet; if the database is unreachable the Parquet file is used instead.
Splits are assigned by hashing `property_group`, so all rows of one property land in the same split.

Outputs: models/*, reports/model_comparison.md, location_evaluation.csv, experiment_summary.json, figures/*.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

import model_common as mc

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED, REPORTS, MODELS = ROOT / "data" / "processed", ROOT / "reports", ROOT / "models"
FIGS = REPORTS / "figures"
CANDIDATES = ["dummy_median", "ridge_log", "random_forest_log", "hist_gb_log", "hist_gb_raw"]
SPLITS = ["train", "validation", "calibration", "test"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
def load_complete_cases() -> tuple[pd.DataFrame, dict]:
    """Complete-case modelling frame from the warehouse, cross-checked against the Parquet file."""
    parquet = pd.read_parquet(PROCESSED / "sold_clean.parquet").sort_values("raw_row_id").reset_index(drop=True)
    info = {"source": "parquet", "warehouse_matches_parquet": None}
    try:
        from sqlalchemy import text
        import warehouse_etl as w
        q = """SELECT f.raw_row_id, f.price::float8 AS price, p.bedrooms AS bed, p.bathrooms::float8 AS bath,
                      f.house_size_sqft AS house_size, f.acre_lot::float8 AS acre_lot, l.state, l.city,
                      f.property_group
               FROM dw.fact_property_listing f JOIN dw.dim_location l USING (location_key)
               JOIN dw.dim_property p USING (property_dim_key) ORDER BY f.raw_row_id"""
        db = pd.read_sql(text(q), w.get_engine())
        same = (len(db) == len(parquet) and (db.raw_row_id.values == parquet.raw_row_id.values).all()
                and (db.price.values == parquet.price.values).all() and (db.bed.values == parquet.bed.values).all()
                and (db.bath.values == parquet.bath.values).all()
                and (db.house_size.values == parquet.house_size.values).all()
                and np.array_equal(db.acre_lot.values, parquet.acre_lot.values, equal_nan=True)
                and (db.state.values == parquet.state.values).all() and (db.city.values == parquet.city.values).all()
                and (db.property_group.values == parquet.property_group.values).all())
        info.update(source="postgresql warehouse (dw.fact_property_listing)", warehouse_matches_parquet=bool(same),
                    warehouse_rows=len(db))
        if not same:
            raise SystemExit("warehouse and parquet disagree; investigate before modelling")
        df = db
    except SystemExit:
        raise
    except Exception as exc:  # database unreachable: fall back, and say so in the report
        info["warehouse_error"] = f"{type(exc).__name__}"
        df = parquet[["raw_row_id", "price", "bed", "bath", "house_size", "acre_lot", "state", "city", "property_group"]]
    return df, info


def group_reliability(full: pd.DataFrame) -> dict:
    """Quantify how trustworthy the (street, zip) property key is."""
    pq = pd.read_parquet(PROCESSED / "sold_clean.parquet", columns=["property_key", "property_group", "repeated_property",
                                                                     "bed", "bath", "house_size", "city", "street", "zip_code"])
    size = pq.groupby("property_group").size()
    rep = pq[pq.repeated_property]
    r = rep.groupby("property_group").agg(nb=("bed", "nunique"), nba=("bath", "nunique"),
                                          ns=("house_size", "nunique"), nc=("city", "nunique"))
    return {"rows": len(pq), "groups": int(len(size)), "rows_without_property_key": int(pq.property_key.isna().sum()),
            "singleton_groups": int((size == 1).sum()), "groups_with_multiple_rows": int((size > 1).sum()),
            "rows_in_multi_row_groups": int(size[size > 1].sum()), "largest_group": int(size.max()),
            "multi_row_groups_whose_rows_differ_in_bed_bath_or_size": int(((r.nb > 1) | (r.nba > 1) | (r.ns > 1)).sum()),
            "multi_row_groups_spanning_cities": int((r.nc > 1).sum()),
            "street_codes_shared_by_multiple_zips": int((pq.dropna(subset=["street"]).groupby("street").zip_code.nunique() > 1).sum()),
            "interpretation": "Rows with no key become singleton groups (cannot be linked to other rows). Where the key repeats, "
                              "rows often differ in bed/bath/size, so the key can merge different units (conservative for "
                              "splitting) but cannot detect the same property listed under a missing or different key."}


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
def fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "n/a"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"
    return str(v)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    d = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "|".join("---" for _ in d.columns) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def group_metrics(df: pd.DataFrame, by, min_n: int) -> pd.DataFrame:
    """df needs columns y, p. Returns n, MAE, MedAE, MdAPE, bias(median log ratio) per group with n >= min_n."""
    d = df.assign(ae=(df.p - df.y).abs(), ape=(df.p - df.y).abs() / df.y, lr=np.log(df.p / df.y))
    g = d.groupby(by).agg(n=("y", "size"), MAE=("ae", "mean"), MedAE=("ae", "median"), MdAPE=("ape", "median"),
                          median_log_bias=("lr", "median"), median_price=("y", "median"))
    return g[g.n >= min_n]


def fit_timed(name: str, X, y):
    m = mc.make_model(name)
    t = time.time()
    m.fit(X, y)
    return m, time.time() - t


def model_size_mb(m) -> float:
    buf = io.BytesIO()
    joblib.dump(m, buf)
    return buf.tell() / 1e6


# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true", help="development run: do not evaluate on the test split")
    args = ap.parse_args()
    test_ok = not args.no_test
    summary: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": mc.CONFIG,
                     "test_split_evaluated": test_ok}

    # ---- 1. data and splits ---------------------------------------------------------------------
    df, info = load_complete_cases()
    summary["data_source"] = info
    summary["property_group_reliability"] = group_reliability(df)
    summary["input_hashes"] = {f: sha256(PROCESSED / f) for f in ("sold_clean.parquet", "sold_incomplete_features.parquet")}
    df["city_key"] = df["state"] + "|" + df["city"]
    df["split"] = mc.assign_split(df["property_group"])
    assert (df.groupby("property_group")["split"].nunique() == 1).all(), "a property group spans several splits"
    split_tbl = df.groupby("split").agg(rows=("price", "size"), groups=("property_group", "nunique"),
                                        singleton_key_rows=("property_group", lambda s: int(s.str.startswith("row_").sum())),
                                        cities=("city_key", "nunique")).reindex(SPLITS)
    split_tbl["share_of_rows"] = split_tbl.rows / split_tbl.rows.sum()
    summary["splits"] = split_tbl.reset_index().to_dict("records")
    log(f"loaded {len(df):,} rows from {info['source']}; splits: " + ", ".join(f"{s}={n:,}" for s, n in split_tbl.rows.items()))

    X = mc.build_features(df)
    y = df.price.to_numpy()
    idx = {s: (df.split == s).to_numpy() for s in SPLITS}
    assert not set(mc.FEATURE_COLS) & {"price", "price_per_sqft"}, "target-derived feature"

    # ---- 2. candidate models on the validation split --------------------------------------------
    fitted, rows = {}, []
    for name in CANDIDATES:
        m, ft = fit_timed(name, X[idx["train"]], y[idx["train"]])
        t = time.time()
        p = mc.predict_dollars(m, X[idx["validation"]])
        row = mc.metrics(y[idx["validation"]], p)
        row.update(model=name, fit_seconds=ft, predict_seconds=time.time() - t, model_size_mb=model_size_mb(m))
        fitted[name] = m
        rows.append(row)
        log(f"validation {name:18s} MAE={row['MAE']:>9,.0f} RMSE={row['RMSE']:>9,.0f} R2={row['R2']:.3f} MdAPE={row['MdAPE']:.3f}")
    val_tbl = pd.DataFrame(rows).set_index("model")
    best = val_tbl.drop(index="dummy_median")["MAE"].idxmin()  # selection uses validation MAE only
    log(f"selected on validation MAE: {best}")
    summary["selected_model"] = best
    selected = fitted[best]
    inner = selected.regressor_ if hasattr(selected, "regressor_") else selected   # the fitted sklearn Pipeline
    final_est = inner.named_steps["model"]
    summary["selected_model_params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                        for k, v in final_est.get_params().items()}
    if hasattr(final_est, "n_iter_"):
        summary["selected_model_boosting_iterations"] = int(final_est.n_iter_)

    # ---- 3. dev-only analysis: held-out predictions on validation + calibration -------------------
    pre = inner.named_steps["pre"]
    P = {s: mc.predict_dollars(selected, X[idx[s]]) for s in SPLITS if s != "test"}
    dev = pd.concat([df.loc[idx[s], ["city_key", "state", "city", "price", "split"]].assign(p=P[s]) for s in ("validation", "calibration")])
    dev = dev.rename(columns={"price": "y"})
    n_train_city = df[idx["train"]].groupby("city_key").size()
    n_total_city = df.groupby("city_key").size()

    # support policy thresholds examined on dev held-out rows only
    dev["n_train"] = dev.city_key.map(n_train_city).fillna(0).astype(int)
    thr_rows = []
    for t in (30, 100, 500):
        for label, m_ in ((f"n_train >= {t}", dev.n_train >= t), (f"n_train < {t}", dev.n_train < t)):
            sub = dev[m_]
            if len(sub):
                thr_rows.append({"group": label, "heldout_rows": len(sub), "cities": sub.city_key.nunique(),
                                 "MdAPE": float((sub.p - sub.y).abs().div(sub.y).median()),
                                 "MAE": float((sub.p - sub.y).abs().mean()),
                                 "share_of_rows_over_0.5_APE": float(((sub.p - sub.y).abs() / sub.y > 0.5).mean())})
    thr_tbl = pd.DataFrame(thr_rows)
    log("threshold analysis (validation+calibration rows):\n" + thr_tbl.to_string(index=False))

    # ---- 4. conformal calibration (scale model on validation, quantiles on calibration) -----------
    def feats(split):
        Z = pre.transform(X[idx[split]])
        return np.column_stack([Z, np.log(P[split])])
    r = {s: np.log(y[idx[s]]) - np.log(P[s]) for s in ("validation", "calibration")}
    scale_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                                                random_state=mc.CONFIG["seed"]).fit(feats("validation"), np.abs(r["validation"]))
    SCALE_FLOOR = 0.03
    sig = lambda Z: np.maximum(scale_model.predict(Z), SCALE_FLOOR)
    sig_cal = sig(feats("calibration"))
    qs = {a: mc.conformal_quantile(np.abs(r["calibration"]) / sig_cal, a) for a in mc.CONFIG["conformal_alphas"]}
    q_const = {a: mc.conformal_quantile(np.abs(r["calibration"]), a) for a in mc.CONFIG["conformal_alphas"]}
    log(f"conformal quantiles (normalised): {qs}; constant: {q_const}")

    def interval(p, Z, a, constant=False):
        if constant:
            half = np.full(len(p), q_const[a])
        else:
            half = qs[a] * sig(Z)
        return p * np.exp(-half), p * np.exp(half)

    # support table (policy from dev rows only)
    sp = mc.CONFIG["support_policy"]
    dev["ape"] = (dev.p - dev.y).abs() / dev.y
    dev["ae"] = (dev.p - dev.y).abs()
    city = dev.groupby("city_key").agg(heldout_rows=("y", "size"), heldout_MdAPE=("ape", "median"), heldout_MAE=("ae", "mean"))
    loc = (df.groupby(["state", "city", "city_key"]).agg(n_total=("price", "size"), median_price=("price", "median")).reset_index()
           .merge(df.groupby(["city_key", "split"]).size().unstack(fill_value=0).reindex(columns=SPLITS, fill_value=0)
                  .add_prefix("n_").reset_index(), on="city_key", how="left")
           .merge(city.reset_index(), on="city_key", how="left"))
    loc["heldout_rows"] = loc.heldout_rows.fillna(0).astype(int)

    def classify(row):
        if row.n_train >= sp["supported_min_train_rows"] and row.heldout_rows >= sp["min_heldout_rows"] \
                and row.heldout_MdAPE <= sp["max_heldout_mdape"]:
            return "supported", "enough training rows and acceptable held-out error"
        if row.n_train >= sp["limited_min_train_rows"]:
            why = []
            if row.n_train < sp["supported_min_train_rows"]:
                why.append(f"only {row.n_train} training rows")
            if row.heldout_rows < sp["min_heldout_rows"]:
                why.append("too few held-out rows to verify accuracy")
            elif row.heldout_MdAPE > sp["max_heldout_mdape"]:
                why.append(f"held-out median error {row.heldout_MdAPE:.0%} exceeds {sp['max_heldout_mdape']:.0%}")
            return "limited", "; ".join(why)
        return "unsupported", f"only {row.n_train} training rows (< {sp['limited_min_train_rows']})"
    cls = loc.apply(classify, axis=1, result_type="expand")
    loc["status"], loc["reason"] = cls[0], cls[1]
    status_counts = loc.groupby("status").agg(cities=("city_key", "size"), rows=("n_total", "sum"))
    log("support status:\n" + status_counts.to_string())
    summary["support_status_counts"] = status_counts.reset_index().to_dict("records")
    summary["support_threshold_analysis"] = thr_tbl.to_dict("records")

    if not test_ok:
        log("--no-test: stopping before any test-split evaluation; nothing written")
        print(val_tbl.round(3).to_string())
        return

    # ====================== everything below uses the test split (once) ===========================
    Xte, yte = X[idx["test"]], y[idx["test"]]
    P["test"] = mc.predict_dollars(selected, Xte)

    # ---- 5. test metrics for every candidate -----------------------------------------------------
    test_rows = []
    for name, m in fitted.items():
        t = time.time()
        p = mc.predict_dollars(m, Xte)
        test_rows.append({**mc.metrics(yte, p), "model": name, "predict_seconds": time.time() - t})
    test_tbl = pd.DataFrame(test_rows).set_index("model")
    cmp_tbl = val_tbl[["MAE", "RMSE", "R2", "MedAE", "MdAPE", "R2_log", "fit_seconds", "model_size_mb"]].add_prefix("val_").join(
        test_tbl[["MAE", "RMSE", "R2", "MedAE", "MdAPE", "predict_seconds"]].add_prefix("test_"))
    summary["model_comparison"] = cmp_tbl.reset_index().to_dict("records")
    test_df = df[idx["test"]].assign(y=yte, p=P["test"])
    by_state = group_metrics(test_df, "state", 100).sort_values("n", ascending=False)
    test_df["price_bracket"] = pd.cut(test_df.y, [0, 100e3, 200e3, 400e3, 700e3, 1e6, 2e6, 1e9],
                                      labels=["<100k", "100-200k", "200-400k", "400-700k", "700k-1M", "1-2M", ">2M"])
    by_price = group_metrics(test_df, "price_bracket", 1)
    test_df["city_size"] = pd.cut(test_df.city_key.map(n_total_city), [0, 29, 99, 499, 10**9], labels=["<30", "30-99", "100-499", "500+"])
    by_citysize = group_metrics(test_df, "city_size", 1)
    log(f"TEST {best}: MAE={test_tbl.loc[best,'MAE']:,.0f} RMSE={test_tbl.loc[best,'RMSE']:,.0f} R2={test_tbl.loc[best,'R2']:.3f} MdAPE={test_tbl.loc[best,'MdAPE']:.3f}")

    # ---- 6. missing-data strategies --------------------------------------------------------------
    inc = mc.to_model_frame(pd.read_parquet(PROCESSED / "sold_incomplete_features.parquet"))
    inc["split"] = mc.assign_split(inc["property_group"])
    Xi, yi = mc.build_features(inc), inc.price.to_numpy()
    ii = {s: (inc.split == s).to_numpy() for s in SPLITS}
    inc_pattern = np.where(inc.house_size.isna(), "house_size missing", "only bed/bath missing")
    miss_rows = []
    for name in ("hist_gb_log", "ridge_log"):
        A = fitted[name]
        B, _ = fit_timed(name, pd.concat([X[idx["train"]], Xi[ii["train"]]]), np.concatenate([y[idx["train"]], yi[ii["train"]]]))
        for split in ("validation", "test"):
            Xc, yc = X[idx[split]], y[idx[split]]
            for tag, model in (("A complete-case training", A), ("B + imputed incomplete rows", B)):
                miss_rows.append({"model": name, "strategy": tag, "evaluated_on": f"{split}: complete-case rows",
                                  **{k: v for k, v in mc.metrics(yc, mc.predict_dollars(model, Xc)).items() if k in ("n", "MAE", "RMSE", "R2", "MdAPE")}})
                for pat in ("house_size missing", "only bed/bath missing"):
                    mk = ii[split] & (inc_pattern == pat)
                    miss_rows.append({"model": name, "strategy": tag, "evaluated_on": f"{split}: incomplete rows, {pat}",
                                      **{k: v for k, v in mc.metrics(yi[mk], mc.predict_dollars(model, Xi[mk])).items() if k in ("n", "MAE", "RMSE", "R2", "MdAPE")}})
    miss_tbl = pd.DataFrame(miss_rows)
    summary["missing_data_experiment"] = miss_tbl.to_dict("records")
    summary["incomplete_rows"] = {"total": len(inc), **{s: int(ii[s].sum()) for s in SPLITS},
                                  "house_size_missing": int((inc_pattern == "house_size missing").sum())}
    log("missing-data experiment done")

    # ---- 7. geographic generalisation: hold out entire cities ---------------------------------------
    held = mc.heldout_city_mask(df.city_key)
    n_held_cities = int(df[held].city_key.nunique())
    geo_train = idx["train"] & ~held
    geo_model, _ = fit_timed("hist_gb_log", X[geo_train], y[geo_train])
    train_groups = set(df.loc[geo_train, "property_group"])
    unseen = held & ~df.property_group.isin(train_groups).to_numpy()          # drop rows whose property is in training
    seen_test = idx["test"] & ~held
    geo_rows = []
    for label, mk in (("held-out cities (never seen)", unseen), ("seen cities (test split)", seen_test)):
        d = df[mk].assign(y=y[mk], p=mc.predict_dollars(geo_model, X[mk]))
        d["city_size"] = pd.cut(d.city_key.map(n_total_city), [0, 29, 99, 499, 10**9], labels=["<30", "30-99", "100-499", "500+"])
        geo_rows.append({"evaluation": label, "city_size": "all", **mc.metrics(d.y, d.p)})
        for sz, g in d.groupby("city_size", observed=True):
            geo_rows.append({"evaluation": label, "city_size": sz, **mc.metrics(g.y, g.p)})
    geo_tbl = pd.DataFrame(geo_rows)[["evaluation", "city_size", "n", "MAE", "MedAE", "MdAPE", "R2", "R2_log"]]
    summary["geographic_holdout"] = {"heldout_cities": n_held_cities, "heldout_rows_evaluated": int(unseen.sum()),
                                     "results": geo_tbl.to_dict("records")}
    log(f"geographic holdout done ({n_held_cities:,} cities held out)")

    # ---- 8. support policy confirmation on the test split ------------------------------------------
    test_df = test_df.merge(loc[["city_key", "status"]], on="city_key", how="left")
    test_df["status"] = test_df["status"].fillna("unsupported")
    sup_test = group_metrics(test_df, "status", 1)
    test_city = group_metrics(test_df, "city_key", 1)[["n", "MdAPE", "MAE"]].add_prefix("test_").rename(columns={"test_n": "test_rows"})
    loc = loc.merge(test_city.reset_index(), on="city_key", how="left")
    loc["test_rows"] = loc.test_rows.fillna(0).astype(int)

    # ---- 9. prediction intervals on the test split ---------------------------------------------------
    Zte = feats_test = np.column_stack([pre.transform(Xte), np.log(P["test"])])
    cov_rows, lohi = [], {}
    for a in mc.CONFIG["conformal_alphas"]:
        for mode, const in (("normalised conformal (selected)", False), ("constant-width conformal", True)):
            lo, hi = interval(P["test"], Zte, a, const)
            lohi[(a, const)] = (lo, hi)
            cov_rows.append({"nominal": 1 - a, "method": mode, "coverage": float(((yte >= lo) & (yte <= hi)).mean()),
                             "mean_width_$": float((hi - lo).mean()), "median_width_$": float(np.median(hi - lo)),
                             "median_hi/lo": float(np.median(hi / lo))})
    cov_tbl = pd.DataFrame(cov_rows)
    a80 = mc.CONFIG["conformal_alphas"][0]
    lo80, hi80 = lohi[(a80, False)]
    test_df["lo"], test_df["hi"] = lo80, hi80
    test_df["covered"] = (test_df.y >= test_df.lo) & (test_df.y <= test_df.hi)
    test_df["width"] = test_df.hi - test_df.lo
    edges = np.quantile(P["calibration"], np.linspace(0, 1, 11)[1:-1])
    test_df["pred_decile"] = np.digitize(test_df.p, edges) + 1
    lo_c, hi_c = lohi[(a80, True)]
    test_df["covered_const"] = (test_df.y >= lo_c) & (test_df.y <= hi_c)

    def cov_by(col, min_n=1):
        g = test_df.groupby(col, observed=True).agg(n=("y", "size"), coverage_80=("covered", "mean"),
                                                    constant_width_coverage=("covered_const", "mean"),
                                                    median_width=("width", "median"), median_price=("y", "median"))
        return g[g.n >= min_n]
    cov_pred_decile, cov_price, cov_status = cov_by("pred_decile"), cov_by("price_bracket"), cov_by("status")
    cov_state = cov_by("state", 200)
    summary["interval_coverage"] = cov_tbl.to_dict("records")
    summary["conformal"] = {"scale_floor": SCALE_FLOOR, "quantiles_normalised": {str(k): v for k, v in qs.items()},
                            "quantiles_constant_log_width": {str(k): v for k, v in q_const.items()},
                            "calibration_rows": int(idx["calibration"].sum()), "scale_model_fit_rows": int(idx["validation"].sum())}
    log("intervals:\n" + cov_tbl.to_string(index=False))

    # ---- 10. figures --------------------------------------------------------------------------------
    FIGS.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(mc.CONFIG["seed"]).choice(len(yte), size=min(30000, len(yte)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    hb = ax[0].hexbin(yte[rs], P["test"][rs], gridsize=70, xscale="log", yscale="log", bins="log", mincnt=1, cmap="viridis")
    lim = [2e4, 5e6]; ax[0].plot(lim, lim, "r--", lw=1); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set(xlabel="recorded price ($)", ylabel="predicted price ($)", title=f"{best}: actual vs predicted (test sample)")
    res = np.log(yte) - np.log(P["test"])
    ax[1].hist(res, bins=120, range=(-2, 2)); ax[1].axvline(0, color="r", lw=1)
    ax[1].set(xlabel="log(actual / predicted)", ylabel="rows", title=f"Residuals on test (median {np.median(res):+.3f})")
    fig.tight_layout(); fig.savefig(FIGS / "actual_vs_predicted_and_residuals.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(cov_pred_decile.index.astype(str), cov_pred_decile.coverage_80, label="normalised conformal")
    ax[0].plot(range(len(cov_pred_decile)), cov_pred_decile.constant_width_coverage, "ko-", label="constant width")
    ax[0].axhline(0.8, color="r", ls="--", lw=1); ax[0].set(ylim=(0.5, 1), xlabel="predicted-price decile", ylabel="test coverage", title="80% interval coverage by predicted price")
    ax[0].legend()
    gm = geo_tbl[geo_tbl.city_size != "all"].pivot(index="city_size", columns="evaluation", values="MdAPE").reindex(["<30", "30-99", "100-499", "500+"])
    gm.plot.bar(ax=ax[1], rot=0); ax[1].set(ylabel="median absolute % error", xlabel="city size (rows)", title="Unseen vs seen cities")
    fig.tight_layout(); fig.savefig(FIGS / "interval_coverage_and_geographic_holdout.png", dpi=130); plt.close(fig)

    # ---- 11. artifacts --------------------------------------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    tr_idx = idx["train"]
    ranges = {"house_size_p0.5": float(np.quantile(df.loc[tr_idx, "house_size"], .005)),
              "house_size_p99.5": float(np.quantile(df.loc[tr_idx, "house_size"], .995)),
              "acre_lot_p99.5": float(np.nanquantile(df.loc[tr_idx, "acre_lot"], .995)),
              "price_p0.5": float(np.quantile(df.loc[tr_idx, "price"], .005)), "price_p99.5": float(np.quantile(df.loc[tr_idx, "price"], .995))}
    bundle = {"model": selected, "scale_model": scale_model, "scale_floor": SCALE_FLOOR,
              "conformal_quantiles": {str(1 - a): q for a, q in qs.items()}, "training_ranges": ranges,
              "feature_columns": mc.FEATURE_COLS, "model_name": best}
    joblib.dump(bundle, MODELS / "house_price_model.joblib", compress=3)
    support_cols = ["state", "city", "status", "reason", "n_train", "heldout_rows", "heldout_MdAPE", "median_price"]
    loc[support_cols].sort_values(["state", "city"]).to_csv(MODELS / "location_support.csv", index=False)
    loc.sort_values("n_total", ascending=False).drop(columns=["city_key"]).to_csv(REPORTS / "location_evaluation.csv", index=False)
    schema = {"inputs": {"state": "string, must exist in location_support.csv", "city": "string, must be supported or limited",
                         "bed": f"integer {mc.BOUNDS['bed']}", "bath": f"number {mc.BOUNDS['bath']}",
                         "house_size": f"square feet {mc.BOUNDS['house_size']}", "acre_lot": "acres, optional (missing allowed), 0 < x <= 1000"},
              "model_features": mc.FEATURE_COLS, "excluded": mc.CONFIG["excluded_features"],
              "target": mc.CONFIG["target"], "output": "estimate in dollars + conformal interval", "training_ranges": ranges}
    (MODELS / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    meta = {"created_utc": summary["generated_utc"], "model_name": best, "seed": mc.CONFIG["seed"],
            "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
            "model_file_mb": round((MODELS / "house_price_model.joblib").stat().st_size / 1e6, 2),
            "training_rows": int(idx["train"].sum()), "interval_levels": list(bundle["conformal_quantiles"]),
            "support_policy": sp, "input_hashes": summary["input_hashes"]}
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # reload check: identical predictions from the saved artifact
    re = joblib.load(MODELS / "house_price_model.joblib")
    chk = np.arange(0, len(Xte), max(1, len(Xte) // 2000))
    same = np.allclose(mc.predict_dollars(re["model"], Xte.iloc[chk]), P["test"][chk], rtol=1e-12)
    summary["artifact_reload_identical_predictions"] = bool(same)
    log(f"artifact reload check: {same}")

    # ---- 12. reports -------------------------------------------------------------------------------------
    summary["test_metrics_selected"] = test_tbl.loc[best].to_dict()
    summary["support_policy_test_confirmation"] = sup_test.reset_index().to_dict("records")
    summary["artifacts"] = ["models/house_price_model.joblib", "models/location_support.csv", "models/feature_schema.json",
                            "models/model_metadata.json", "reports/location_evaluation.csv", "reports/model_comparison.md"]
    (REPORTS / "experiment_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_comparison_report(summary, split_tbl, cmp_tbl, best, by_state, by_price, by_citysize, miss_tbl, geo_tbl, thr_tbl,
                            status_counts, sup_test, cov_tbl, cov_pred_decile, cov_price, cov_status, cov_state, qs, q_const, info)
    log("done")


def write_comparison_report(summary, split_tbl, cmp_tbl, best, by_state, by_price, by_citysize, miss_tbl, geo_tbl, thr_tbl,
                            status_counts, sup_test, cov_tbl, cov_pred_decile, cov_price, cov_status, cov_state, qs, q_const, info):
    g = summary["property_group_reliability"]
    t = summary["test_metrics_selected"]
    sp = mc.CONFIG["support_policy"]
    L = ["# Phase 4 model comparison (generated by `python src/train_model.py`)", "",
         "**Target.** The recorded `price` of `sold`-status records. It is not documented as a verified completed-sale price, so "
         "all estimates describe this dataset's recorded prices, not market value or an appraisal.", "",
         f"Data source: {info['source']}; matches `sold_clean.parquet`: {info['warehouse_matches_parquet']}. Seed {mc.CONFIG['seed']}.", "",
         "## Features", "",
         f"- Location: `state`, `city` (encoded by a hierarchical target encoder learned on training rows only; unseen cities fall back to their state).",
         "- Numeric: `bed`, `bath`, log house size, log lot size, plus missing-value indicators.",
         "- Excluded: " + "; ".join(mc.CONFIG["excluded_features"]) + ".", "",
         "## Leakage controls", "",
         "- Each property group (street code + zip) is hashed to one split, so a property never appears in two splits "
         "(asserted in code). Splits are 60% train, 10% validation, 10% calibration, 20% test.",
         "- Imputers, scalers and the target encoder are fitted on training rows inside the pipeline (target encoding is cross-fitted).",
         "- Validation chose the model; the calibration split only sets interval widths; the test split was evaluated once at the end.", "",
         md_table(split_tbl.reset_index().rename(columns={"split": "split"})), "",
         f"Group-key reliability: {g['rows_without_property_key']:,} rows have no key (singleton groups); only "
         f"{g['groups_with_multiple_rows']:,} groups have more than one row ({g['rows_in_multi_row_groups']:,} rows), and in "
         f"{g['multi_row_groups_whose_rows_differ_in_bed_bath_or_size']:,} of them the rows differ in bed/bath/size, so the key sometimes "
         f"joins different units. {g['street_codes_shared_by_multiple_zips']:,} street codes occur in several zips. The key cannot link "
         "the same property listed under a missing key. Repeated sold properties are rare, so leakage risk is small but not zero.", "",
         "## Model comparison", "",
         "Selection used **validation MAE**; test metrics are reported afterwards for every candidate (dollars; R2 on dollars).", "",
         md_table(cmp_tbl.reset_index()), "",
         f"**Selected: `{best}`.** Test: MAE ${t['MAE']:,.0f}, RMSE ${t['RMSE']:,.0f}, R2 {t['R2']:.3f}, median absolute error "
         f"${t['MedAE']:,.0f}, median absolute % error {t['MdAPE']:.1%}, on {int(t['n']):,} test rows. "
         "In a validation-only check made before the encoder change below, larger boosting settings (up to 1,500 iterations / 127 leaves) "
         "changed validation MAE by under 0.5%, so model capacity was not the bottleneck; the information in the features is.", "",
         "### Development note: why the location encoder is size-adjusted", "",
         "The first full run showed a systematic bias in Florida (median log residual +0.19 in train, validation, calibration and test alike, "
         "other states about 0.00). Cause: Florida and Texas have the same unconditional mean log price (12.697), and an unconditional state/city mean "
         "mixes price level with house-size mix, so the model could not tell the states apart. Fix: the encoder first removes the effect of size, "
         "bedrooms and bathrooms with a linear fit learned on training rows, then encodes state/city means of the remainder. "
         "Validation-only comparison (scratch experiment, same split and model settings):", "",
         md_table(pd.DataFrame([
             {"variant": "unadjusted encoder (first run)", "val_MAE": 115785, "val_MdAPE": 0.163, "Florida_median_log_resid": 0.197},
             {"variant": "+ state one-hot", "val_MAE": 112207, "val_MdAPE": 0.155, "Florida_median_log_resid": 0.007},
             {"variant": "size-adjusted encoder (adopted)", "val_MAE": 106871, "val_MdAPE": 0.148, "Florida_median_log_resid": 0.014},
             {"variant": "size-adjusted + state one-hot", "val_MAE": 106565, "val_MdAPE": 0.147, "Florida_median_log_resid": 0.010}])), "",
         "**Test-set exposure disclosure:** the Florida bias was noticed in the by-state test table of the first full run, so the test split had been "
         "viewed once before this change. The change itself was designed and chosen using training/validation residuals and validation metrics only. "
         "The reported test numbers come from the rerun of the final pipeline.", "",
         "### Test performance by price bracket", "", md_table(by_price.reset_index()), "",
         "### Test performance by city size (rows in the dataset)", "", md_table(by_citysize.reset_index()), "",
         "### Test performance by state (states with at least 100 test rows)", "", md_table(by_state.reset_index()), "",
         "![actual vs predicted](figures/actual_vs_predicted_and_residuals.png)", "",
         "## Missing-data strategies", "",
         f"A = trained on the {summary['splits'][0]['rows']:,} complete-case training rows. B = A plus the "
         f"{summary['incomplete_rows']['train']:,} training rows that lack bed, bath or house size (median imputation fitted on training data, "
         "with missing indicators; prices are never imputed). Both are scored on the same complete-case rows; B-only coverage is the incomplete rows.", "",
         md_table(miss_tbl), "",
         "## Geographic generalisation (whole cities held out)", "",
         f"{summary['geographic_holdout']['heldout_cities']:,} cities were removed from training entirely; their "
         f"{summary['geographic_holdout']['heldout_rows_evaluated']:,} rows are the 'never seen' evaluation. 'Seen cities' are test-split rows from cities that remained in training.", "",
         md_table(geo_tbl), "",
         "![coverage and holdout](figures/interval_coverage_and_geographic_holdout.png)", "",
         "## Location support", "",
         "Candidate minimum training-row thresholds (error measured on validation+calibration rows, never the test split):", "",
         md_table(thr_tbl), "",
         f"Policy: **supported** = at least {sp['supported_min_train_rows']} training rows, at least {sp['min_heldout_rows']} held-out rows and held-out median error "
         f"<= {sp['max_heldout_mdape']:.0%}; **limited** = at least {sp['limited_min_train_rows']} training rows but failing a supported criterion (warn, show a wider-uncertainty message); "
         "**unsupported** = fewer rows or a city not in the table (no prediction). Cities in the table:", "",
         md_table(status_counts.reset_index()), "",
         "Confirmation on the untouched test split, by city status:", "", md_table(sup_test.reset_index()), "",
         "Per-city results: `reports/location_evaluation.csv`; app-facing table: `models/location_support.csv`.", "",
         "## Prediction intervals", "",
         "Method: split conformal prediction on log price. A small gradient-boosting model, fitted on the validation split, predicts the typical "
         "size of the log residual for a home; the calibration split sets a single multiplier "
         f"(80%: {qs[0.2]:.3f}, 90%: {qs[0.1]:.3f}). The interval is `estimate * exp(+/- multiplier * predicted spread)`. "
         "A constant-width alternative is shown for comparison.", "",
         md_table(cov_tbl), "",
         "### 80% interval coverage by predicted-price decile (usable at prediction time)", "", md_table(cov_pred_decile.reset_index()), "",
         "### By actual price bracket (conditioning on the outcome, so coverage naturally drifts for the extremes)", "", md_table(cov_price.reset_index()), "",
         "### By location status", "", md_table(cov_status.reset_index()), "",
         "### By state (at least 200 test rows)", "", md_table(cov_state.reset_index()), "",
         "Coverage is marginal over the test distribution and holds on average if the future data resemble this snapshot "
         "(Oct 2021 - May 2022, per `prev_sold_date`). It is not a guarantee for an individual home or city, and it describes the "
         "recorded dataset target, not market value or an appraisal.", ""]
    (REPORTS / "model_comparison.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
