"""Shared definitions for Phase 4: experiment configuration, features, encoders and metrics.

Imported by src/train_model.py (training/evaluation) and src/predict.py (inference) so that the exact same
feature code runs in both places. Nothing in this module reads or writes files.
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")  # silences joblib's physical-core probe on Windows

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------------------------
# Experiment configuration (written to reports/experiment_summary.json)
# ------------------------------------------------------------------------------------------------
CONFIG = {
    "seed": 42,
    "target": "price (recorded price of sold-status records; not a verified completed-sale price)",
    # Rows are assigned to a split by hashing property_group, so every row of a property shares a split.
    "split_fractions": {"train": 0.60, "validation": 0.10, "calibration": 0.10, "test": 0.20},
    "heldout_city_fraction": 0.15,        # share of cities held out entirely in the geographic experiment
    "conformal_alphas": [0.20, 0.10],     # 80% and 90% intervals
    # Location support policy. Thresholds 30 / 100 / 500 were examined (reports/location_evaluation.csv);
    # status is decided from training-row count and held-out error on validation+calibration rows only.
    "support_policy": {"supported_min_train_rows": 100, "limited_min_train_rows": 30,
                       "min_heldout_rows": 10, "max_heldout_mdape": 0.25},
    "features": {
        "location": ["state", "city_key"],
        "numeric": ["bed", "bath", "log_house_size", "log_acre_lot"],
        "missing_indicators": ["bed_missing", "bath_missing", "size_missing", "lot_missing"],
    },
    "excluded_features": ["price_per_sqft (derived from the target)", "prev_sold_date / sold_year (meaning unverified)",
                          "street, brokered_by, raw_row_id, property_key (identifiers)", "zip_code (not a user-facing input; high cardinality)"],
}
NUMERIC_COLS = CONFIG["features"]["numeric"] + CONFIG["features"]["missing_indicators"]
ADJUST_COLS = ["log_house_size", "bed", "bath"]   # used only to size-adjust the location encoding
LOCATION_COLS = CONFIG["features"]["location"]
FEATURE_COLS = LOCATION_COLS + NUMERIC_COLS

# Input validity bounds, taken from the Phase 2 cleaning rules.
BOUNDS = {"bed": (1, 20), "bath": (1, 20), "house_size": (200, 50_000), "acre_lot": (0.0, 1_000)}


# ------------------------------------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------------------------------------
def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise warehouse/parquet column names to: state, city, bed, bath, house_size, acre_lot."""
    return df.rename(columns={"bedrooms": "bed", "bathrooms": "bath", "house_size_sqft": "house_size"})


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pure, stateless feature construction. Every learned step (encoding, imputation) lives in the pipeline."""
    out = pd.DataFrame(index=df.index)
    out["state"] = df["state"].astype(str)
    out["city_key"] = out["state"] + "|" + df["city"].astype(str)
    out["bed"] = pd.to_numeric(df["bed"], errors="coerce").astype(float)
    out["bath"] = pd.to_numeric(df["bath"], errors="coerce").astype(float)
    size = pd.to_numeric(df["house_size"], errors="coerce").astype(float)
    lot = pd.to_numeric(df["acre_lot"], errors="coerce").astype(float)
    out["log_house_size"] = np.log(size.where(size > 0))
    out["log_acre_lot"] = np.log(lot.where(lot > 0))
    out["bed_missing"] = out["bed"].isna().astype(float)
    out["bath_missing"] = out["bath"].isna().astype(float)
    out["size_missing"] = out["log_house_size"].isna().astype(float)
    out["lot_missing"] = out["log_acre_lot"].isna().astype(float)
    return out[FEATURE_COLS]


def stable_uniform(keys: pd.Series, seed: int, salt: str) -> np.ndarray:
    """Deterministic value in [0, 1) per key (same key -> same value, independent of dataset composition)."""
    s = pd.Series(f"{salt}|{seed}|", index=keys.index) + keys.astype(str)
    h = pd.util.hash_pandas_object(s, index=False).to_numpy()
    return (h % 1_000_003) / 1_000_003


def assign_split(groups: pd.Series, seed: int = CONFIG["seed"]) -> np.ndarray:
    u = stable_uniform(groups, seed, "split")
    f = CONFIG["split_fractions"]
    edges = np.cumsum([f["train"], f["validation"], f["calibration"], f["test"]])
    return np.select([u < edges[0], u < edges[1], u < edges[2]], ["train", "validation", "calibration"], "test")


def heldout_city_mask(city_keys: pd.Series, seed: int = CONFIG["seed"]) -> np.ndarray:
    return stable_uniform(city_keys, seed, "city") < CONFIG["heldout_city_fraction"]


# ------------------------------------------------------------------------------------------------
# Hierarchical target encoder: state mean -> city mean shrunk toward its state
# ------------------------------------------------------------------------------------------------
class HierarchicalTargetEncoder(BaseEstimator, TransformerMixin):
    """Encode (state, city) as smoothed mean target values, learned from the training data only.

    state_te = (n_s * mean_s + k_s * global_mean) / (n_s + k_s)
    city_te  = (n_c * mean_c + k_c * state_te)    / (n_c + k_c)     # unseen city -> its state's value
    city_logn = log1p(n_c)                                          # how much training evidence the city has

    fit_transform() is cross-fitted: each training row is encoded with statistics computed on the *other*
    folds, so a row's own target never leaks into its own feature. transform() (validation, test, inference)
    uses statistics from all training rows. Unseen states/cities are handled without error.
    """

    def __init__(self, prior_state=20.0, prior_city=10.0, n_folds=5, log_target=False, random_state=42, n_adjust=0):
        self.prior_state = prior_state
        self.prior_city = prior_city
        self.n_folds = n_folds
        self.log_target = log_target
        self.random_state = random_state
        self.n_adjust = n_adjust   # if > 0, columns 2..2+n_adjust of X are covariates used to size-adjust the target

    def _adjust(self, X, y):
        """Replace y by its residual from a linear fit on the covariates (learned on training rows only), so the
        state/city values describe the price level *given* size/bed/bath, not a mix of level and house-size mix."""
        if self.n_adjust == 0:
            return y
        C = X.iloc[:, 2:2 + self.n_adjust].to_numpy(float)
        self.cov_median_ = np.nanmedian(C, axis=0)
        A = np.column_stack([np.ones(len(C)), np.where(np.isnan(C), self.cov_median_, C)])
        self.coef_ = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - A @ self.coef_

    def _y(self, y):
        y = np.asarray(y, dtype=float)
        return np.log(y) if self.log_target else y

    def _fit_stats(self, X, y):
        s_codes, s_uni = pd.factorize(X.iloc[:, 0])
        c_codes, c_uni = pd.factorize(X.iloc[:, 1])
        self.state_index_, self.city_index_ = pd.Index(s_uni), pd.Index(c_uni)
        self.global_mean_ = float(y.mean())
        self.n_s_ = np.bincount(s_codes, minlength=len(s_uni)).astype(float)
        self.sum_s_ = np.bincount(s_codes, weights=y, minlength=len(s_uni))
        self.n_c_ = np.bincount(c_codes, minlength=len(c_uni)).astype(float)
        self.sum_c_ = np.bincount(c_codes, weights=y, minlength=len(c_uni))
        return s_codes, c_codes

    def fit(self, X, y):
        self._fit_stats(X, self._adjust(X, self._y(y)))
        return self

    def fit_transform(self, X, y=None, **kw):
        y = self._adjust(X, self._y(y))
        s_codes, c_codes = self._fit_stats(X, y)
        fold = np.random.default_rng(self.random_state).integers(0, self.n_folds, size=len(y))
        out = np.empty((len(y), 3))
        ns, nc = len(self.n_s_), len(self.n_c_)
        for f in range(self.n_folds):
            m = fold == f
            n_s = self.n_s_ - np.bincount(s_codes[m], minlength=ns)
            sum_s = self.sum_s_ - np.bincount(s_codes[m], weights=y[m], minlength=ns)
            n_c = self.n_c_ - np.bincount(c_codes[m], minlength=nc)
            sum_c = self.sum_c_ - np.bincount(c_codes[m], weights=y[m], minlength=nc)
            out[m] = self._encode(s_codes[m], c_codes[m], n_s, sum_s, n_c, sum_c)
        return out

    def transform(self, X):
        sc = self.state_index_.get_indexer(X.iloc[:, 0])
        cc = self.city_index_.get_indexer(X.iloc[:, 1])
        return self._encode(sc, cc, self.n_s_, self.sum_s_, self.n_c_, self.sum_c_)

    def _encode(self, sc, cc, n_s, sum_s, n_c, sum_c):
        g = self.global_mean_
        ok_s, ok_c = sc >= 0, cc >= 0
        ns = np.where(ok_s, n_s[np.maximum(sc, 0)], 0.0)
        ss = np.where(ok_s, sum_s[np.maximum(sc, 0)], 0.0)
        state_te = (ss + self.prior_state * g) / (ns + self.prior_state)
        nc = np.where(ok_c, n_c[np.maximum(cc, 0)], 0.0)
        cs = np.where(ok_c, sum_c[np.maximum(cc, 0)], 0.0)
        city_te = (cs + self.prior_city * state_te) / (nc + self.prior_city)
        return np.column_stack([state_te, city_te, np.log1p(nc)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["state_te", "city_te", "city_logn"])


# ------------------------------------------------------------------------------------------------
# ZIP handling (moved from src/zip_experiment.py for the v2 hybrid model; code unchanged)
# ------------------------------------------------------------------------------------------------
_ZIP_STR = re.compile(r"(\d{5})(?:-\d{4})?")


def normalize_zip(v) -> str | None:
    """Return a 5-character ZIP string or None.

    Strings must already contain 5 digits (ZIP+4 is cut to 5); a shorter string is ambiguous and rejected.
    Integers (and integral floats) are zero-padded, because numeric parsing is exactly what drops leading zeros
    (the raw CSV reads 00601 as 601). 00000, negatives, booleans, NaN and other types are invalid.
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v) or v != int(v):
            return None
        v = int(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):05d}" if 0 < int(v) <= 99_999 else None
    if isinstance(v, str):
        m = _ZIP_STR.fullmatch(v.strip())
        return m.group(1) if m and m.group(1) != "00000" else None
    return None


ZIP_FEATURE_COLS = FEATURE_COLS + ["zip_key"]


def build_zip_features(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    zips = df["zip_code"] if "zip_code" in df else pd.Series(None, index=df.index, dtype=object)
    X["zip_key"] = pd.Series([normalize_zip(z) for z in zips], index=df.index, dtype=object)
    return X[ZIP_FEATURE_COLS]


class ZipHierarchicalEncoder(BaseEstimator, TransformerMixin):
    """state -> city -> ZIP target encoding on size-adjusted log price, learned from training rows only.

    Columns: state, city_key, zip_key, then n_adjust covariates. Same scheme as model_common's production encoder,
    one level deeper:  zip_te = (n_z * mean_z + k_z * city_te) / (n_z + k_z).  A missing/invalid ZIP or a ZIP never
    seen in training gets n_z = 0, i.e. exactly its city's value. fit_transform is cross-fitted (5 folds), so no
    training row's own price enters its own encoding.
    Output: state_te, city_te, city_logn, zip_te, zip_logn.
    """

    def __init__(self, prior_state=20.0, prior_city=10.0, prior_zip=20.0, n_folds=5, random_state=42, n_adjust=3):
        self.prior_state, self.prior_city, self.prior_zip = prior_state, prior_city, prior_zip
        self.n_folds, self.random_state, self.n_adjust = n_folds, random_state, n_adjust

    def _adjust(self, X, y):
        C = X.iloc[:, 3:3 + self.n_adjust].to_numpy(float)
        self.cov_median_ = np.nanmedian(C, axis=0)
        A = np.column_stack([np.ones(len(C)), np.where(np.isnan(C), self.cov_median_, C)])
        self.coef_ = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - A @ self.coef_

    def _fit_stats(self, X, y):
        self.index_, codes, self.n_, self.sum_ = [], [], [], []
        for j in range(3):
            c, uni = pd.factorize(X.iloc[:, j])            # missing ZIP -> code -1
            self.index_.append(pd.Index(uni))
            codes.append(c)
            ok = c >= 0
            self.n_.append(np.bincount(c[ok], minlength=len(uni)).astype(float))
            self.sum_.append(np.bincount(c[ok], weights=y[ok], minlength=len(uni)))
        self.global_mean_ = float(y.mean())
        return codes

    def fit(self, X, y):
        self._fit_stats(X, self._adjust(X, np.asarray(y, float)))
        return self

    def fit_transform(self, X, y=None, **kw):
        y = self._adjust(X, np.asarray(y, float))
        codes = self._fit_stats(X, y)
        fold = np.random.default_rng(self.random_state).integers(0, self.n_folds, size=len(y))
        out = np.empty((len(y), 5))
        for f in range(self.n_folds):
            m = fold == f
            n, s = [], []
            for j in range(3):
                ok = m & (codes[j] >= 0)
                n.append(self.n_[j] - np.bincount(codes[j][ok], minlength=len(self.n_[j])))
                s.append(self.sum_[j] - np.bincount(codes[j][ok], weights=y[ok], minlength=len(self.n_[j])))
            out[m] = self._encode([c[m] for c in codes], n, s)
        return out

    def transform(self, X):
        codes = [self.index_[j].get_indexer(X.iloc[:, j]) for j in range(3)]
        return self._encode(codes, self.n_, self.sum_)

    def _encode(self, codes, n, s):
        def lookup(j):
            ok = codes[j] >= 0
            k = np.maximum(codes[j], 0)
            return np.where(ok, n[j][k], 0.0), np.where(ok, s[j][k], 0.0)
        ns, ss = lookup(0)
        state_te = (ss + self.prior_state * self.global_mean_) / (ns + self.prior_state)
        nc, sc = lookup(1)
        city_te = (sc + self.prior_city * state_te) / (nc + self.prior_city)
        nz, sz = lookup(2)
        zip_te = (sz + self.prior_zip * city_te) / (nz + self.prior_zip)
        return np.column_stack([state_te, city_te, np.log1p(nc), zip_te, np.log1p(nz)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["state_te", "city_te", "city_logn", "zip_te", "zip_logn"])


# ------------------------------------------------------------------------------------------------
# Pipelines
# ------------------------------------------------------------------------------------------------
def _preprocessor(scale: bool, log_target_encoder: bool, adjust: bool = True) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median"))]  # missing values are imputed from training rows only
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("loc", HierarchicalTargetEncoder(log_target=log_target_encoder, random_state=CONFIG["seed"],
                                           n_adjust=len(ADJUST_COLS) if adjust else 0),
          LOCATION_COLS + (ADJUST_COLS if adjust else [])),
         ("num", Pipeline(num), NUMERIC_COLS)],
        remainder="drop")


def make_model(name: str):
    """Return an unfitted estimator that maps build_features() output to a price in dollars."""
    seed = CONFIG["seed"]
    if name == "dummy_median":
        return DummyRegressor(strategy="median")
    if name == "ridge_log":
        reg = Pipeline([("pre", _preprocessor(True, False)), ("model", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))])
    elif name == "random_forest_log":
        reg = Pipeline([("pre", _preprocessor(False, False)),
                        ("model", RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=0.6,
                                                        max_samples=0.3, n_jobs=4, random_state=seed))])
    elif name == "hist_gb_log":
        reg = Pipeline([("pre", _preprocessor(False, False)),
                        ("model", HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
                                                                min_samples_leaf=50, l2_regularization=1.0,
                                                                early_stopping=True, n_iter_no_change=15,
                                                                random_state=seed))])
    elif name == "hist_gb_raw":  # same model, raw dollar target (encoder works on log prices internally)
        return Pipeline([("pre", _preprocessor(False, True)),
                         ("model", HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.1, max_leaf_nodes=63,
                                                                 min_samples_leaf=50, l2_regularization=1.0,
                                                                 early_stopping=True, n_iter_no_change=15,
                                                                 random_state=seed))])
    else:
        raise ValueError(f"unknown model {name}")
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp, check_inverse=False)


def predict_dollars(model, X: pd.DataFrame) -> np.ndarray:
    """Predictions in dollars, never below $1 (the log models are always positive; the raw model may not be)."""
    return np.clip(np.asarray(model.predict(X), dtype=float), 1.0, None)


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1.0, None)
    err = p - y
    return {"n": int(len(y)), "MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(r2_score(y, p)), "MedAE": float(np.median(np.abs(err))),
            "MdAPE": float(np.median(np.abs(err) / y)), "R2_log": float(r2_score(np.log(y), np.log(p)))}


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    k = min(int(np.ceil((len(s) + 1) * (1 - alpha))), len(s))
    return float(s[k - 1])
