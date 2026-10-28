"""ZIP-code experiment checks: ZIP normalisation, leading zeros, missing values, leakage, split integrity,
inference compatibility and isolation from the production artifacts. Run: python -m pytest tests -q"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model_common as mc  # noqa: E402
import zip_experiment as zx  # noqa: E402

CANDIDATES = ROOT / "models" / "experiments" / "zip" / "zip_candidates.joblib"


# ---- ZIP normalisation ------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("00601", "00601"), (" 02134 ", "02134"), ("12345-6789", "12345"), (601, "00601"), (601.0, "00601"),
    (np.int64(2134), "02134"), (99999, "99999"),
    ("601", None), ("0601", None), ("00000", None), (0, None), (-5, None), (100_000, None), (601.5, None),
    (float("nan"), None), (None, None), (True, None), ("abcde", None), ("123456", None), ("", None), ([], None),
])
def test_normalize_zip(raw, expected):
    assert zx.normalize_zip(raw) == expected


def test_leading_zeros_preserved_from_cleaned_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet", columns=["zip_code", "state", "city", "bed",
                                                                                        "bath", "house_size", "acre_lot"])
    lead = df[df.zip_code.str.startswith("0")].head(500)
    assert len(lead) == 500
    X = zx.build_zip_features(lead)
    assert (X.zip_key == lead.zip_code).all() and X.zip_key.str.len().eq(5).all()


def test_missing_zip_column_and_values_become_none():
    base = pd.DataFrame({"state": ["Texas"] * 3, "city": ["Houston"] * 3, "bed": [3] * 3, "bath": [2] * 3,
                         "house_size": [1800] * 3, "acre_lot": [np.nan] * 3})
    assert zx.build_zip_features(base).zip_key.isna().all()                       # no zip column at all
    X = zx.build_zip_features(base.assign(zip_code=[None, "bad", "77002"]))
    assert X.zip_key.tolist()[:2] == [None, None] and X.zip_key.iloc[2] == "77002"
    assert list(X.columns) == zx.ZIP_FEATURE_COLS


def test_zip_is_categorical_not_numeric():
    assert "zip_key" in zx.ZIP_FEATURE_COLS and "zip_code" not in mc.NUMERIC_COLS
    X = zx.build_zip_features(pd.DataFrame({"state": ["A"], "city": ["B"], "bed": [1], "bath": [1], "house_size": [900],
                                            "acre_lot": [0.1], "zip_code": ["00601"]}))
    assert isinstance(X.zip_key.iloc[0], str)


def test_no_target_derived_feature_columns():
    assert not any("price" in c for c in zx.ZIP_FEATURE_COLS)


# ---- encoder: fallback and leakage --------------------------------------------------------------------------
def _toy(n_per_zip=(1, 40, 40)):
    rows, y = [], []
    rng = np.random.default_rng(0)
    for i, n in enumerate(n_per_zip):
        for _ in range(n):
            rows.append({"state": "S", "city_key": "S|C", "zip_key": f"0000{i + 1}", "log_house_size": 7.5,
                         "bed": 3.0, "bath": 2.0})
            y.append(12 + i + rng.normal(0, 0.1))
    return pd.DataFrame(rows), np.array(y)


def test_unseen_or_missing_zip_falls_back_to_city_value():
    X, y = _toy()
    enc = zx.ZipHierarchicalEncoder(n_adjust=3).fit(X, y)
    new = X.head(3).copy()
    new["zip_key"] = [None, "99999", "00002"]
    out = enc.transform(new)
    assert out[0, 3] == pytest.approx(out[0, 1]) and out[1, 3] == pytest.approx(out[1, 1])   # zip_te == city_te
    assert out[0, 4] == 0 and out[1, 4] == 0                                                   # zip_logn == 0
    assert out[2, 4] > 0 and np.isfinite(out).all()


def test_cross_fitting_excludes_a_rows_own_price():
    X, y = _toy()                                     # ZIP 00001 has exactly one row
    enc = zx.ZipHierarchicalEncoder(n_adjust=3)
    out = enc.fit_transform(X, y)
    single = (X.zip_key == "00001").to_numpy()
    assert out[single, 4] == 0                        # no other row of its ZIP is available in its fold
    assert out[single, 3] == pytest.approx(out[single, 1])
    y2 = y.copy()
    y2[single] += 100.0                               # changing that row's own price...
    out2 = zx.ZipHierarchicalEncoder(n_adjust=3).fit_transform(X, y2)
    # ...must not move its ZIP encoding relative to its city encoding (which it cannot influence via the ZIP level)
    assert (out2[single, 3] - out2[single, 1]) == pytest.approx(out[single, 3] - out[single, 1])


def test_encoder_learns_only_from_training_rows():
    X, y = _toy()
    enc = zx.ZipHierarchicalEncoder(n_adjust=3).fit(X, y)
    before = enc.transform(X.head(5)).copy()
    enc.transform(X.assign(zip_key="00003"))          # transforming other data must not change fitted state
    assert np.array_equal(enc.transform(X.head(5)), before)


# ---- split integrity and production isolation ---------------------------------------------------------------
def test_experiment_uses_production_split_and_groups_do_not_cross():
    df = zx.load_data()
    zx.check_split_matches_production(df)            # raises if counts or group integrity differ


def test_v1_production_artifacts_preserved():
    """The experiment recorded the v1 production hashes. After the approved v2 integration, v1 must still be
    recoverable from the verified backup, and the v1 model file itself (still used without ZIP) must be unchanged."""
    path = ROOT / "reports" / "zip_experiment_validation.json"
    if not path.exists():
        pytest.skip("run src/zip_experiment.py first")
    recorded = json.loads(path.read_text(encoding="utf-8"))["production_hashes"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    backup = ROOT / "models" / "backups" / "v1_no_zip"
    if backup.exists():
        for name, h in recorded.items():
            assert sha(backup / name) == h
    else:                                                       # not integrated yet: production must still be v1
        for name, h in recorded.items():
            assert sha(ROOT / "models" / name) == h
    assert sha(ROOT / "models" / "house_price_model.joblib") == recorded["house_price_model.joblib"]
    assert not (ROOT / "models" / "zip_candidates.joblib").exists()   # experimental files live under models/experiments/


# ---- inference compatibility -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bundles():
    if not CANDIDATES.exists():
        pytest.skip("run src/zip_experiment.py first")
    import joblib
    return joblib.load(ROOT / "models" / "house_price_model.joblib")["model"], joblib.load(CANDIDATES)


def _rows(zips):
    n = len(zips)
    return pd.DataFrame({"state": ["Texas"] * n, "city": ["Houston"] * n, "bed": [3] * n, "bath": [2.0] * n,
                         "house_size": [1800] * n, "acre_lot": [np.nan] * n, "zip_code": zips})


def test_hybrid_uses_production_model_without_a_usable_zip(bundles):
    prod, cand = bundles
    df = _rows([None, "not-a-zip", "00000", "99998"])            # missing, invalid, invalid, never seen
    np.testing.assert_allclose(zx.hybrid_predict(prod, cand, df), mc.predict_dollars(prod, mc.build_features(df)))


def test_hybrid_uses_zip_model_for_well_covered_zip(bundles):
    prod, cand = bundles
    houston = [z for z, n in cand["train_zip_counts"].items() if z.startswith("770") and n >= cand["hybrid_min_zip_rows"]]
    assert houston
    df = _rows([houston[0]])
    np.testing.assert_allclose(zx.hybrid_predict(prod, cand, df),
                               mc.predict_dollars(cand["zip_model"], zx.build_zip_features(df)))


def test_integer_and_string_zip_inputs_agree(bundles):
    prod, cand = bundles
    lead = next(z for z, n in cand["train_zip_counts"].items() if z.startswith("0") and n >= cand["hybrid_min_zip_rows"])
    a = zx.hybrid_predict(prod, cand, _rows([lead]).assign(state="Massachusetts", city="Boston"))
    b = zx.hybrid_predict(prod, cand, _rows([int(lead)]).assign(state="Massachusetts", city="Boston"))
    assert a[0] == b[0] and math.isfinite(a[0]) and a[0] > 0


def test_candidate_bundle_is_marked_experimental(bundles):
    _, cand = bundles
    assert "EXPERIMENTAL" in cand["note"] and cand["feature_columns"] == zx.ZIP_FEATURE_COLS
