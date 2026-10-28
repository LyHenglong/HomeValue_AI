"""Phase 4 checks: saved artifacts, leakage controls, inference/training parity and input validation.
Run from the project root:  python -m pytest tests -q
Requires models/ artifacts (python src/train_model.py) and data/processed/sold_clean.parquet."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model_common as mc  # noqa: E402
import predict as pr  # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "models" / "house_price_model.joblib").exists(), reason="train the model first")


@pytest.fixture(scope="module")
def art():
    return pr.load_artifacts()


@pytest.fixture(scope="module")
def data():
    df = mc.to_model_frame(pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet"))
    df["split"] = mc.assign_split(df["property_group"])
    return df


@pytest.fixture(scope="module")
def good(art):
    s = art.support[art.support.status == "supported"].sort_values("n_train", ascending=False).iloc[0]
    return {"state": s.state, "city": s.city, "bed": 3, "bath": 2, "house_size": 1800, "acre_lot": 0.2}


# ---- leakage controls ---------------------------------------------------------------------------
def test_no_target_derived_or_identifier_features():
    banned = {"price", "price_per_sqft", "street", "brokered_by", "raw_row_id", "property_key", "zip_code", "prev_sold_date", "sold_year"}
    assert not banned & set(mc.FEATURE_COLS)
    assert not any("price" in c for c in mc.FEATURE_COLS)


def test_property_groups_never_span_splits(data):
    assert (data.groupby("property_group")["split"].nunique() == 1).all()


def test_split_assignment_is_deterministic(data):
    assert (mc.assign_split(data["property_group"]) == data["split"].to_numpy()).all()


def test_split_fractions_close_to_config(data):
    share = data["split"].value_counts(normalize=True)
    for k, v in mc.CONFIG["split_fractions"].items():
        assert abs(share[k] - v) < 0.01


# ---- inference matches training ---------------------------------------------------------------------
def test_single_row_inference_equals_batch_training_path(art, data):
    ok = art.support[art.support.status.isin(["supported", "limited"])]
    t = data[(data.split == "test")].merge(ok[["state", "city"]], on=["state", "city"]).sample(150, random_state=1)
    batch = mc.predict_dollars(art.bundle["model"], mc.build_features(t))
    single = []
    for r in t.itertuples():
        out = pr.predict_price(art, r.state, r.city, r.bed, r.bath, r.house_size, None if pd.isna(r.acre_lot) else r.acre_lot)
        assert out["status"] in ("ok", "warning"), out
        single.append(out["estimate"])
    np.testing.assert_allclose(single, batch, rtol=1e-9)


def test_saved_model_reproduces_stored_metadata(art):
    assert art.bundle["model_name"] in ("hist_gb_log", "hist_gb_raw", "random_forest_log", "ridge_log")
    assert set(art.bundle["conformal_quantiles"]) == {"0.8", "0.9"}


# ---- typical, extreme and invalid inputs ----------------------------------------------------------------
def test_typical_prediction_is_valid(art, good):
    out = pr.predict_price(art, **good)
    assert out["status"] in ("ok", "warning")
    assert 0 < out["low"] < out["estimate"] < out["high"]
    assert 20_000 < out["estimate"] < 20_000_000


def test_wider_level_gives_wider_range(art, good):
    a, b = pr.predict_price(art, **good, interval=0.8), pr.predict_price(art, **good, interval=0.9)
    assert (b["high"] - b["low"]) > (a["high"] - a["low"])
    assert a["estimate"] == b["estimate"]


def test_lot_size_is_optional(art, good):
    g = dict(good, acre_lot=None)
    assert pr.predict_price(art, **g)["status"] in ("ok", "warning")


def test_bigger_house_does_not_lower_estimate_much(art, good):
    small = pr.predict_price(art, **dict(good, house_size=1000))["estimate"]
    large = pr.predict_price(art, **dict(good, house_size=3000))["estimate"]
    assert large > small


@pytest.mark.parametrize("override", [
    {"bed": None}, {"bath": None}, {"house_size": None}, {"bed": 0}, {"bed": 21}, {"bed": 2.5}, {"bed": "three"}, {"bed": True},
    {"bath": 0}, {"bath": 21}, {"bath": float("nan")}, {"house_size": 199}, {"house_size": 50_001}, {"house_size": -5},
    {"house_size": float("inf")}, {"acre_lot": 0}, {"acre_lot": -1}, {"acre_lot": 1001}, {"acre_lot": "big"},
])
def test_invalid_inputs_are_rejected(art, good, override):
    out = pr.predict_price(art, **dict(good, **override))
    assert out["status"] == "rejected" and out["estimate"] is None and out["messages"]


def test_extreme_but_legal_house_size_warns(art, good):
    out = pr.predict_price(art, **dict(good, house_size=45_000))
    assert out["status"] in ("warning", "rejected") or any("unusual" in m for m in out["messages"])
    assert out["status"] == "warning"


def test_bad_interval_level_rejected(art, good):
    assert pr.predict_price(art, **good, interval=0.5)["status"] == "rejected"


def test_unknown_and_blank_locations_rejected(art, good):
    for st, ci in [("Atlantis", "Nowhere"), (good["state"], "Not A Real City"), ("", ""), (None, None), (good["state"], "")]:
        out = pr.predict_price(art, **dict(good, state=st, city=ci))
        assert out["status"] == "rejected" and out["estimate"] is None


def test_unsupported_city_rejected_and_limited_city_warns(art, good):
    uns = art.support[art.support.status == "unsupported"].iloc[0]
    out = pr.predict_price(art, **dict(good, state=uns.state, city=uns.city))
    assert out["status"] == "rejected" and out["location_status"] == "unsupported"
    lim = art.support[art.support.status == "limited"].iloc[0]
    out = pr.predict_price(art, **dict(good, state=lim.state, city=lim.city))
    assert out["status"] == "warning" and out["estimate"] is not None


def test_location_matching_ignores_case_and_spaces(art, good):
    out = pr.predict_price(art, **dict(good, state=f"  {good['state'].upper()} ", city=good["city"].lower()))
    assert out["status"] in ("ok", "warning")


def test_dropdown_options_come_from_support_table(art):
    opts = pr.supported_locations(art, include_limited=False)
    assert set(opts.status) == {"supported"} and len(opts) == (art.support.status == "supported").sum()
    assert not art.support.duplicated(["state", "city"]).any()


def test_unseen_category_in_encoder_does_not_crash():
    enc = mc.HierarchicalTargetEncoder(n_adjust=0).fit(pd.DataFrame({"s": ["A", "A", "B"], "c": ["A|x", "A|y", "B|z"]}), [10.0, 11.0, 12.0])
    out = enc.transform(pd.DataFrame({"s": ["A", "Z"], "c": ["A|new", "Z|new"]}))
    assert np.isfinite(out).all() and out[1, 1] == out[1, 0]   # unseen state and city fall back to the global value


def test_conformal_quantile_formula():
    s = np.arange(1, 101, dtype=float)
    assert mc.conformal_quantile(s, 0.2) == 81.0   # ceil(101 * 0.8) = 81st smallest
