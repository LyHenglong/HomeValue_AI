"""Model v2 (hybrid ZIP) integration checks: backup/rollback, unchanged no-ZIP behaviour, ZIP formatting and
city consistency, fallback routing, interval calibration provenance, location policy, artifact loading, app behaviour
and safe errors. Run from the project root:  python -m pytest tests -q"""
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BACKUP = MODELS / "backups" / "v1_no_zip"
APP = ROOT / "app" / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "src"))
import model_common as mc  # noqa: E402
import predict as pr  # noqa: E402

pytestmark = pytest.mark.skipif(not (MODELS / "zip_model.joblib").exists(), reason="run src/integrate_zip_model.py first")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
spec = importlib.util.spec_from_file_location("streamlit_app_v2", APP)
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


@pytest.fixture(scope="module")
def art():
    return pr.load_artifacts()


@pytest.fixture(scope="module")
def art_v1():
    """The v1 installation exactly as restored by the documented rollback (backup files, no ZIP artifacts).
    Uses tempfile rather than pytest's tmp_path, whose base folder is not writable on every Windows setup."""
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="hp_v1_rollback_"))
    try:
        for f in ("house_price_model.joblib", "location_support.csv", "feature_schema.json", "model_metadata.json"):
            shutil.copy2(BACKUP / f, d / f)
        yield pr.load_artifacts(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def ex(art):
    """Concrete examples taken from the shipped ZIP-city map and support table."""
    m, s = art.zip_map, art.support
    offered = s[s.status.isin(["supported", "limited"])][["state", "city", "status"]]
    mm = m.merge(offered, on=["state", "city"])
    zip_lead = mm[(mm.route == "zip_model") & mm.zip_code.str.startswith("0")].iloc[0]
    zip_any = mm[(mm.route == "zip_model") & (mm.status == "supported")].iloc[0]
    fb_zero = mm[(mm.route == "city_fallback") & (mm.zip_train_rows == 0)].iloc[0]
    fb_rare = mm[(mm.route == "city_fallback") & (mm.zip_train_rows > 0)].iloc[0]
    other = mm[(mm.state == zip_any.state) & (mm.city != zip_any.city)].iloc[0]
    uns = s[s.status == "unsupported"].merge(m, on=["state", "city"]).iloc[0]
    return {"lead": zip_lead, "any": zip_any, "fb_zero": fb_zero, "fb_rare": fb_rare, "other_city": other, "unsupported": uns}


def call(art, r, zip_code=None, interval=0.8, **kw):
    args = {"bed": 3, "bath": 2.0, "house_size": 1800, "acre_lot": None, **kw}
    return pr.predict_price(art, r.state, r.city, args["bed"], args["bath"], args["house_size"], args["acre_lot"],
                            interval=interval, zip_code=zip_code)


# ---- backup and rollback --------------------------------------------------------------------------------------
def test_v1_backup_verified_against_manifest():
    man = json.loads((BACKUP / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    for f, h in man.items():
        assert sha(BACKUP / f) == h
    assert sha(MODELS / "house_price_model.joblib") == man["house_price_model.joblib"]   # v1 model still in place


def test_rollback_installation_disables_zip(art_v1, ex):
    assert not art_v1.zip_enabled
    r = ex["any"]
    out = call(art_v1, r, zip_code=r.zip_code)
    assert out["status"] == "rejected" and "not available" in out["messages"][0]
    assert call(art_v1, r)["status"] in ("ok", "warning")


# ---- no-ZIP behaviour is unchanged from v1 ------------------------------------------------------------------------
def test_no_zip_estimates_and_ranges_identical_to_v1(art, art_v1):
    both = art.support[art.support.status != "unsupported"].merge(
        art_v1.support[art_v1.support.status != "unsupported"][["state", "city"]], on=["state", "city"]).sample(60, random_state=3)
    for r in both.itertuples():
        for lvl in (0.8, 0.9):
            a, b = call(art, r, interval=lvl, house_size=1500 + 10 * (r.Index % 50)), call(art_v1, r, interval=lvl, house_size=1500 + 10 * (r.Index % 50))
            assert a["model_path"] == "no_zip"
            assert (a["estimate"], a["low"], a["high"]) == (b["estimate"], b["low"], b["high"])


def test_blank_zip_means_no_zip(art, ex):
    r = ex["any"]
    for blank in (None, "", "   "):
        assert call(art, r, zip_code=blank)["model_path"] == "no_zip"


# ---- ZIP formatting and consistency ---------------------------------------------------------------------------------
def test_zip_map_keeps_leading_zeros(art):
    assert art.zip_map.zip_code.str.fullmatch(r"\d{5}").all()
    assert art.zip_map.zip_code.str.startswith("0").any()
    raw = pd.read_csv(MODELS / "zip_city_map.csv", dtype=str)
    assert raw.zip_code.str.len().eq(5).all()


def test_leading_zero_zip_uses_zip_model(art, ex):
    r = ex["lead"]
    out = call(art, r, zip_code=r.zip_code)
    assert out["status"] in ("ok", "warning") and out["model_path"] == "zip_model" and out["zip_code"] == r.zip_code


@pytest.mark.parametrize("variant, ok", [("{z}", True), (" {z} ", True), ("{z}-1234", True), ("{z3}", False),
                                         ("{z}9", False), ("abcde", False), ("00000", False)])
def test_zip_string_formats(art, ex, variant, ok):
    r = ex["lead"]
    out = call(art, r, zip_code=variant.format(z=r.zip_code, z3=r.zip_code[1:]))
    assert (out["status"] != "rejected") == ok
    if not ok:
        assert out["estimate"] is None


def test_numeric_zip_rejected_so_leading_zeros_cannot_be_lost(art, ex):
    r = ex["lead"]
    out = call(art, r, zip_code=int(r.zip_code))
    assert out["status"] == "rejected" and "as text" in out["messages"][0]


def test_zip_from_another_city_or_state_is_rejected_without_substitution(art, ex):
    r, other = ex["any"], ex["other_city"]
    out = call(art, r, zip_code=other.zip_code)
    assert out["status"] == "rejected" and out["estimate"] is None and "not recorded for" in out["messages"][0]
    far = art.zip_map[art.zip_map.state != r.state].iloc[0]
    assert call(art, r, zip_code=far.zip_code)["status"] == "rejected"


def test_unsupported_city_rejected_even_with_valid_zip(art, ex):
    u = ex["unsupported"]
    out = call(art, u, zip_code=u.zip_code)
    assert out["status"] == "rejected" and out["location_status"] == "unsupported"


# ---- routing -----------------------------------------------------------------------------------------------------------
def test_well_covered_zip_uses_zip_model_prediction(art, ex):
    r = ex["any"]
    out = call(art, r, zip_code=r.zip_code)
    X = mc.build_zip_features(pd.DataFrame([{"state": r.state, "city": r.city, "bed": 3.0, "bath": 2.0, "house_size": 1800.0,
                                             "acre_lot": np.nan, "zip_code": r.zip_code}]))
    assert out["model_path"] == "zip_model"
    assert out["estimate"] == pytest.approx(float(mc.predict_dollars(art.zip_bundle["zip_model"], X)[0]), rel=1e-12)


@pytest.mark.parametrize("key", ["fb_zero", "fb_rare"])
def test_rare_or_unseen_zip_falls_back_to_v1_estimate(art, ex, key):
    r = ex[key]
    with_zip, without = call(art, r, zip_code=r.zip_code), call(art, r)
    assert with_zip["model_path"] == "zip_fallback" and "fewer than 5" in with_zip["path_note"]
    assert with_zip["estimate"] == without["estimate"]             # same city-level estimate...
    assert with_zip["low"] != without["low"]                       # ...with the ZIP-path calibrated range


def test_ninety_percent_range_wider_on_zip_path(art, ex):
    r = ex["any"]
    a, b = call(art, r, zip_code=r.zip_code, interval=0.8), call(art, r, zip_code=r.zip_code, interval=0.9)
    assert (b["high"] - b["low"]) > (a["high"] - a["low"]) and a["estimate"] == b["estimate"]


# ---- calibration provenance --------------------------------------------------------------------------------------------
def test_zip_multipliers_recomputed_from_calibration_split_match(art):
    zb, v1 = art.zip_bundle, art.bundle
    df = mc.to_model_frame(pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet"))
    df = df.sort_values("raw_row_id").reset_index(drop=True)
    cal = df[mc.assign_split(df.property_group) == "calibration"].reset_index(drop=True)
    X = mc.build_features(cal)
    p1 = mc.predict_dollars(v1["model"], X)
    pz = mc.predict_dollars(zb["zip_model"], mc.build_zip_features(cal))
    route = cal.zip_code.map(zb["train_zip_counts"]).fillna(0).to_numpy() >= zb["min_zip_rows"]
    ph = np.where(route, pz, p1)
    Z = np.column_stack([v1["model"].regressor_.named_steps["pre"].transform(X), np.log(ph), route.astype(float)])
    score = np.abs(np.log(cal.price.to_numpy()) - np.log(ph)) / np.maximum(zb["scale_model"].predict(Z), zb["scale_floor"])
    for name, mask in (("zip", route), ("fallback", ~route)):
        assert zb["calibration_rows"][name]["rows"] == int(mask.sum())
        for lvl, a in (("0.8", 0.2), ("0.9", 0.1)):
            assert zb["conformal_quantiles"][name][lvl] == pytest.approx(mc.conformal_quantile(score[mask], a), rel=1e-12)


def test_reported_coverage_close_to_nominal(art):
    ev = art.metadata["evaluation_test_split_reporting_only"]
    for mode in ("no_zip", "with_zip"):
        assert abs(ev[mode]["coverage"]["0.8"][0]["coverage"] - 0.8) < 0.02
        assert abs(ev[mode]["coverage"]["0.9"][0]["coverage"] - 0.9) < 0.02


# ---- location policy v2 ----------------------------------------------------------------------------------------------------
def test_support_statuses_follow_documented_rule(art):
    sp, s = art.metadata["support_policy"], art.support
    assert "validation split only" in sp["evidence"]
    mdape = pd.to_numeric(s.heldout_MdAPE, errors="coerce")
    sup = (s.n_train >= sp["supported_min_train_rows"]) & (s.heldout_rows >= sp["min_heldout_rows"]) & (mdape <= sp["max_heldout_mdape"])
    lim = ~sup & (s.n_train >= sp["limited_min_train_rows"])
    expected = np.select([sup, lim], ["supported", "limited"], "unsupported")
    assert (s.status.to_numpy() == expected).all()
    no_zip = pd.to_numeric(s.heldout_MdAPE_no_zip, errors="coerce")
    with_zip = pd.to_numeric(s.heldout_MdAPE_zip, errors="coerce")
    assert np.allclose(mdape, np.maximum(no_zip, with_zip), equal_nan=True)   # status uses the worse of both paths


def test_support_heldout_rows_are_validation_rows(art):
    df = pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet", columns=["state", "city", "property_group"])
    val = df[mc.assign_split(df.property_group) == "validation"].groupby(["state", "city"]).size().rename("v").reset_index()
    m = art.support.merge(val, on=["state", "city"], how="left").fillna({"v": 0})
    assert (m.heldout_rows == m.v.astype(int)).all()


def test_every_offered_city_has_zip_options(art):
    offered = art.support[art.support.status != "unsupported"]
    for r in offered.sample(40, random_state=1).itertuples():
        assert len(pr.zip_options(art, r.state, r.city)) > 0


# ---- artifact loading -------------------------------------------------------------------------------------------------------
def test_zip_model_loads_from_production_module(art):
    enc = art.zip_bundle["zip_model"].regressor_.named_steps["pre"].named_transformers_["loc"]
    assert type(enc).__module__ == "model_common"
    meta = json.loads((MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["version"] == art.metadata["version"] == "2.0.0-zip-hybrid"


# ---- app -----------------------------------------------------------------------------------------------------------------------
def new_app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def choose(at, r, zip_code=None):
    at.selectbox(key="state").select(r.state).run()
    at.selectbox(key="city").select(r.city).run()
    if zip_code:
        at.selectbox(key="zip").select(zip_code).run()
    return at


def texts(at):
    return "\n".join(str(e.value) for k in ("markdown", "caption", "info", "warning", "error", "success") for e in getattr(at, k))


def test_app_zip_options_constrained_to_city_and_cleared_on_change(art, ex):
    r = ex["any"]
    at = choose(new_app(), r)
    opts = [o.split()[0] for o in at.selectbox(key="zip").options]
    assert opts == list(pr.zip_options(art, r.state, r.city).zip_code)
    at.selectbox(key="zip").select(r.zip_code).run()
    other = next(c for c in at.selectbox(key="city").options if not c.startswith(r.city))
    at.selectbox(key="city").select(other.split("  (")[0]).run()
    assert at.selectbox(key="zip").value is None


def test_app_zip_estimate_shows_route_and_both_ranges(ex):
    r = ex["any"]
    at = choose(new_app(), r, r.zip_code)
    at.button(key="estimate").click().run()
    t = texts(at)
    assert not at.exception and at.metric and "ZIP-level detail" in t
    assert "80% prediction range" in t and "90% prediction range" in t and "with a ZIP code" in t


def test_app_fallback_zip_explained(ex):
    r = ex["fb_rare"]
    at = choose(new_app(), r, r.zip_code)
    at.button(key="estimate").click().run()
    assert at.metric and any("city-level estimate is used" in i.value for i in at.info)


def test_app_without_zip_unchanged_path(ex):
    at = choose(new_app(), ex["any"])
    at.button(key="estimate").click().run()
    assert at.metric and "Model used: city-level" in texts(at) and "without a ZIP code" in texts(at)


def test_mismatched_zip_rejected_by_app_logic(art, ex):
    r, other = ex["any"], ex["other_city"]
    res = appmod.run_estimate(art, {"state": r.state, "city": r.city, "zip": other.zip_code, "bed": 3, "bath": 2.0,
                                    "house_size": 1800, "acre_lot": None})
    assert res["status"] == "rejected" and "estimate" not in res


def test_app_hides_internal_errors_on_zip_path(ex, monkeypatch):
    def boom(*a, **k):
        raise KeyError(r"zip_model C:\models\zip_model.joblib")
    monkeypatch.setattr(pr, "predict_price", boom)
    r = ex["any"]
    at = choose(new_app(), r, r.zip_code)
    at.button(key="estimate").click().run()
    t = texts(at)
    assert not at.exception and not at.metric and "Something went wrong" in t
    for leak in ("KeyError", "zip_model", "joblib", "C:\\", "Traceback"):
        assert leak not in t


def test_app_figures_come_from_metadata(art):
    stats = appmod.evaluation_stats(art.metadata, "with_zip")
    ev = art.metadata["evaluation_test_split_reporting_only"]["with_zip"]
    assert stats["cov80"] == ev["coverage"]["0.8"][0]["coverage"] and stats["mdape"] == ev["metrics"]["MdAPE"]
    assert appmod.evaluation_stats({"version": "1.0"}, "no_zip") is None      # old metadata -> generic wording
    assert "under $100,000" in appmod.price_caution(90_000, None) and "under $100,000" in appmod.price_caution(90_000, {})
