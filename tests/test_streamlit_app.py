"""Phase 5 checks for the Streamlit app: startup, location choices, policy enforcement, results, failure handling.
Run from the project root:  python -m pytest tests -q
Uses the saved artifacts in models/; nothing is retrained. UI behaviour is driven with streamlit.testing.v1.AppTest."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "src"))
import predict as pr  # noqa: E402

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "house_price_model.joblib").exists(), reason="train the model first")

# The app module is imported (not run) for its helper functions: main() runs only under `streamlit run` / AppTest.
spec = importlib.util.spec_from_file_location("streamlit_app_module", APP)
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)


@pytest.fixture(scope="module")
def art():
    return pr.load_artifacts()


@pytest.fixture(scope="module")
def picks(art):
    s = art.support
    sup = s[s.status == "supported"].sort_values("n_train", ascending=False).iloc[0]
    lim = s[s.status == "limited"].sort_values("n_train", ascending=False).iloc[0]
    uns = s[(s.status == "unsupported") & s.state.isin(s[s.status == "supported"].state)].iloc[0]
    return {"supported": (sup.state, sup.city), "limited": (lim.state, lim.city), "unsupported": (uns.state, uns.city)}


def new_app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def choose(at, state, city):
    at.selectbox(key="state").select(state).run()
    at.selectbox(key="city").select(city).run()
    return at


def page_text(at) -> str:
    parts = [e.value for kind in ("markdown", "caption", "info", "warning", "error", "success", "title", "subheader")
             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts) + "\n" + "\n".join(str(m.value) for m in at.metric)


# ---- startup and artifacts ---------------------------------------------------------------------------
def test_app_starts_without_exceptions():
    at = new_app()
    assert not at.exception
    text = page_text(at)
    assert appmod.APP_NAME == "HomeValue AI" and appmod.TAGLINE in text
    assert "Your estimate will appear here" in text    # empty state before any estimate
    assert at.button(key="estimate").disabled          # nothing to estimate until a location is chosen
    assert at.button(key="estimate").label == "Estimate Home Value"
    assert appmod.DISCLOSURE in text


def test_artifacts_load_and_policy_columns_present(art):
    assert {"state", "city", "status", "reason"} <= set(art.support.columns)
    assert set(art.support.status) <= {"supported", "limited", "unsupported"}
    assert set(art.bundle["conformal_quantiles"]) == {"0.8", "0.9"}


# ---- location choices --------------------------------------------------------------------------------
def test_state_and_city_choices_match_policy(art):
    choices = appmod.location_choices(art)
    allowed = art.support[art.support.status.isin(["supported", "limited"])]
    assert sum(len(v) for v in choices.values()) == len(allowed)
    for state, cities in choices.items():
        for city, status, _ in cities:
            assert status in ("supported", "limited")
            assert ((allowed.state == state) & (allowed.city == city)).any()


def test_city_list_is_filtered_by_state_and_resets(picks):
    at = new_app()
    st_sup, city_sup = picks["supported"]
    at.selectbox(key="state").select(st_sup).run()
    opts = at.selectbox(key="city").options
    assert any(o.startswith(city_sup) for o in opts)
    other = next(s for s in at.selectbox(key="state").options if s != st_sup)
    at.selectbox(key="city").select(city_sup).run()
    at.selectbox(key="state").select(other).run()
    assert at.selectbox(key="city").value is None       # never silently keeps or substitutes a city


def test_unsupported_city_is_not_offered(picks):
    state, city = picks["unsupported"]
    at = new_app()
    at.selectbox(key="state").select(state).run()
    assert not any(o == city or o.startswith(city + "  (") for o in at.selectbox(key="city").options)


# ---- estimates -----------------------------------------------------------------------------------------
def test_supported_location_shows_estimate_and_both_ranges(picks):
    at = choose(new_app(), *picks["supported"])
    assert not any("(limited)" in o for o in at.selectbox(key="city").options if o.startswith(picks["supported"][1]))
    assert not at.warning                                                    # a supported city needs no warning
    at.button(key="estimate").click().run()
    assert not at.exception
    assert at.metric and at.metric[0].value.startswith("$")
    assert not any("Limited coverage" in w.value for w in at.warning)
    text = page_text(at)
    assert "80% prediction range" in text and "90% prediction range" in text
    assert "not a guarantee" in text and appmod.DISCLOSURE in text


def test_limited_location_shows_prominent_warning(picks):
    at = choose(new_app(), *picks["limited"])
    assert any(o.startswith(picks["limited"][1]) and "(limited)" in o for o in at.selectbox(key="city").options)
    at.button(key="estimate").click().run()
    assert at.metric                                                         # an estimate is still shown...
    assert any(w.value.startswith("**Limited coverage") for w in at.warning)  # ...with the pipeline's warning, prominently


def test_unsupported_location_cannot_get_estimate(art, picks):
    state, city = picks["unsupported"]
    res = appmod.run_estimate(art, {"state": state, "city": city, "bed": 3, "bath": 2.0, "house_size": 1800, "acre_lot": None})
    assert res["status"] == "rejected" and "estimate" not in res
    assert any("no estimate" in m for m in res["messages"])


def test_changed_inputs_hide_stale_estimate(picks):
    at = choose(new_app(), *picks["supported"])
    at.button(key="estimate").click().run()
    assert at.metric
    at.number_input(key="house_size").set_value(2500).run()
    assert not at.metric and any("changed since the last estimate" in i.value for i in at.info)


def test_reset_clears_inputs_and_result(picks):
    at = choose(new_app(), *picks["supported"])
    at.number_input(key="bed").set_value(5).run()
    at.button(key="estimate").click().run()
    at.button(key="reset").click().run()
    assert at.selectbox(key="state").value is None and at.number_input(key="bed").value == 3 and not at.metric


# ---- invalid and extreme input ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [{"bed": 0}, {"bed": 2.5}, {"bed": True}, {"bath": 25}, {"house_size": 100},
                                 {"house_size": float("nan")}, {"acre_lot": -1.0}, {"acre_lot": 5000.0}, {"state": None}])
def test_invalid_inputs_rejected_before_prediction(art, picks, bad, monkeypatch):
    called = []
    monkeypatch.setattr(pr, "predict_price", lambda *a, **k: called.append(1))
    state, city = picks["supported"]
    inp = {"state": state, "city": city, "bed": 3, "bath": 2.0, "house_size": 1800, "acre_lot": None, **bad}
    res = appmod.run_estimate(art, inp)
    assert res["status"] == "rejected" and res["messages"] and not called


def test_extreme_but_legal_input_warns(picks):
    at = choose(new_app(), *picks["supported"])
    at.number_input(key="house_size").set_value(45_000).run()
    at.button(key="estimate").click().run()
    assert not at.exception
    assert any("unusual" in w.value for w in at.warning)


def test_optional_lot_size_is_passed_only_when_known(picks, monkeypatch):
    seen = []
    real = pr.predict_price
    monkeypatch.setattr(pr, "predict_price", lambda *a, **k: seen.append(a[6]) or real(*a, **k))
    at = choose(new_app(), *picks["supported"])
    at.button(key="estimate").click().run()
    at.checkbox(key="lot_known").check().run()
    at.number_input(key="acre_lot").set_value(0.5).run()
    at.button(key="estimate").click().run()
    assert seen[:2] == [None, None] and seen[-1] == 0.5


def test_price_caution_for_extreme_estimates():
    assert "under $100,000" in appmod.price_caution(90_000)
    assert "over $2 million" in appmod.price_caution(2_500_000)
    assert appmod.price_caution(400_000) is None


def test_displayed_range_never_narrower_than_calibrated():
    lo, hi = 215_432.0, 496_811.0
    shown = appmod.usd_range(lo, hi).replace("$", "").replace(",", "").split(" - ")
    assert float(shown[0]) <= lo and float(shown[1]) >= hi


# ---- failures never leak internals or fabricate a number -------------------------------------------------------
def test_prediction_exception_shows_generic_error(picks, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError(r"secret C:\internal\path PGPASSWORD=xyz")
    monkeypatch.setattr(pr, "predict_price", boom)
    at = choose(new_app(), *picks["supported"])
    at.button(key="estimate").click().run()
    assert not at.exception and not at.metric
    text = page_text(at)
    assert "Something went wrong" in text
    for leak in ("Traceback", "RuntimeError", "secret", "PGPASSWORD", "internal"):
        assert leak not in text


def test_inconsistent_prediction_output_is_not_shown(art, picks, monkeypatch):
    monkeypatch.setattr(pr, "predict_price", lambda *a, **k: {"status": "ok", "estimate": float("nan"), "low": 1, "high": 2,
                                                               "messages": []})
    state, city = picks["supported"]
    res = appmod.run_estimate(art, {"state": state, "city": city, "bed": 3, "bath": 2.0, "house_size": 1800, "acre_lot": None})
    assert res["status"] == "error" and "estimate" not in res


def test_pipeline_rejection_is_displayed(picks, monkeypatch):
    monkeypatch.setattr(pr, "predict_price", lambda *a, **k: {"status": "rejected", "estimate": None, "low": None, "high": None,
                                                               "messages": ["House size must be between 200 and 50,000 square feet."]})
    at = choose(new_app(), *picks["supported"])
    at.button(key="estimate").click().run()
    assert not at.metric and any("House size must be" in e.value for e in at.error)


def test_model_load_failure_is_handled(monkeypatch):
    def fail(*a, **k):
        raise FileNotFoundError(r"C:\models\house_price_model.joblib")
    monkeypatch.setattr(pr, "load_artifacts", fail)
    import streamlit as st
    st.cache_resource.clear()
    at = new_app()
    st.cache_resource.clear()      # do not leave a failed state cached for the other tests
    assert not at.exception and not at.metric
    text = page_text(at)
    assert "could not be loaded" in text
    for leak in ("joblib", "C:\\", "FileNotFoundError", "Traceback"):
        assert leak not in text


def test_city_lookup_explains_unsupported_city(picks):
    state, city = picks["unsupported"]
    at = new_app()
    at.selectbox(key="lookup_state").select(state).run()
    at.text_input(key="lookup_city").input(city.lower()).run()
    assert any("Not supported - no estimate" in m.value for m in at.markdown)
    assert not at.metric


# ---- redesigned UI (HomeValue AI) -------------------------------------------------------------------------
def test_estimate_button_disabled_while_estimate_is_current(picks):
    at = choose(new_app(), *picks["supported"])
    assert not at.button(key="estimate").disabled
    at.button(key="estimate").click().run()
    assert at.metric and at.button(key="estimate").disabled           # no duplicate submission of the same inputs
    at.number_input(key="bed").set_value(4).run()
    assert not at.metric and not at.button(key="estimate").disabled   # changed inputs: stale result hidden, button enabled


def test_failed_estimate_can_be_retried(picks, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(pr, "predict_price", boom)
    at = choose(new_app(), *picks["supported"])
    at.button(key="estimate").click().run()
    assert "Something went wrong" in page_text(at) and not at.button(key="estimate").disabled


def test_range_bar_uses_displayed_rounded_values_and_escapes():
    out = appmod.range_html(500_400, (410_600, 610_200), (380_900, 700_100))
    assert "80% prediction range" in out and "90% prediction range" in out
    assert "&#36;410,000 - &#36;611,000" in out and "&#36;380,000 - &#36;701,000" in out
    assert "$" not in out                                               # never parsed as LaTeX by markdown
    assert appmod.esc('<b>"x"</b> & $1') == "&lt;b&gt;&quot;x&quot;&lt;/b&gt; &amp; &#36;1"


def test_result_card_shows_location_and_model_status(picks):
    at = choose(new_app(), *picks["limited"])
    at.button(key="estimate").click().run()
    warnings = [w.value for w in at.warning if "Limited coverage" in w.value]
    assert len(warnings) == 1                                                # shown once, not repeated in the form and the card
    assert "hv-pill" not in page_text(at)


def test_page_is_simplified_one_disclosure_and_collapsed_expanders(picks):
    at = new_app()
    assert [e.label for e in at.expander] == ["About this estimate", "Location coverage", "Model limitations"]
    assert not any(e.proto.expanded for e in at.expander)                    # secondary explanations start collapsed
    assert page_text(at).count(appmod.DISCLOSURE) == 1
    at = choose(at, *picks["supported"])
    at.button(key="estimate").click().run()
    text = page_text(at)
    assert text.count(appmod.DISCLOSURE) == 1                                # still one copy next to the estimate
    assert appmod.NOTE in text and "Estimated Home Value" in [m.label for m in at.metric]
    assert not at.warning                                                    # nothing to caution about for this estimate


def test_lot_size_field_only_appears_when_included(picks):
    def has_lot(at):
        return any(n.key == "acre_lot" for n in at.number_input)
    at = choose(new_app(), *picks["supported"])
    assert not has_lot(at)                                                   # hidden until the user opts in
    at.checkbox(key="lot_known").check().run()
    assert has_lot(at) and at.number_input(key="acre_lot").value == appmod.DEFAULTS["acre_lot"]
    at.checkbox(key="lot_known").uncheck().run()
    assert not has_lot(at)
