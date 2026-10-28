"""HomeValue AI (House Price Recommendation System) - Streamlit front end.

Launch from the project root:   streamlit run app/streamlit_app.py

The app only reads the saved artifacts in models/ (no database connection, no retraining). Every estimate comes from
src/predict.py, which enforces the location policy, the ZIP-city check and input validation; this file adds the UI, a
pre-check of the inputs and responsible presentation of the result. Evaluation figures shown to users are read from
models/model_metadata.json, so they always describe the installed model version.
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import predict as pr  # noqa: E402

MODELS = ROOT / "models"
STYLES = Path(__file__).resolve().parent / "styles.css"
ARTIFACT_FILES = ["house_price_model.joblib", "location_support.csv", "model_metadata.json", "zip_model.joblib", "zip_city_map.csv"]
log = logging.getLogger("house_price_app")

DISCLOSURE = ("The model was trained on historical U.S. real-estate records from approximately October 2021 to May 2022. "
              "Results describe estimated recorded prices in that dataset and may differ substantially from current market "
              "values. Prediction ranges are uncertain and do not guarantee that a property will sell within the displayed range.")
LOW_PRICE_WARN_BELOW, HIGH_PRICE_WARN_FROM = 150_000, 1_000_000
DEFAULTS = {"bed": 3, "bath": 2.0, "house_size": 1800, "lot_known": False, "acre_lot": 0.25}
INPUT_KEYS = ["state", "city", "zip", *DEFAULTS]

APP_NAME = "HomeValue AI"
TAGLINE = "Smarter Insights for Every Home"
# Inline SVG icons (no image files or network requests).
HOUSE_ICON = ('<svg viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20h14V9.5"/>'
              '<path d="M10 20v-5.5h4V20"/></svg>')


# ------------------------------------------------------------------------------------------------
# Artifacts (cached; the cache key includes file modification times so replaced artifacts are picked up)
# ------------------------------------------------------------------------------------------------
def _artifact_key() -> tuple:
    return tuple((f, (MODELS / f).stat().st_mtime_ns if (MODELS / f).exists() else None) for f in ARTIFACT_FILES)


@st.cache_resource(show_spinner="Loading the model...")
def _load_artifacts(key: tuple):
    return pr.load_artifacts(MODELS)


def load_artifacts():
    return _load_artifacts(_artifact_key())


def location_choices(art) -> dict:
    """{state: [(city, status, reason), ...]} for supported and limited cities only, from location_support.csv."""
    sup = art.support[art.support.status.isin(["supported", "limited"])].sort_values(["state", "city"])
    out: dict = {}
    for r in sup.itertuples(index=False):
        out.setdefault(r.state, []).append((r.city, r.status, r.reason))
    return out


def evaluation_stats(meta: dict | None, mode: str) -> dict | None:
    """Test-split figures for one request mode ('no_zip' or 'with_zip') from model_metadata.json; None if unavailable."""
    try:
        ev = meta["evaluation_test_split_reporting_only"][mode]

        def cov(lvl, group, value):
            return next(r["coverage"] for r in ev["coverage"][lvl] if r["group"] == group and r["value"] == value)
        return {"cov80": cov("0.8", "all", "all"), "cov90": cov("0.9", "all", "all"), "mdape": ev["metrics"]["MdAPE"],
                "medae": ev["metrics"]["MedAE"], "test_rows": int(ev["metrics"]["n"]),
                "band80": {b: cov("0.8", "price_band", b) for b in ("<100k", "1-2M", ">2M")},
                "limited80": cov("0.8", "support_status", "limited"), "version": meta.get("version", "")}
    except Exception:  # older metadata (e.g. after a rollback): the app shows generic wording instead
        return None


def _metadata_from_disk() -> dict | None:
    try:
        return json.loads((MODELS / "model_metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested without the UI)
# ------------------------------------------------------------------------------------------------
def md(text: str) -> str:
    """Escape dollar signs for Streamlit's markdown elements; otherwise a pair like "$215,000 - $497,000" is rendered as
    LaTeX math (found in the real-browser check). Apply only at render time; helper functions return plain text."""
    return str(text).replace("$", "\\$")


def usd(x: float) -> str:
    return f"${round(x / 1000) * 1000:,.0f}"


def usd_range(lo: float, hi: float) -> str:
    """Round outward to whole thousands so the displayed range is never narrower than the calibrated one."""
    return f"${math.floor(lo / 1000) * 1000:,.0f} - ${math.ceil(hi / 1000) * 1000:,.0f}"


def validate_inputs(inp: dict) -> list[str]:
    """Light pre-check before calling the prediction pipeline (which validates again, authoritatively)."""
    errors = []
    if not inp.get("state") or not inp.get("city"):
        errors.append("Choose a state and a city from the lists.")

    def num(name):
        v = inp.get(name)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
    bed, bath, size = num("bed"), num("bath"), num("house_size")
    if bed is None or not (1 <= bed <= 20) or bed != int(bed):
        errors.append("Bedrooms must be a whole number from 1 to 20.")
    if bath is None or not (1 <= bath <= 20):
        errors.append("Bathrooms must be between 1 and 20.")
    if size is None or not (200 <= size <= 50_000):
        errors.append("House size must be between 200 and 50,000 square feet.")
    if inp.get("acre_lot") is not None:
        lot = num("acre_lot")
        if lot is None or not (0 < lot <= 1000):
            errors.append("Lot size must be above 0 and at most 1,000 acres, or marked as unknown.")
    z = inp.get("zip")
    if z is not None and not (isinstance(z, str) and re.fullmatch(r"\d{5}", z)):
        errors.append("ZIP code must be one of the listed five-digit codes, or left empty.")
    return errors


def _usable(r: dict) -> bool:
    vals = [r.get("estimate"), r.get("low"), r.get("high")]
    return all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in vals) and r["low"] <= r["estimate"] <= r["high"]


def run_estimate(art, inp: dict) -> dict:
    """Call src/predict.py for the 80% and 90% ranges. Never returns a made-up number: anything unexpected -> 'error'."""
    errors = validate_inputs(inp)
    if errors:
        return {"inputs": inp, "status": "rejected", "messages": errors}
    args = (art, inp["state"], inp["city"], inp["bed"], inp["bath"], inp["house_size"], inp.get("acre_lot"))
    zip_kw = {"zip_code": inp["zip"]} if inp.get("zip") else {}
    try:
        r80 = pr.predict_price(*args, interval=0.8, **zip_kw)
        if r80.get("status") == "rejected":
            return {"inputs": inp, "status": "rejected", "messages": list(r80.get("messages") or ["No estimate for these inputs."])}
        r90 = pr.predict_price(*args, interval=0.9, **zip_kw)
        if r90.get("status") == "rejected" or not (_usable(r80) and _usable(r90)) \
                or not math.isclose(r80["estimate"], r90["estimate"], rel_tol=1e-9):
            raise ValueError("prediction output failed consistency checks")
        notes = [m for m in r80.get("messages", []) if not m.startswith("About ")]  # drop the generic coverage line
        return {"inputs": inp, "status": r80["status"], "estimate": r80["estimate"], "r80": (r80["low"], r80["high"]),
                "r90": (r90["low"], r90["high"]), "location_status": r80.get("location_status"), "messages": notes,
                "model_path": r80.get("model_path", "no_zip"), "path_note": r80.get("path_note")}
    except Exception:
        log.exception("prediction failed")  # details go to the server log only, never to the page
        return {"inputs": inp, "status": "error"}


def price_caution(estimate: float, stats: dict | None = None) -> str | None:
    """Warn where interval coverage was weaker in testing. `stats` defaults to the installed model's no-ZIP figures."""
    if stats is None:
        stats = evaluation_stats(_metadata_from_disk(), "no_zip")
    band = (stats or {}).get("band80", {})
    if estimate < LOW_PRICE_WARN_BELOW:
        if "<100k" in band:
            return (f"Ranges are least reliable at the low end of the market: in testing, the 80% range contained only "
                    f"{band['<100k']:.0%} of homes whose recorded price was under $100,000.")
        return "Ranges are least reliable at the low end of the market (homes recorded under $100,000)."
    if estimate >= HIGH_PRICE_WARN_FROM:
        if "1-2M" in band and ">2M" in band:
            return (f"Ranges are less reliable for expensive homes: in testing, the 80% range contained {band['1-2M']:.0%} "
                    f"of homes recorded at $1-2 million and {band['>2M']:.0%} of homes over $2 million.")
        return "Ranges are less reliable for expensive homes ($1-2 million and over $2 million)."
    return None


def esc(text) -> str:
    """HTML-escape text for the custom HTML blocks; '$' becomes an entity so markdown never reads it as LaTeX."""
    return html.escape(str(text)).replace("$", "&#36;")


def range_html(estimate: float, r80: tuple, r90: tuple) -> str:
    """Range bar (90% track, 80% band, estimate marker) plus the two ranges as text. The bar uses the displayed (outward
    rounded) ends, so it never shows a narrower range than the calibrated one."""
    lo90, hi90 = math.floor(r90[0] / 1000) * 1000, math.ceil(r90[1] / 1000) * 1000
    lo80, hi80 = math.floor(r80[0] / 1000) * 1000, math.ceil(r80[1] / 1000) * 1000
    span = max(hi90 - lo90, 1)

    def pct(v):
        return max(0.0, min(100.0, (v - lo90) / span * 100))
    label = f"Estimate {usd(estimate)}. 80% prediction range {usd_range(*r80)}. 90% prediction range {usd_range(*r90)}."
    return (f'<div class="hv-bar" role="img" aria-label="{esc(label)}">'
            f'<div class="hv-bar-80" style="left:{pct(lo80):.2f}%;width:{pct(hi80) - pct(lo80):.2f}%"></div>'
            f'<div class="hv-bar-est" style="left:{pct(estimate):.2f}%"></div></div>'
            '<div class="hv-ranges">'
            f'<div class="hv-rtile"><div class="hv-rtile-k"><span class="hv-sw hv-sw-80"></span>80% Prediction Range</div>'
            f'<div class="hv-rtile-v">{esc(usd_range(*r80))}</div></div>'
            f'<div class="hv-rtile"><div class="hv-rtile-k"><span class="hv-sw hv-sw-90"></span>90% Prediction Range</div>'
            f'<div class="hv-rtile-v">{esc(usd_range(*r90))}</div></div></div>')


# ------------------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------------------
# Widgets take their values from session state (no `value=` argument) and callbacks ASSIGN values instead of deleting
# keys: deleting a key reset the server side only, while the browser kept showing the old selection (real-browser check).
WIDGET_DEFAULTS = {"state": None, "city": None, "zip": None, **DEFAULTS, "lookup_state": None, "lookup_city": ""}
NOTE = ("Based on historical recorded prices (Oct 2021 - May 2022). Not an appraisal or a guaranteed current market value.")


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_styles() -> None:
    try:
        html_block(f"<style>{STYLES.read_text(encoding='utf-8')}</style>")
    except OSError:  # a missing stylesheet only changes the look, never the behaviour
        log.warning("styles.css not found; using the default theme")


def _init_state():
    for k, v in WIDGET_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def _on_state_change():
    st.session_state["city"] = None          # a new state never keeps a city or ZIP from another state
    st.session_state["zip"] = None


def _on_city_change():
    st.session_state["zip"] = None           # a ZIP always belongs to the currently selected city


def _reset():
    for k, v in WIDGET_DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.pop("result", None)
    st.session_state.pop("pending", None)


def _request_estimate():
    # The estimate runs in the script body (so the spinner shows next to the button), before the button is drawn,
    # which lets the button be disabled in the same run once the estimate is up to date.
    st.session_state["pending"] = True


def current_inputs() -> dict:
    ss = st.session_state
    return {"state": ss.get("state"), "city": ss.get("city"), "zip": ss.get("zip"),
            "bed": ss.get("bed", DEFAULTS["bed"]), "bath": ss.get("bath", DEFAULTS["bath"]),
            "house_size": ss.get("house_size", DEFAULTS["house_size"]),
            "acre_lot": ss.get("acre_lot", DEFAULTS["acre_lot"]) if ss.get("lot_known") else None}


def result_is_current() -> bool:
    """True when the shown result matches the inputs; a failed attempt ('error') can always be retried."""
    res = st.session_state.get("result")
    return res is not None and res["status"] != "error" and res["inputs"] == current_inputs()


def displayed_estimate() -> dict | None:
    """The result currently shown as an estimate (current inputs, not an error or rejection), else None."""
    res = st.session_state.get("result")
    return res if result_is_current() and res["status"] not in ("error", "rejected") else None


def render_header() -> None:
    html_block('<div class="hv-brand">'
               f'<span class="hv-logo">{HOUSE_ICON.format(c="#FFFFFF")}</span>'
               f'<div><h1 class="hv-name">HomeValue <span class="hv-ai">AI</span></h1>'
               f'<div class="hv-tagline">{esc(TAGLINE)}</div></div></div>')


def render_inputs(art, choices: dict) -> None:
    _init_state()
    html_block('<div class="hv-card-title">Property Details</div>')
    c1, c2 = st.columns(2)
    state = c1.selectbox("State", sorted(choices), index=None, placeholder="Choose a state", key="state",
                         on_change=_on_state_change)
    cities = choices.get(state, [])
    status_of = {c: (s, r) for c, s, r in cities}
    city = c2.selectbox("City", [c for c, _, _ in cities], index=None, key="city", disabled=state is None,
                        on_change=_on_city_change, placeholder="Choose a city" if state else "Choose a state first",
                        format_func=lambda c: f"{c}  (limited)" if status_of.get(c, ("",))[0] == "limited" else c)

    row = st.columns(2)
    size_col = row[0]
    if art.zip_enabled:
        opts = pr.zip_options(art, state, city) if city in status_of else None
        route = dict(zip(opts.zip_code, opts.route)) if opts is not None else {}
        row[0].selectbox("ZIP code (optional)", list(route), index=None, key="zip", disabled=not route,
                         placeholder="Optional" if route else "Choose a city first",
                         format_func=lambda z: z if route.get(z) == "zip_model" else f"{z}  (few records)",
                         help="Only ZIP codes recorded for the selected city are listed. ZIPs marked (few records) use the city-level estimate.")
        size_col = row[1]
    size_col.number_input("House size (sq ft)", min_value=200, max_value=50_000, step=50, key="house_size")

    with st.container(key="hv_pair"):
        b1, b2 = st.columns(2)
        b1.number_input("Bedrooms", min_value=1, max_value=20, step=1, key="bed")
        b2.number_input("Bathrooms", min_value=1.0, max_value=20.0, step=0.5, format="%.1f", key="bath",
                        help="The training data contains whole bathroom counts only.")
    if st.checkbox("Include lot size", key="lot_known"):
        st.number_input("Lot size (acres)", min_value=0.01, max_value=1000.0, step=0.05, format="%.2f", key="acre_lot")

    if st.session_state.pop("pending", False) and state and city:
        with st.spinner("Estimating..."):
            st.session_state["result"] = run_estimate(load_artifacts(), current_inputs())
    with st.container(key="hv_actions"):
        a1, a2 = st.columns([3, 1.2])
        a1.button("Estimate Home Value", type="primary", key="estimate", on_click=_request_estimate,
                  disabled=not (state and city) or result_is_current(), use_container_width=True)
        a2.button("Reset", key="reset", on_click=_reset, use_container_width=True)


def render_empty_state() -> None:
    html_block('<div class="hv-empty">'
               f'<div class="hv-empty-icon">{HOUSE_ICON.format(c="#0F766E")}</div>'
               '<div class="hv-empty-title">Your estimate will appear here</div>'
               '<p>Enter the property details, then press <b>Estimate Home Value</b>.</p></div>')


def render_result(art) -> None:
    res = st.session_state.get("result")
    if res is None:
        render_empty_state()
        return
    if res["inputs"] != current_inputs():
        st.info("Inputs changed since the last estimate. Press **Estimate Home Value** to update it.")
        return
    if res["status"] == "error":
        st.error("Something went wrong while producing the estimate, so no price is shown. Please try again; "
                 "if the problem continues, the model files may need to be checked.")
        return
    if res["status"] == "rejected":
        st.error(md("No estimate is given for these inputs:\n\n" + "\n".join(f"- {m}" for m in res["messages"])))
        return

    inp, path = res["inputs"], res.get("model_path", "no_zip")
    stats = evaluation_stats(art.metadata, "no_zip" if path == "no_zip" else "with_zip")
    st.metric("Estimated Home Value", usd(res["estimate"]))
    st.caption(f"{inp['city']}, {inp['state']}" + (f" {inp['zip']}" if inp.get("zip") else "")
               + f" · {inp['bed']} bed · {inp['bath']:g} bath · {inp['house_size']:,} sq ft"
               + (f" · {inp['acre_lot']:g} acres" if inp.get("acre_lot") else ""))
    html_block(range_html(res["estimate"], res["r80"], res["r90"]))

    # Status and warnings: only what changes how this particular estimate should be read.
    if path == "zip_model":
        st.caption(f"Includes ZIP-level detail ({res.get('zip_code') or inp.get('zip')}).")
    elif path == "zip_fallback":
        st.info(md(res.get("path_note") or "This ZIP has too few records, so the city-level estimate is used."))
    messages = res.get("messages", [])
    if res.get("location_status") == "limited":
        pipeline_note = next((m for m in messages if m.startswith("Limited coverage")), "Limited-coverage location.")
        st.warning(md(f"**{pipeline_note}**"))
        messages = [m for m in messages if m != pipeline_note]
    for m in messages:
        st.warning(md(m))
    caution = price_caution(res["estimate"], stats)
    if caution:
        st.warning(md(caution))
    st.caption(NOTE)


def render_about_estimate(art) -> None:
    nz, wz = evaluation_stats(art.metadata, "no_zip"), evaluation_stats(art.metadata, "with_zip")
    res = displayed_estimate()
    with st.expander("About this estimate"):
        acc = (f" The median error on held-out test homes was {nz['mdape']:.1%} of the recorded price without a ZIP"
               + (f" and {wz['mdape']:.1%} with the home's ZIP" if wz else "") + ".") if nz else ""
        st.markdown(md("A gradient-boosting model estimates the **recorded price** from the location, bedrooms, bathrooms, "
                       "house size and, optionally, lot size. It was trained on about 696,000 cleaned sold-status records "
                       "from October 2021 to May 2022." + acc + " Estimates are rounded to the nearest $1,000."))
        st.markdown("The **80% and 90% prediction ranges** were calibrated on separate held-out homes so that about 80% "
                    "(or 90%) of their recorded prices fell inside. That is an average over many homes, not a guarantee "
                    "for any one property. Range ends are rounded outward to whole thousands.")
        if res:
            path = res.get("model_path", "no_zip")
            zip_code = res.get("zip_code") or res["inputs"].get("zip")
            route = {"zip_model": f"ZIP-level detail ({zip_code})",
                     "zip_fallback": "city-level (the ZIP has too few records)"}.get(path, "city-level (no ZIP selected)")
            stats = wz if path != "no_zip" else nz
            mode_txt = "with a ZIP code" if path != "no_zip" else "without a ZIP code"
            st.markdown(f"Model used: {route}." + (
                f" In testing, {stats['cov80']:.1%} of {stats['test_rows']:,} held-out homes estimated {mode_txt} fell inside "
                f"their 80% range and {stats['cov90']:.1%} inside their 90% range." if stats else ""))
        st.markdown(DISCLOSURE)
        if art.metadata and art.metadata.get("version"):
            st.caption(f"Model version {art.metadata['version']}.")


def render_location_coverage(art) -> None:
    counts = art.support.status.value_counts()
    nz = evaluation_stats(art.metadata, "no_zip")
    with st.expander("Location coverage"):
        st.markdown(
            f"- **Supported** ({int(counts.get('supported', 0)):,} cities): at least 100 training records and acceptable "
            "held-out error.\n"
            f"- **Limited coverage** ({int(counts.get('limited', 0)):,} cities): at least 30 training records, but too few "
            "to verify accuracy or a larger held-out error. An estimate is shown with a warning"
            + (md(f"; in testing, the 80% range contained {nz['limited80']:.0%} of recorded prices there.") if nz else ".") + "\n"
            "- **Not offered**: fewer than 30 training records. No estimate is given rather than a guess.\n"
            + ("- **ZIP codes** are optional and must belong to the chosen city. A ZIP with at least 5 training records adds "
               "neighbourhood detail; with fewer, the city-level estimate is used. Any other ZIP is rejected."
               if art.zip_enabled else "- **ZIP codes** are not used by the installed model."))
        st.markdown("**Check a city**")
        s = st.selectbox("State to check", sorted(art.support.state.unique()), index=None, key="lookup_state",
                         placeholder="Choose a state")
        q = st.text_input("City name", key="lookup_city", max_chars=80)
        if s and q.strip():
            sup = art.support
            hit = sup[(sup.state == s) & (sup.city.str.casefold() == q.strip().casefold())]
            if hit.empty:
                st.write(f"'{q.strip()}' does not appear among the cleaned sold records for {s}, so no estimate is available.")
            else:
                r = hit.iloc[0]
                label = {"supported": "Supported", "limited": "Limited coverage (estimate shown with a warning)",
                         "unsupported": "Not supported - no estimate"}[r.status]
                st.write(f"**{r.city}, {r.state}:** {label}. Reason: {r.reason}.")


def render_limitations(art) -> None:
    stats = evaluation_stats(art.metadata, "no_zip")
    no_city = sorted(set(art.support.state) - set(art.support[art.support.status != "unsupported"].state))
    band = (f" In testing, the 80% range contained {stats['band80']['<100k']:.0%} of homes recorded under $100,000 and "
            f"{stats['band80']['>2M']:.0%} of homes over $2 million.") if stats else ""
    with st.expander("Model limitations"):
        st.markdown(md(
            "- **Historical data.** Prices are recorded prices from about October 2021 to May 2022, not current values; "
            "the source does not document them as verified completed-sale prices.\n"
            "- **Very low and high prices.** Ranges are least reliable for homes recorded under $100,000 and for homes over "
            "$1 million." + band + "\n"
            "- **Missing detail.** The model does not know the exact address, condition, age, view or renovations.\n"
            "- **ZIP codes.** The dataset's ZIP-city pairs are not independently verified. A wrong but listed ZIP gives a "
            "confidently wrong ZIP-level estimate.\n"
            "- **Coverage.** Cities absent from training were estimated far less accurately in testing, so they are refused."
            + (f" No city is available in: {', '.join(no_city)}." if no_city else "")))


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🏠", layout="wide")
    inject_styles()
    render_header()
    try:
        art = load_artifacts()
        choices = location_choices(art)
    except Exception:
        log.exception("could not load model artifacts")
        st.error("The prediction model could not be loaded, so no estimates can be made right now. "
                 "Please ask the maintainer to check the model files.")
        st.stop()

    left, right = st.columns([10, 9], gap="large")
    with left:
        with st.container(key="hv_form"):
            render_inputs(art, choices)
    with right:
        with st.container(key="hv_result"):
            render_result(art)
    with st.container(key="hv_more"):
        render_about_estimate(art)
        render_location_coverage(art)
        render_limitations(art)


if __name__ == "__main__":
    main()
