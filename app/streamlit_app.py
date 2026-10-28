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
INTRO = "Estimate a home's recorded price using property characteristics and historical real-estate data."
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
    """Range bar (90% track, 80% band, estimate marker) plus the two ranges as text. Ends use the displayed (outward
    rounded) values, so the bar never shows a narrower range than the calibrated one."""
    lo90, hi90 = math.floor(r90[0] / 1000) * 1000, math.ceil(r90[1] / 1000) * 1000
    lo80, hi80 = math.floor(r80[0] / 1000) * 1000, math.ceil(r80[1] / 1000) * 1000
    span = max(hi90 - lo90, 1)

    def pct(v):
        return max(0.0, min(100.0, (v - lo90) / span * 100))
    label = (f"Estimate {usd(estimate)}. 80% prediction range {usd_range(*r80)}. 90% prediction range {usd_range(*r90)}.")
    return (f'<div class="hv-range" role="img" aria-label="{esc(label)}">'
            f'<div class="hv-bar"><div class="hv-bar-80" style="left:{pct(lo80):.2f}%;width:{pct(hi80) - pct(lo80):.2f}%"></div>'
            f'<div class="hv-bar-est" style="left:{pct(estimate):.2f}%"></div></div>'
            f'<div class="hv-bar-ends"><span>{esc(usd(lo90))}</span><span>{esc(usd(hi90))}</span></div></div>'
            '<div class="hv-ranges">'
            f'<div class="hv-rtile"><div class="hv-rtile-k"><span class="hv-sw hv-sw-80"></span>80% prediction range</div>'
            f'<div class="hv-rtile-v">{esc(usd_range(*r80))}</div></div>'
            f'<div class="hv-rtile"><div class="hv-rtile-k"><span class="hv-sw hv-sw-90"></span>90% prediction range</div>'
            f'<div class="hv-rtile-v">{esc(usd_range(*r90))}</div></div></div>')


# ------------------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------------------
# Widgets take their values from session state (no `value=` argument) and callbacks ASSIGN values instead of deleting
# keys: deleting a key reset the server side only, while the browser kept showing the old selection (real-browser check).
WIDGET_DEFAULTS = {"state": None, "city": None, "zip": None, **DEFAULTS, "lookup_state": None, "lookup_city": ""}


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


def section(step: int, title: str) -> None:
    html_block(f'<div class="hv-section"><span class="hv-step">{step}</span>{esc(title)}</div>')


def render_header(art) -> None:
    version = (art.metadata or {}).get("version", "") if art is not None else ""
    ready = (f'<div class="hv-ready" role="status" title="{esc("Model version " + version) if version else ""}">'
             '<span class="hv-dot"></span>Ready to estimate</div>') if art is not None else ""
    html_block('<div class="hv-header"><div class="hv-brand">'
               f'<span class="hv-logo">{HOUSE_ICON.format(c="#FFFFFF")}</span>'
               f'<div><h1 class="hv-name">HomeValue <span class="hv-ai">AI</span></h1>'
               f'<div class="hv-tagline">{esc(TAGLINE)}</div></div></div>{ready}</div>'
               f'<div class="hv-intro"><p class="hv-lead">{esc(INTRO)}</p>'
               '<p class="hv-sub">Estimates are of the <b>recorded price</b> in historical sold-status records '
               '(Oct 2021 - May 2022), with 80% and 90% prediction ranges. Not an appraisal or a current market value.</p></div>')


def render_facts(art) -> None:
    """Fact strip built only from the dataset size, location_support.csv and model_metadata.json."""
    counts = art.support.status.value_counts()
    nz, wz = evaluation_stats(art.metadata, "no_zip"), evaluation_stats(art.metadata, "with_zip")
    facts = [("Training data", "~696,000", "cleaned sold-status records"),
             ("Data period", "Oct 2021 - May 2022", "historical, not current prices"),
             ("Cities offered", f"{int(counts.get('supported', 0)):,} supported",
              f"+ {int(counts.get('limited', 0)):,} with limited coverage")]
    if nz:
        facts.append(("Median error (test)", f"{nz['mdape']:.1%}",
                      f"{wz['mdape']:.1%} with the home's ZIP" if wz else "on held-out homes"))
    html_block('<div class="hv-facts">' + "".join(
        f'<div class="hv-fact"><div class="hv-fact-k">{esc(k)}</div><div class="hv-fact-v">{esc(v)}</div>'
        f'<div class="hv-fact-s">{esc(s)}</div></div>' for k, v, s in facts) + "</div>")


def render_inputs(art, choices: dict) -> None:
    _init_state()
    html_block('<div class="hv-card-title">Property details</div>'
               '<p class="hv-card-sub">Pick a listed location, then describe the home.</p>')
    section(1, "Location")
    state = st.selectbox("State or territory", sorted(choices), index=None, placeholder="Choose a state",
                         key="state", on_change=_on_state_change,
                         help="Only states with at least one city that passed the reliability checks are listed.")
    cities = choices.get(state, [])
    status_of = {c: (s, r) for c, s, r in cities}
    city = st.selectbox("City", [c for c, _, _ in cities], index=None, key="city", disabled=state is None,
                        on_change=_on_city_change, placeholder="Choose a city" if state else "Choose a state first",
                        format_func=lambda c: f"{c}  (limited coverage)" if status_of.get(c, ("",))[0] == "limited" else c)
    if state:
        n_sup = sum(s == "supported" for _, s, _ in cities)
        st.caption(f"{n_sup} supported and {len(cities) - n_sup} limited-coverage cities in {state}. "
                   "Cities without enough reliable data are not offered.")
    if city in status_of:
        s, reason = status_of[city]
        if s == "supported":
            st.success(f"Supported location: {city}, {state}.")
        else:
            st.warning(f"Limited coverage for {city}, {state}: {reason}. An estimate is shown, but treat it with extra caution.")

    if art.zip_enabled:
        opts = pr.zip_options(art, state, city) if city in status_of else None
        route = dict(zip(opts.zip_code, opts.route)) if opts is not None else {}
        st.selectbox("ZIP code (optional)", list(route), index=None, key="zip", disabled=not route,
                     placeholder="Optional: choose a ZIP code" if route else "Choose a city first",
                     format_func=lambda z: z if route.get(z) == "zip_model" else f"{z}  (few records: city-level estimate)",
                     help="Only ZIP codes recorded for the selected city are listed. A ZIP adds neighbourhood detail when "
                          "the data has enough records for it; otherwise the city-level estimate is used. Leave it empty "
                          "for a city-level estimate.")

    section(2, "Home")
    c1, c2 = st.columns(2)
    c1.number_input("Bedrooms", min_value=1, max_value=20, step=1, key="bed")
    c2.number_input("Bathrooms", min_value=1.0, max_value=20.0, step=0.5, format="%.1f", key="bath",
                    help="The training data contains whole bathroom counts only.")
    st.number_input("House size (square feet)", min_value=200, max_value=50_000, step=50,
                    key="house_size", help="Interior living area, 200-50,000 sq ft.")
    lot_known = st.checkbox("I know the lot size", key="lot_known")
    st.number_input("Lot size (acres)", min_value=0.01, max_value=1000.0, step=0.05,
                    format="%.2f", key="acre_lot", disabled=not lot_known,
                    help="Optional. Leave unchecked if unknown; the model handles a missing lot size.")

    if st.session_state.pop("pending", False) and state and city:
        with st.spinner("Estimating..."):
            st.session_state["result"] = run_estimate(load_artifacts(), current_inputs())
    current = result_is_current()
    b1, b2 = st.columns([3, 2])
    b1.button("Estimate House Price", type="primary", key="estimate", on_click=_request_estimate,
              disabled=not (state and city) or current, use_container_width=True)
    b2.button("Reset inputs", key="reset", on_click=_reset, use_container_width=True)
    if current:
        st.caption("The estimate is up to date. Change any input to estimate again.")
    elif not (state and city):
        st.caption("Choose a state and a city to enable the estimate.")


def render_empty_state() -> None:
    steps = ["Choose a state and a city", "Add the home's size and rooms", "Press Estimate House Price"]
    html_block('<div class="hv-empty">'
               f'<div class="hv-empty-icon">{HOUSE_ICON.format(c="#0F766E")}</div>'
               '<div class="hv-empty-title">Your estimate will appear here</div>'
               '<p>Choose a location, enter the property details and press <b>Estimate House Price</b>.</p>'
               '<ol class="hv-steps">' + "".join(f'<li><span class="hv-step">{i}</span>{esc(s)}</li>'
                                                 for i, s in enumerate(steps, 1)) + '</ol>'
               '<p>You will see the estimated recorded price with its 80% and 90% prediction ranges.</p></div>')


def render_result(art) -> None:
    res = st.session_state.get("result")
    if res is None or res["status"] != "ok" or res["inputs"] != current_inputs():
        html_block('<div class="hv-card-title">Estimated price</div>')   # a shown estimate carries its own label
    if res is None:
        render_empty_state()
        return
    if res["inputs"] != current_inputs():
        st.info("The inputs have changed since the last estimate. Press **Estimate House Price** to update it.")
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
    st.metric("Estimated Recorded Price", usd(res["estimate"]),
              help="An estimate of the historical recorded price in the dataset - not an appraisal or current market value.")
    st.caption(f"{inp['city']}, {inp['state']}" + (f" {inp['zip']}" if inp.get("zip") else "")
               + f" · {inp['bed']} bed · {inp['bath']:g} bath · {inp['house_size']:,} sq ft"
               + (f" · {inp['acre_lot']:g} acres" if inp.get("acre_lot") else "") + md(" · rounded to the nearest $1,000"))
    html_block(range_html(res["estimate"], res["r80"], res["r90"]))
    if stats:
        mode_txt = "with a ZIP code" if path != "no_zip" else "without a ZIP code"
        st.caption(f"How to read the ranges: when tested on {stats['test_rows']:,} held-out homes estimated {mode_txt}, "
                   f"{stats['cov80']:.1%} fell inside their 80% range and {stats['cov90']:.1%} inside their 90% range. "
                   "That is an average over many homes, not a guarantee for this property.")
    else:
        st.caption("How to read the ranges: they are built so that, on average over many held-out homes, about 80% (or 90%) "
                   "fall inside them. That is not a guarantee for this property.")

    limited = res.get("location_status") == "limited"
    loc_pill = ('<span class="hv-pill hv-pill-warn"><b>Location</b> Limited coverage</span>' if limited
                else '<span class="hv-pill hv-pill-ok"><b>Location</b> Supported</span>')
    route_label = {"zip_model": f"ZIP-level detail ({res.get('zip_code') or inp.get('zip')})",
                   "zip_fallback": "City-level (ZIP has few records)"}.get(path, "City-level")
    html_block(f'<hr class="hv-rule"><div class="hv-status">{loc_pill}<span class="hv-pill"><b>Model</b> {esc(route_label)}</span></div>')
    if path == "zip_model":
        st.caption(f"Model used: ZIP-level detail ({res.get('zip_code') or inp.get('zip')}).")
    elif path == "zip_fallback":
        st.info(md(res.get("path_note") or "This ZIP has too few records, so the city-level estimate is used."))
    else:
        st.caption("Model used: city-level (no ZIP selected). Choosing a ZIP can add neighbourhood detail where the data allows.")

    messages = res.get("messages", [])
    if limited:
        pipeline_note = next((m for m in messages if m.startswith("Limited coverage")), "Limited-coverage location.")
        extra = (f"In testing, ranges in limited-coverage cities contained the recorded price somewhat less often "
                 f"({stats['limited80']:.0%} for the 80% range) and errors were larger than in supported cities."
                 if stats else "In testing, ranges in limited-coverage cities were less reliable than in supported cities.")
        st.warning(md(f"**{pipeline_note}**  \n{extra}"))
        messages = [m for m in messages if m != pipeline_note]
    for m in messages:
        st.warning(md(m))
    caution = price_caution(res["estimate"], stats)
    if caution:
        st.warning(md(caution))
    st.info(DISCLOSURE)


def render_city_lookup(art) -> None:
    with st.expander("Why isn't my city listed?"):
        st.write("A city is offered only if the model had enough training records for it and its held-out error was "
                 "acceptable. You can check any city's status here; no estimate is produced for unlisted cities.")
        s = st.selectbox("State", sorted(art.support.state.unique()), index=None, key="lookup_state", placeholder="Choose a state")
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


def render_about(art) -> None:
    html_block('<div class="hv-info-title">About the estimates</div>')
    nz, wz = evaluation_stats(art.metadata, "no_zip"), evaluation_stats(art.metadata, "with_zip")
    with st.container(key="hv_about"):
        t1, t2, t3, t4 = st.tabs(["How it works", "Prediction ranges", "Cities and ZIP codes", "Model limitations"])
    with t1:
        acc = ""
        if nz:
            acc = (f" On held-out test homes the median error was {nz['mdape']:.1%} of the recorded price without a ZIP"
                   + (f" and {wz['mdape']:.1%} with the home's ZIP" if wz else "") + ".")
        zip_txt = ("3. **Optional ZIP.** If you choose a ZIP with at least 5 training records, a second model adds "
                   "neighbourhood detail; ZIPs with fewer records use the city-level model. A ZIP must belong to the chosen city.\n"
                   if art.zip_enabled else "")
        st.markdown(
            "1. **Data.** About 696,000 cleaned U.S. sold-status records (Oct 2021 - May 2022) from a PostgreSQL data warehouse.\n"
            "2. **Model.** A gradient-boosting model predicts the logarithm of the recorded price from state, city, bedrooms, "
            "bathrooms, house size and (optionally) lot size. Locations are summarised from training data only." + acc + "\n"
            + zip_txt +
            f"{4 if art.zip_enabled else 3}. **Ranges.** Split conformal prediction: separate held-out homes were used to "
            "calibrate how wide the 80% and 90% ranges must be, scaled to how uncertain each kind of home is.\n"
            f"{5 if art.zip_enabled else 4}. **Location policy.** Cities need at least 30 training records to be offered; "
            "'supported' cities also need at least 100 records and acceptable held-out error. Everything else is refused "
            "rather than guessed.")
        if art.metadata and art.metadata.get("version"):
            st.caption(f"Model version {art.metadata['version']}.")
    with t2:
        st.markdown("The **estimate** is the model's single best guess. The **80% and 90% prediction ranges** show the "
                    "uncertainty around it: they were calibrated so that, over many held-out homes, about 80% (or 90%) of "
                    "recorded prices fell inside them. They describe an average over many homes, **not a guarantee for any "
                    "one property**, and the 90% range is always the wider of the two. Displayed range ends are rounded "
                    "outward to whole thousands.")
        rows = [(label, s) for label, s in (("Without a ZIP", nz), ("With the home's ZIP", wz)) if s]
        if rows:
            st.markdown(md("| Held-out test homes | Inside 80% range | Inside 90% range | 80% range, recorded under $100k | "
                           "80% range, recorded over $2M |\n|---|---|---|---|---|\n" + "\n".join(
                               f"| {label} ({s['test_rows']:,}) | {s['cov80']:.1%} | {s['cov90']:.1%} | "
                               f"{s['band80']['<100k']:.0%} | {s['band80']['>2M']:.0%} |" for label, s in rows)))
            st.caption(md(f"Coverage was weaker at the extremes of the market and in limited-coverage cities "
                          f"({nz['limited80']:.0%} for the 80% range without a ZIP), so the app warns in those cases."
                          if nz else "Coverage was weaker at the extremes of the market and in limited-coverage cities."))
    with t3:
        counts = art.support.status.value_counts()
        st.markdown(
            f"- **Supported** ({int(counts.get('supported', 0)):,} cities): at least 100 training records and acceptable "
            "held-out error. Estimates are shown normally.\n"
            f"- **Limited coverage** ({int(counts.get('limited', 0)):,} cities): at least 30 training records but too few to "
            "verify accuracy, or larger held-out error. An estimate is shown with a warning.\n"
            "- **Not offered**: fewer than 30 training records. No estimate is given; use *Why isn't my city listed?* to "
            "check a city.\n"
            + ("- **ZIP codes** are optional. Only ZIPs recorded for the chosen city are listed, with leading zeros kept. A "
               "ZIP with at least 5 training records uses the ZIP-level model; a ZIP with fewer records is labelled and uses "
               "the city-level estimate. A ZIP that does not belong to the chosen city is rejected, never substituted."
               if art.zip_enabled else "- **ZIP codes** are not used by the installed model."))
    with t4:
        no_city = sorted(set(art.support.state) - set(art.support[art.support.status != "unsupported"].state))
        st.markdown(
            "- **Historical data.** Prices are recorded prices from about October 2021 to May 2022, not current values; the "
            "source does not document them as verified completed-sale prices.\n"
            "- **Location coverage.** Only cities with enough reliable data are offered. Cities absent from training were "
            "estimated far less accurately in testing, so they are refused."
            + (f" No city is available in: {', '.join(no_city)}." if no_city else "") + "\n"
            "- **ZIP codes.** Only ZIPs recorded for the chosen city in this dataset are accepted, and the dataset's own "
            "ZIP-city pairs may contain errors. A wrong but listed ZIP gives a confidently wrong estimate.\n"
            "- **Uncertainty.** The ranges hold on average, not for every home; they are least reliable for very cheap and very "
            "expensive homes and in limited-coverage cities.\n"
            "- **Missing detail.** The model does not know the exact address, condition, age, view or renovations.\n"
            "- **Not an appraisal.** Use the result as a rough historical reference, never as a professional valuation.")
    st.caption(DISCLOSURE)
    version = (art.metadata or {}).get("version")
    html_block(f'<div class="hv-footer"><span>{esc(APP_NAME)} · {esc(TAGLINE)}</span>'
               f'<span>{esc("Model " + version + " · " if version else "")}Historical reference, not an appraisal</span></div>')


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} · House Price Estimates", page_icon="🏠", layout="wide")
    inject_styles()
    try:
        art = load_artifacts()
        choices = location_choices(art)
    except Exception:
        log.exception("could not load model artifacts")
        render_header(None)
        st.error("The prediction model could not be loaded, so no estimates can be made right now. "
                 "Please ask the maintainer to check the model files.")
        st.stop()

    render_header(art)
    render_facts(art)
    left, right = st.columns([11, 10], gap="large")
    with left:
        with st.container(key="hv_form"):
            render_inputs(art, choices)
        render_city_lookup(art)
    with right:
        with st.container(key="hv_result"):
            render_result(art)
    render_about(art)


if __name__ == "__main__":
    main()
