"""Real-browser validation of the HomeValue AI Streamlit app (Playwright + Chromium). Not collected by pytest.

Setup (not in requirements.txt, because it is only needed for this check):
    pip install playwright && python -m playwright install chromium
    (or set PW_CHROMIUM to the path of an existing chrome.exe instead of installing)

Usage (start the app first:  streamlit run app/streamlit_app.py --server.port 8601 --server.headless true):
    python tests/browser_check.py http://127.0.0.1:8601 reports/figures/app_screenshots full
    python tests/browser_check.py http://127.0.0.1:8602 reports/figures/app_screenshots failure <served_app_root>

`full`    desktop (1366x900) and mobile (390x844) functional and layout checks against the real model.
`failure` the friendly model-load error page. Serve it from a TEMPORARY COPY of the project whose model file is broken,
          never from production files; pass that copy's root as the 4th argument.
Results go to <out>/browser_check_results.json, tagged with the SHA-256 of the app files that were actually served.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PROJECT = Path(__file__).resolve().parents[1]
URL, OUT, MODE = sys.argv[1], Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "full"
SERVED = Path(sys.argv[4]) if len(sys.argv) > 4 else PROJECT
OUT.mkdir(parents=True, exist_ok=True)
DISCLOSURE_START = "The model was trained on historical U.S. real-estate records"
obs = []


def rec(view, n, name, ok, detail=""):
    obs.append({"view": view, "check": n, "name": name, "passed": bool(ok), "observed": str(detail)[:300]})
    print(("PASS " if ok else "FAIL ") + f"[{view}] {n}: {name}" + (f"  [{str(detail)[:150]}]" if detail else ""), flush=True)


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


# ---------------------------------------------------------------------------------------------- helpers
def idle(page, extra=400):
    page.wait_for_timeout(extra)
    try:
        page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=30000)
    except Exception:
        pass
    page.wait_for_timeout(300)


def wait_ready(page):
    page.get_by_text("Smarter Insights for Every Home").first.wait_for(timeout=120000)   # cold start imports sklearn
    idle(page, 1500)


def box(page, label):
    return page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label, exact=True)).first


def open_and_type(page, label, text):
    box(page, label).locator("input").click()
    page.keyboard.type(text)
    page.wait_for_timeout(500)
    return page.get_by_role("option")


def select(page, label, text, starts_with=None):
    opts = open_and_type(page, label, text)
    opts.filter(has_text=starts_with or text).first.click()
    idle(page)


def num_input(page, label):
    return page.locator('[data-testid="stNumberInput"]').filter(has=page.get_by_text(label, exact=True)).locator("input").first


def set_number(page, label, value):
    inp = num_input(page, label)
    inp.fill(str(value))
    inp.press("Enter")
    idle(page)


def est_button(page):
    return page.get_by_role("button", name="Estimate Home Value")


def estimate(page):
    est_button(page).click()
    try:
        page.locator('[data-testid="stMetricValue"]').first.wait_for(timeout=30000)
    except Exception:
        pass
    idle(page, 600)


def metric(page):
    m = page.locator('[data-testid="stMetricValue"]')
    return m.first.inner_text().strip() if m.count() else None


def alerts(page):
    return [t.strip() for t in page.locator('[data-testid="stAlert"]').all_inner_texts()]


def body(page):
    return page.locator("body").inner_text()


def expander(page, label):
    return page.locator('[data-testid="stExpander"]').filter(has=page.get_by_text(label, exact=True)).first


def open_expander(page, label):
    ex = expander(page, label)
    ex.locator("summary").click()
    idle(page, 500)
    return ex


def ranges(page):
    return [t.strip() for t in page.locator(".hv-rtile-v").all_inner_texts()]


def leaks(page):
    t = body(page)
    bad = [w for w in ("Traceback", "Exception", "Error:", "C:\\", "D:\\", "AppData", ".joblib", "PGPASSWORD", 'File "', "pickle")
           if w in t]
    return bad + (["stException element"] if page.locator('[data-testid="stException"]').count() else [])


LAYOUT_JS = """() => {
  const vw = window.innerWidth, out = [];
  // Elements inside a deliberate horizontal scroller (the tab strip) are reachable by scrolling; the scroller itself must fit.
  const inScroller = (el) => { let a = el.parentElement; while (a && a !== document.body) {
    const ox = getComputedStyle(a).overflowX;
    if ((ox === 'auto' || ox === 'scroll') && a.scrollWidth > a.clientWidth) { const r = a.getBoundingClientRect(); return r.right <= vw + 1 && r.left >= -1; }
    a = a.parentElement; } return false; };
  document.querySelectorAll('button, input, [data-baseweb="select"], [data-testid="stMetric"], .hv-rtile, .hv-bar, .hv-pill, [data-testid="stAlert"], .hv-fact').forEach(el => {
    if (inScroller(el)) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0 && (r.right > vw + 1 || r.left < -1)) out.push((el.className || el.tagName).toString().slice(0, 40) + ' ' + Math.round(r.left) + '..' + Math.round(r.right));
  });
  let minFont = 99, tiny = 0;
  document.querySelectorAll('p, li, label, span, div, td, th').forEach(e => {
    if (e.childElementCount === 0 && e.textContent.trim().length > 2) {
      const r = e.getBoundingClientRect(); if (r.width === 0 || r.height === 0) return;
      const f = parseFloat(getComputedStyle(e).fontSize); minFont = Math.min(minFont, f); if (f < 11) tiny++;
    }});
  const lead = document.querySelector('.hv-lead'); const body = parseFloat(getComputedStyle(document.body).fontSize);
  return {vw, scrollWidth: document.documentElement.scrollWidth, outside: out.slice(0, 6), minFont, tiny, bodyFont: body,
          leadFont: lead ? parseFloat(getComputedStyle(lead).fontSize) : null, katex: document.querySelectorAll('.katex').length};
}"""


def layout(page, view, n, when):
    r = page.evaluate(LAYOUT_JS)
    rec(view, n, f"no horizontal overflow ({when})", r["scrollWidth"] <= r["vw"] + 1, f"scrollWidth={r['scrollWidth']} viewport={r['vw']}")
    rec(view, n, f"no clipped controls or tiles ({when})", not r["outside"], f"outside viewport: {r['outside']}" if r["outside"] else "all inside")
    rec(view, n, f"text readable: no text under 11px ({when})", r["tiny"] == 0, f"min font {r['minFont']}px, body {r['bodyFont']}px, intro {r['leadFont']}px")
    return r


def monitor(page, store):
    page.on("pageerror", lambda e: store["pageerrors"].append(str(e)[:120]))
    page.on("console", lambda m: store["console"].append(m.text[:120]) if m.type == "error" else None)
    page.on("request", lambda r: store["external"].append(r.url[:100]) if r.url.startswith("http") and "127.0.0.1" not in r.url and "localhost" not in r.url else None)


def shot(page, name):
    """Streamlit scrolls inside its own container, so `full_page` alone keeps the current scroll position. Enlarge the
    viewport (no script rerun), scroll the container to the top, capture, then restore the test viewport."""
    vp = page.viewport_size
    page.set_viewport_size({"width": vp["width"], "height": 4600 if vp["width"] < 600 else 2100})
    page.wait_for_timeout(800)
    page.evaluate("(() => { const m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, 0); window.scrollTo(0, 0); })()")
    page.wait_for_timeout(300)
    page.screenshot(path=str(OUT / name))
    page.set_viewport_size(vp)
    page.wait_for_timeout(500)


# ---------------------------------------------------------------------------------------------- full run
def run_full(browser):
    # ---------------- desktop ----------------
    store = {"pageerrors": [], "console": [], "external": []}
    page = browser.new_page(viewport={"width": 1366, "height": 900})
    monitor(page, store)
    page.goto(URL, wait_until="networkidle")
    wait_ready(page)
    V = "desktop 1366"
    title = page.title()
    h1 = page.locator("h1").first.inner_text().replace("\n", " ")
    rec(V, 1, "loads with HomeValue AI branding", "HomeValue AI" in title and "HomeValue" in h1 and "AI" in h1
        and "Smarter Insights for Every Home" in body(page), f"title={title!r}, h1={h1!r}")
    rec(V, 1, "loads without visible errors", not leaks(page), str(leaks(page)))
    words = len(body(page).split())
    rec(V, 16, "first screen is uncluttered: no intro paragraph or facts strip, under 70 words of visible text",
        words < 70 and "Estimate a home's recorded price" not in body(page) and page.locator(".hv-fact, .hv-intro").count() == 0,
        f"{words} visible words")
    states = page.locator('[data-testid="stExpander"] details').evaluate_all("els => els.map(e => e.open)")
    rec(V, 16, "About / Location coverage / Model limitations expanders exist and start collapsed",
        len(states) == 3 and not any(states), f"open states={states}")
    shot(page, "desktop_01_initial.png")
    layout(page, V, 14, "initial page")

    # 2, 4 supported location
    rec(V, 2, "city list disabled until a state is chosen", box(page, "City").locator("input").is_disabled())
    rec(V, 2, "estimate button disabled until state and city are chosen", est_button(page).is_disabled())
    select(page, "State", "Texas")
    rec(V, 2, "city list enabled after choosing a state", box(page, "City").locator("input").is_enabled())
    opts = open_and_type(page, "City", "Houston")
    label = opts.first.inner_text().strip() if opts.count() else ""
    rec(V, 4, "supported city shown without a limited label", label == "Houston", f"option={label!r}")
    opts.first.click()
    idle(page)
    rec(V, 4, "supported city shows no banner or warning (nothing to caution about)", alerts(page) == [], str(alerts(page)))
    rec(V, 2, "ZIP list enabled once a city is chosen", box(page, "ZIP code (optional)").locator("input").is_enabled())
    rec(V, 2, "estimate button enabled", est_button(page).is_enabled())

    # 9, 10, 11 no-ZIP v1 behaviour
    estimate(page)
    t = body(page)
    rec(V, 9, "no ZIP: estimate equals the recorded v1 value", metric(page) == "$327,000", f"metric={metric(page)}")
    rr = ranges(page)
    rec(V, 9, "no ZIP: 80% and 90% ranges equal the recorded v1 values", rr == ["$215,000 - $497,000", "$185,000 - $579,000"], str(rr))
    rec(V, 10, "dollar signs present in estimate, ranges and bar ends",
        metric(page).startswith("$") and all(x.startswith("$") for x in rr)
        and "$" in (page.locator(".hv-bar").first.get_attribute("aria-label") or ""),
        f"bar label={page.locator('.hv-bar').first.get_attribute('aria-label')}")
    rec(V, 10, "no LaTeX-rendered text anywhere (earlier $...$ defect)", page.evaluate("document.querySelectorAll('.katex').length") == 0)
    bar = {k: page.locator(f".hv-bar-{k}").first.bounding_box() for k in ("80", "est")}
    track = page.locator(".hv-bar").first.bounding_box()
    inside = (track["x"] - 1 <= bar["80"]["x"] and bar["80"]["x"] + bar["80"]["width"] <= track["x"] + track["width"] + 1
              and bar["80"]["x"] - 1 <= bar["est"]["x"] + bar["est"]["width"] / 2 <= bar["80"]["x"] + bar["80"]["width"] + 1)
    rec(V, 10, "range bar: estimate marker inside the 80% band, inside the 90% track", inside)
    rec(V, 11, "one short historical-data / not-an-appraisal note visible with the estimate",
        t.count("Not an appraisal") == 1 and "Based on historical recorded prices" in t and DISCLOSURE_START not in t,
        f"'Not an appraisal' x{t.count('Not an appraisal')}")
    rec(V, 16, "no status pills or explanatory captions in the result card",
        page.locator(".hv-pill").count() == 0 and "How to read the ranges" not in t and "Model used" not in t)
    rec(V, 16, "labels read 'Estimated Home Value', '80% Prediction Range', '90% Prediction Range'",
        "Estimated Home Value" in t and "80% Prediction Range" in t and "90% Prediction Range" in t)
    ex = open_expander(page, "About this estimate")
    et = ex.inner_text()
    rec(V, 11, "About this estimate: full disclosure word for word, once on the page, plus the range caveat",
        DISCLOSURE_START in et and "do not guarantee" in et and "not a guarantee for any one property" in et
        and body(page).count(DISCLOSURE_START) == 1)
    rec(V, 9, "no ZIP: About this estimate names the city-level model and the no-ZIP test coverage",
        "Model used: city-level (no ZIP selected)" in et and "without a ZIP code" in et)
    ex.locator("summary").click()
    idle(page, 400)
    rec(V, 16, "expander closes again", not expander(page, "About this estimate").locator("details").evaluate("e => e.open"))
    rec(V, 2, "estimate button disabled while the estimate is current (no duplicate request)", est_button(page).is_disabled())
    shot(page, "desktop_02_estimate_houston_no_zip.png")
    layout(page, V, 14, "estimate shown")

    # 12 stale
    set_number(page, "House size (sq ft)", 2500)
    rec(V, 12, "changing an input hides the stale estimate", metric(page) is None and "Inputs changed since the last estimate" in body(page))
    rec(V, 2, "estimate button re-enabled after an input change", est_button(page).is_enabled())

    # 3 state change clears city, 4 limited
    select(page, "State", "Florida")
    rec(V, 3, "changing state clears the selected city", "Houston" not in box(page, "City").inner_text()
        and "Choose a city" in box(page, "City").inner_text(), box(page, "City").inner_text().replace("\n", " "))
    opts = open_and_type(page, "City", "Alachua")
    label = opts.first.inner_text().strip() if opts.count() else ""
    rec(V, 4, "limited city labelled in the dropdown", "(limited)" in label, f"option={label!r}")
    opts.first.click()
    idle(page)
    rec(V, 4, "limited city: no limited-coverage warning before estimating (the dropdown label carries it)",
        not any("Limited coverage" in a for a in alerts(page)), str(alerts(page)))
    estimate(page)
    limited = [a for a in alerts(page) if "Limited coverage for Alachua" in a]
    rec(V, 4, "limited city: estimate shown with exactly one visible limited-coverage warning",
        metric(page) is not None and len(limited) == 1, f"metric={metric(page)}, warnings={len(limited)}")
    lc = open_expander(page, "Location coverage").inner_text()
    rec(V, 11, "limited-city testing evidence lives in the Location coverage expander",
        "of recorded prices there" in lc and "Check a city" in lc)
    expander(page, "Location coverage").locator("summary").click()
    idle(page, 400)
    shot(page, "desktop_03_limited_location_alachua.png")

    # 5 leading zeros + valid ZIP
    set_number(page, "House size (sq ft)", 1800)
    select(page, "State", "Massachusetts")
    select(page, "City", "Arlington")
    opts = open_and_type(page, "ZIP code (optional)", "02474")
    labels = [o.strip() for o in opts.all_inner_texts()]
    rec(V, 5, "ZIP option keeps its leading zero", labels[:1] == ["02474"], str(labels[:3]))
    opts.first.click()
    idle(page)
    rec(V, 5, "selected ZIP displays with its leading zero", "02474" in box(page, "ZIP code (optional)").inner_text())
    estimate(page)
    t = body(page)
    rec(V, 5, "valid ZIP: ZIP-level path with leading-zero ZIP in the result",
        page.get_by_text("Includes ZIP-level detail (02474).").count() == 1 and "02474" in t and metric(page) is not None,
        f"metric={metric(page)}")
    rec(V, 10, "ZIP path shows both ranges with dollar signs", len(ranges(page)) == 2 and all(x.startswith("$") for x in ranges(page)), str(ranges(page)))
    zt = open_expander(page, "About this estimate").inner_text()
    rec(V, 10, "ZIP path: About this estimate says ZIP-level model and 'with a ZIP code' test coverage",
        "Model used: ZIP-level detail (02474)" in zt and "with a ZIP code" in zt)
    expander(page, "About this estimate").locator("summary").click()
    idle(page, 400)
    shot(page, "desktop_04_zip_02474_zip_model.png")

    # 6, 7 invalid / foreign ZIPs cannot be chosen
    for typed, why in (("7702x", "malformed (letter)"), ("abcde", "not digits"), ("02474-1234", "ZIP+4 form"),
                       ("77002", "Houston ZIP, wrong state"), ("02108", "Boston ZIP, same state, wrong city")):
        opts = open_and_type(page, "ZIP code (optional)", typed)
        n = opts.count()
        page.keyboard.press("Enter")                 # Enter with no matching option must select nothing
        page.keyboard.press("Escape")
        page.get_by_text("Smarter Insights for Every Home").first.click()   # blur without altering the selection
        idle(page, 500)
        still = "02474" in box(page, "ZIP code (optional)").inner_text()
        rec(V, 6 if why.startswith(("malformed", "not", "ZIP+4")) else 7, f"ZIP '{typed}' ({why}) cannot be selected",
            n == 0 and still, f"{n} matching options; selection still 02474={still}")
    rec(V, 7, "after the invalid attempts the valid result is unchanged", metric(page) is not None and "02474" in box(page, "ZIP code (optional)").inner_text())

    # 8 rare ZIP fallback
    select(page, "State", "Arizona")
    select(page, "City", "Tucson")
    estimate(page)
    base = metric(page)
    opts = open_and_type(page, "ZIP code (optional)", "85619")
    label = opts.first.inner_text().strip() if opts.count() else ""
    rec(V, 8, "rare ZIP marked in the dropdown", "few records" in label, f"option={label!r}")
    opts.first.click()
    idle(page)
    estimate(page)
    t = body(page)
    rec(V, 8, "rare ZIP uses the city-level fallback: same estimate as without a ZIP, with explanation",
        metric(page) == base and sum("city-level estimate is used" in a for a in alerts(page)) == 1,
        f"without ZIP {base}, with ZIP 85619 {metric(page)}")
    shot(page, "desktop_05_rare_zip_fallback_tucson.png")

    # 13 reset (browser state, not just server state)
    set_number(page, "Bedrooms", 5)
    page.get_by_text("Include lot size").click()
    idle(page)
    set_number(page, "Lot size (acres)", 1.5)
    rec(V, 13, "before reset: inputs changed from defaults", num_input(page, "Bedrooms").input_value() == "5" and metric(page) is None)
    page.get_by_role("button", name="Reset", exact=True).click()
    idle(page, 1500)
    ph = {k: box(page, k).inner_text().replace("\n", " ") for k in ("State", "City", "ZIP code (optional)")}
    rec(V, 13, "reset: selections cleared in the browser", "Choose a state" in ph["State"] and "Tucson" not in ph["City"]
        and "Arizona" not in ph["State"] and "85619" not in ph["ZIP code (optional)"], str(ph))
    rec(V, 13, "reset: numeric inputs back to defaults in the browser",
        (num_input(page, "Bedrooms").input_value(), num_input(page, "Bathrooms").input_value(), num_input(page, "House size (sq ft)").input_value())
        == ("3", "2.0", "1800") and not page.get_by_role("checkbox", name="Include lot size").is_checked()
        and page.locator('[data-testid="stNumberInput"]').filter(has=page.get_by_text("Lot size (acres)", exact=True)).count() == 0,
        f"bed={num_input(page, 'Bedrooms').input_value()} bath={num_input(page, 'Bathrooms').input_value()} size={num_input(page, 'House size (sq ft)').input_value()}")
    rec(V, 13, "reset: result removed and empty state shown", metric(page) is None and "Your estimate will appear here" in body(page))
    shot(page, "desktop_06_after_reset.png")
    rec(V, 1, "no traceback, path or secret anywhere in the desktop session", not leaks(page), str(leaks(page)))
    rec(V, 15, "no JavaScript page errors (desktop)", not store["pageerrors"], str(store["pageerrors"][:2]))
    rec(V, 15, "no requests to external hosts (renders offline)", not store["external"], str(store["external"][:3]))
    page.close()

    # ---------------- mobile ----------------
    V = "mobile 390"
    store = {"pageerrors": [], "console": [], "external": []}
    mob = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
    monitor(mob, store)
    mob.goto(URL, wait_until="networkidle")
    wait_ready(mob)
    rec(V, 1, "loads with HomeValue AI branding", "HomeValue" in mob.locator("h1").first.inner_text() and "HomeValue AI" in mob.title())
    layout(mob, V, 14, "initial page")
    shot(mob, "mobile_01_initial.png")
    select(mob, "State", "Texas")
    select(mob, "City", "Houston")
    estimate(mob)
    rec(V, 9, "no ZIP: estimate and ranges equal the recorded v1 values",
        metric(mob) == "$327,000" and ranges(mob) == ["$215,000 - $497,000", "$185,000 - $579,000"], f"{metric(mob)} {ranges(mob)}")
    inp, res = box(mob, "State").bounding_box(), mob.locator('[data-testid="stMetricValue"]').first.bounding_box()
    rec(V, 14, "single column: result sits below the inputs", res["y"] > inp["y"] + 50, f"inputs y={round(inp['y'])}, result y={round(res['y'])}")
    b = est_button(mob).bounding_box()
    rb = mob.get_by_role("button", name="Reset", exact=True).bounding_box()
    rec(V, 14, "touch targets: Estimate button at least 40px tall and 55% of the viewport wide; Reset on the same row, 40px tall",
        b["height"] >= 40 and b["width"] >= 0.55 * 390 and rb["height"] >= 40 and abs(rb["y"] - b["y"]) < 6,
        f"estimate {round(b['width'])}x{round(b['height'])}px, reset {round(rb['width'])}x{round(rb['height'])}px, same row={abs(rb['y'] - b['y']) < 6}")
    bb, bth = num_input(mob, "Bedrooms").bounding_box(), num_input(mob, "Bathrooms").bounding_box()
    rec(V, 14, "Bedrooms and Bathrooms share one row on mobile (compact form)", abs(bb["y"] - bth["y"]) < 6, f"y={round(bb['y'])}/{round(bth['y'])}")
    lay = layout(mob, V, 14, "estimate shown")
    rec(V, 10, "no LaTeX-rendered text; dollar signs present", lay["katex"] == 0 and all(x.startswith("$") for x in ranges(mob)))
    rec(V, 11, "short not-an-appraisal note visible", "Not an appraisal" in body(mob) and DISCLOSURE_START not in body(mob))
    shot(mob, "mobile_02_estimate_houston.png")
    # long numbers: seven-figure ranges must not overflow
    select(mob, "State", "California")
    select(mob, "City", "Los Angeles")
    estimate(mob)
    lay = layout(mob, V, 14, "seven-figure ranges (Los Angeles)")
    rec(V, 10, "seven-figure estimate and ranges shown", metric(mob) is not None and len(ranges(mob)) == 2, f"{metric(mob)} {ranges(mob)}")
    shot(mob, "mobile_03_estimate_seven_figures.png")
    # ZIP on mobile
    select(mob, "State", "Massachusetts")
    select(mob, "City", "Arlington")
    select(mob, "ZIP code (optional)", "02474")
    rec(V, 5, "mobile: leading-zero ZIP selectable", "02474" in box(mob, "ZIP code (optional)").inner_text())
    estimate(mob)
    rec(V, 5, "mobile: ZIP-level result with both ranges", "ZIP-level detail (02474)" in body(mob) and len(ranges(mob)) == 2)
    layout(mob, V, 14, "ZIP result")
    # limited warning on mobile
    select(mob, "State", "Florida")
    select(mob, "City", "Alachua", starts_with="Alachua")
    estimate(mob)
    rec(V, 4, "mobile: limited-location warning visible with the estimate", any(a.startswith("Limited coverage for Alachua") for a in alerts(mob)) and metric(mob) is not None)
    layout(mob, V, 14, "limited warning")
    shot(mob, "mobile_04_limited_location.png")
    # supporting information: three collapsed expanders; each must open to a non-empty panel without overflow
    names = ["About this estimate", "Location coverage", "Model limitations"]
    opened = []
    for nm in names:
        ex = expander(mob, nm)
        ex.scroll_into_view_if_needed()
        ex.locator("summary").click()
        idle(mob, 500)
        opened.append(len(ex.inner_text().split()) > 15)
    rec(V, 14, "expanders: all three open to non-empty content on mobile with no horizontal overflow",
        all(opened) and mob.evaluate("document.documentElement.scrollWidth") <= 391, f"opened={opened}")
    layout(mob, V, 14, "all expanders open")
    mob.locator('[data-testid="stExpander"]').first.scroll_into_view_if_needed()
    shot(mob, "mobile_05_expanders_open.png")
    for nm in names:
        expander(mob, nm).locator("summary").click()
    idle(mob, 400)
    mob.get_by_role("button", name="Reset", exact=True).click()
    idle(mob, 1500)
    rec(V, 13, "mobile: reset clears selections and result in the browser",
        "Choose a state" in box(mob, "State").inner_text() and metric(mob) is None and "Your estimate will appear here" in body(mob))
    rec(V, 1, "no traceback, path or secret anywhere in the mobile session", not leaks(mob), str(leaks(mob)))
    rec(V, 15, "no JavaScript page errors (mobile)", not store["pageerrors"], str(store["pageerrors"][:2]))
    mob.close()


# ---------------------------------------------------------------------------------------------- failure page
def run_failure(browser):
    for V, vp, mobile in (("desktop 1366", {"width": 1366, "height": 900}, False), ("mobile 390", {"width": 390, "height": 844}, True)):
        store = {"pageerrors": [], "console": [], "external": []}
        page = browser.new_page(viewport=vp, is_mobile=mobile, has_touch=mobile, device_scale_factor=2 if mobile else 1)
        monitor(page, store)
        page.goto(URL, wait_until="networkidle")
        try:
            page.locator('[data-testid="stAlert"]').filter(has_text="could not be loaded").first.wait_for(timeout=120000)
        except Exception:
            pass
        idle(page, 1500)
        t = body(page)
        rec(V, 15, "friendly model-load error shown", "could not be loaded" in t and "no estimates can be made" in t, " | ".join(alerts(page))[:140])
        rec(V, 15, "no traceback, file path, secret or internal name on the page", not leaks(page), str(leaks(page)))
        rec(V, 15, "no estimate and no input form shown", metric(page) is None and est_button(page).count() == 0)
        rec(V, 15, "no JavaScript page errors", not store["pageerrors"], str(store["pageerrors"][:2]))
        if not mobile:
            layout(page, V, 15, "error page")
        shot(page, f"{'desktop' if not mobile else 'mobile'}_error_model_load_failure.png")
        page.close()


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("PW_CHROMIUM") or None)   # PW_CHROMIUM: use an already installed Chromium
    (run_full if MODE == "full" else run_failure)(browser)
    version = browser.version
    browser.close()

res_path = OUT / "browser_check_results.json"
try:
    doc = json.loads(res_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "runs" not in doc:
        doc = {"runs": {}}
except Exception:
    doc = {"runs": {}}
served = {f: sha(SERVED / f) for f in ("app/streamlit_app.py", "app/styles.css", ".streamlit/config.toml")}
doc.update({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "browser": f"Chromium {version} (Playwright)",
            "project_app_sha256": {f: sha(PROJECT / f) for f in served}})
doc["runs"][MODE] = {"served_app_sha256": served, "served_matches_project_app": served == {f: sha(PROJECT / f) for f in served},
                     "served_root_is_project": SERVED.resolve() == PROJECT.resolve(),
                     "checks": obs, "passed": sum(o["passed"] for o in obs), "total": len(obs)}
res_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
print(f"\n{sum(o['passed'] for o in obs)}/{len(obs)} checks passed ({MODE}); results -> {res_path}")
sys.exit(0 if all(o["passed"] for o in obs) else 1)
