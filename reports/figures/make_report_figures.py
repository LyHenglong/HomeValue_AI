"""Regenerates the figures used in reports/final_report.md from existing project artifacts only.

    python reports/figures/make_report_figures.py        (run from the project root)

Inputs : reports/*.csv, reports/experiment_summary.json, models/model_metadata.json,
         reports/example_query_results/q1_state_summary.csv (output of sql/example_analytical_query.sql),
         data/processed/sold_clean.parquet and models/*.joblib (read-only, for the price histogram and the
         predicted-vs-recorded plot).
Outputs: reports/figures/eda_*.png and results_*.png.
Style  : one palette (blue / orange), text in ink colours, recessive grid, direct value labels, a legend whenever two
         series share a plot, and the source printed under every figure. No figure uses a second y-axis.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a7a69f"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

dollars = FuncFormatter(lambda v, _: f"${v:,.0f}")
thousands = FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


def finish(fig, name, source):
    """Save the chart, then append the source line in its own strip underneath, so it can never overlap a label or legend."""
    path = FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    img = plt.imread(path)
    h, w = img.shape[:2]
    extra = 44
    out = plt.figure(figsize=(w / 150, (h + extra) / 150), dpi=150, facecolor=SURFACE)
    ax = out.add_axes([0, extra / (h + extra), 1, h / (h + extra)])
    ax.imshow(img)
    ax.axis("off")
    out.text(0.012 * 1500 / w, extra * 0.5 / (h + extra), f"Source: {source}", fontsize=8, color=INK2, ha="left", va="center")
    out.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(out)
    print("wrote", name)


def money(v, _=None):
    return f"${v / 1e6:g}M" if v >= 1e6 else f"${v / 1e3:g}k"


def hbar(ax, labels, values, colors, fmt, pad=0.012, fontsize=9.5, room=1.22):
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    top = max(values)
    for yi, v in zip(y, values):
        ax.text(v + top * pad, yi, fmt(v), va="center", fontsize=fontsize, color=INK)
    ax.set_xlim(0, top * room)


# ------------------------------------------------------------------------------------------------ EDA
def fig_status_missing():
    st = pd.read_csv(REP / "phase1_status_summary.csv").sort_values("rows", ascending=False)
    n_raw = int(st.rows.sum())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    cols = [BLUE if s == "sold" else GREY for s in st.status]
    fig.subplots_adjust(wspace=0.55)
    hbar(ax[0], st.status, st.rows, cols, lambda v: f"{v:,.0f} ({v / n_raw:.1%})", room=1.62)
    ax[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax[0].set_xlabel("rows in the raw file")
    ax[0].set_title(f"Raw rows by listing status (n = {n_raw:,})")
    ax[0].text(0.99, 0.03, "blue = status modelled (sold)", transform=ax[0].transAxes, ha="right", fontsize=9, color=BLUE)

    raw = pd.read_csv(REP / "phase1_schema_missingness.csv", index_col=0).missing_pct
    clean = pd.read_csv(REP / "cleaning_missingness.csv", index_col=0).missing_pct_sold_clean
    cols_show = [c for c in raw.sort_values(ascending=False).index if raw[c] >= 0.05 or clean[c] > 0]
    y = np.arange(len(cols_show))
    h = 0.38
    ax[1].barh(y - h / 2, raw[cols_show], h, color=BLUE, label="All raw rows (n = 2,226,382)")
    ax[1].barh(y + h / 2, clean[cols_show], h, color=ORANGE, label="Cleaned sold rows used for modelling (n = 696,287)")
    for yi, c in zip(y, cols_show):
        ax[1].text(raw[c] + 0.4, yi - h / 2, f"{raw[c]:.1f}%", va="center", fontsize=9)
        ax[1].text(clean[c] + 0.4, yi + h / 2, f"{clean[c]:.1f}%", va="center", fontsize=9)
    ax[1].set_yticks(y, cols_show)
    ax[1].invert_yaxis()
    ax[1].grid(axis="y", visible=False)
    ax[1].set_xlim(0, max(raw[cols_show]) * 1.18)
    ax[1].set_xlabel("% of rows with a missing value")
    ax[1].set_title("Missing values by column")
    ax[1].legend(loc="lower right", frameon=False, fontsize=9)
    finish(fig, "eda_status_and_missingness.png", "reports/phase1_status_summary.csv, phase1_schema_missingness.csv, cleaning_missingness.csv")


def fig_price_distribution():
    p = pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet", columns=["price"]).price.to_numpy()
    q = np.quantile(p, [0.05, 0.5, 0.95])
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.hist(np.log10(p), bins=np.arange(3.6, 7.95, 0.05), color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    ax.set_xticks([4, 5, 6, 7], ["$10,000", "$100,000", "$1,000,000", "$10,000,000"])
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.3)          # head-room so the labels sit above the bars
    for v, lab, ha, dx in zip(q, ["5th percentile", "median", "95th percentile"], ["right", "left", "left"], [-0.03, 0.03, 0.03]):
        ax.axvline(np.log10(v), color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax.text(np.log10(v) + dx, ymax * 1.26, f"{lab}\n${v:,.0f}", fontsize=9, va="top", ha=ha, color=INK)
    ax.set_xlabel("recorded price (logarithmic scale)")
    ax.set_ylabel("number of homes per bin")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(f"Recorded price of the {len(p):,} cleaned sold-status records (maximum ${p.max():,.0f})")
    finish(fig, "eda_price_distribution.png", "data/processed/sold_clean.parquet (price, bins of 0.05 in log10 units)")


def fig_state_coverage():
    s = pd.read_csv(REP / "cleaning_state_coverage.csv").sort_values("rows", ascending=False)
    top = s.head(15)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    hbar(ax, top.state, top.rows, BLUE, lambda v: f"{v:,.0f}")
    ax.set_xlabel("cleaned sold-status records")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    small = s[s.rows < 100].sort_values("rows")
    ax.set_title(f"Cleaned records by state: the 15 largest of {len(s)} states and territories")
    ax.text(0.99, 0.04, f"{len(small)} states/territories have fewer than 100 records\n(e.g. {small.state.iloc[0]}: {int(small.rows.iloc[0])})",
            transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    finish(fig, "eda_state_coverage.png", "reports/cleaning_state_coverage.csv")


def fig_median_by_state():
    q = pd.read_csv(REP / "example_query_results" / "q1_state_summary.csv").sort_values("median_price", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    hbar(ax, q.state, q.median_price, BLUE, lambda v: f"${v / 1000:,.0f}k", fontsize=8.8)
    labels = [f"{s}  (n = {n:,})" for s, n in zip(q.state, q.homes)]
    ax.set_yticks(np.arange(len(q)), labels, fontsize=8.8)
    ax.xaxis.set_major_formatter(thousands)
    ax.set_xlabel("median recorded price")
    ax.set_title(f"Median recorded price by state ({len(q)} states with at least 1,000 sold records)")
    finish(fig, "eda_median_price_by_state.png", "warehouse query 1, sql/example_analytical_query.sql (dw.fact_property_listing joined to dw.dim_location)")


# ------------------------------------------------------------------------------------------------ results
def fig_model_comparison():
    s = json.load(open(REP / "experiment_summary.json"))
    meta = json.load(open(ROOT / "models" / "model_metadata.json"))
    m = {r["model"]: r for r in s["model_comparison"]}
    hz = meta["evaluation_test_split_reporting_only"]["with_zip"]["metrics"]
    rows = [("Median-price baseline", m["dummy_median"]["test_MAE"], m["dummy_median"]["test_MdAPE"], GREY),
            ("Ridge regression (log price)", m["ridge_log"]["test_MAE"], m["ridge_log"]["test_MdAPE"], GREY),
            ("Gradient boosting (raw price)", m["hist_gb_raw"]["test_MAE"], m["hist_gb_raw"]["test_MdAPE"], GREY),
            ("Random forest (log price)", m["random_forest_log"]["test_MAE"], m["random_forest_log"]["test_MdAPE"], GREY),
            ("Gradient boosting (log price) = v1, no ZIP", m["hist_gb_log"]["test_MAE"], m["hist_gb_log"]["test_MdAPE"], BLUE),
            ("Hybrid ZIP model = v2", hz["MAE"], hz["MdAPE"], ORANGE)]
    rows.sort(key=lambda r: -r[1])
    labels = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    hbar(ax[0], labels, [r[1] for r in rows], [r[3] for r in rows], lambda v: f"${v:,.0f}")
    ax[0].xaxis.set_major_formatter(thousands)
    ax[0].set_xlabel("mean absolute error (MAE), dollars; lower is better")
    ax[0].set_title("Test MAE")
    hbar(ax[1], labels, [r[2] for r in rows], [r[3] for r in rows], lambda v: f"{v:.1%}")
    ax[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax[1].set_xlabel("median absolute percentage error; lower is better")
    ax[1].set_title("Test median % error")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, BLUE, ORANGE)]
    fig.legend(handles, ["candidate model", "selected on validation (v1)", "final model with optional ZIP (v2)"], loc="upper center",
               ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.03))
    finish(fig, "results_model_comparison.png", f"reports/experiment_summary.json and models/model_metadata.json; test split, n = {int(hz['n']):,}")


def fig_coverage_by_band():
    meta = json.load(open(ROOT / "models" / "model_metadata.json"))["evaluation_test_split_reporting_only"]
    order = ["<100k", "100-200k", "200-400k", "400-700k", "700k-1M", "1-2M", ">2M"]
    nice = {"<100k": "under $100k", "100-200k": "$100-200k", "200-400k": "$200-400k", "400-700k": "$400-700k", "700k-1M": "$700k-1M",
            "1-2M": "$1-2M", ">2M": "over $2M"}
    def get(mode):
        return {r["value"]: r for r in meta[mode]["coverage"]["0.8"] if r["group"] == "price_band"}
    a, b = get("no_zip"), get("with_zip")
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, [a[k]["coverage"] for k in order], w, color=BLUE, label="Without ZIP (v1 path)")
    ax.bar(x + w / 2, [b[k]["coverage"] for k in order], w, color=ORANGE, label="With the home's ZIP (v2 hybrid)")
    tag = dict(ha="center", fontsize=9, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8))   # keeps labels legible over the dashed line
    for xi, k in zip(x, order):
        ax.text(xi - w / 2, a[k]["coverage"] + 0.012, f"{a[k]['coverage']:.0%}", **tag)
        ax.text(xi + w / 2, b[k]["coverage"] + 0.012, f"{b[k]['coverage']:.0%}", **tag)
    ax.axhline(0.8, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(len(order) - 0.45, 0.805, "nominal 80%", fontsize=9, color=INK2, ha="right", va="bottom")
    ax.set_xticks(x, [f"{nice[k]}\n(n = {a[k]['n']:,})" for k in order])
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("share of homes inside their 80% range")
    ax.set_xlabel("recorded price band (conditioning on the recorded price, so the extremes drift)")
    ax.grid(axis="x", visible=False)
    ax.set_title("80% prediction-range coverage by recorded-price band (test split)")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9.5)
    finish(fig, "results_coverage_by_price_band.png", "models/model_metadata.json (evaluation_test_split_reporting_only); n = 139,100 test homes")


def fig_seen_unseen():
    g = pd.DataFrame(json.load(open(REP / "experiment_summary.json"))["geographic_holdout"]["results"])
    order = ["all", "<30", "30-99", "100-499", "500+"]
    seen = g[g.evaluation.str.startswith("seen")].set_index("city_size").reindex(order)
    unseen = g[g.evaluation.str.startswith("held-out")].set_index("city_size").reindex(order)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5))
    ax.bar(x - w / 2, seen.MdAPE, w, color=BLUE, label="Cities that were in the training data")
    ax.bar(x + w / 2, unseen.MdAPE, w, color=ORANGE, label="Cities held out of training entirely")
    for xi, (s, u) in enumerate(zip(seen.MdAPE, unseen.MdAPE)):
        ax.text(xi - w / 2, s + 0.006, f"{s:.1%}", ha="center", fontsize=9)
        ax.text(xi + w / 2, u + 0.006, f"{u:.1%}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{'all cities' if o == 'all' else o + ' rows'}\n{int(seen.n[o]):,} seen\n{int(unseen.n[o]):,} held out" for o in order], fontsize=9)
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("median absolute percentage error")
    ax.set_xlabel("city size (rows in the dataset); under each label: homes evaluated, seen | held out")
    ax.grid(axis="x", visible=False)
    ax.set_title("Seen versus never-seen cities (v1 model, retrained without 2,341 held-out cities)")
    ax.legend(loc="upper left", ncol=1, frameon=False, fontsize=9.5)
    finish(fig, "results_seen_vs_unseen_cities.png", "reports/experiment_summary.json (geographic_holdout)")


def fig_actual_vs_predicted():
    import joblib
    import model_common as mc
    df = mc.to_model_frame(pd.read_parquet(ROOT / "data" / "processed" / "sold_clean.parquet",
                                           columns=["raw_row_id", "price", "bed", "bath", "house_size", "acre_lot", "state", "city", "zip_code", "property_group"]))
    t = df[mc.assign_split(df.property_group) == "test"].reset_index(drop=True)
    v1 = joblib.load(ROOT / "models" / "house_price_model.joblib")["model"]
    zb = joblib.load(ROOT / "models" / "zip_model.joblib")
    p1 = mc.predict_dollars(v1, mc.build_features(t))
    pz = mc.predict_dollars(zb["zip_model"], mc.build_zip_features(t))
    route = t.zip_code.map(zb["train_zip_counts"]).fillna(0).to_numpy() >= zb["min_zip_rows"]
    ph = np.where(route, pz, p1)
    mae = float(np.abs(ph - t.price.to_numpy()).mean())
    expected = json.load(open(ROOT / "models" / "model_metadata.json"))["evaluation_test_split_reporting_only"]["with_zip"]["metrics"]["MAE"]
    assert abs(mae - expected) < 1e-6 * expected, f"reproduced MAE {mae} differs from recorded {expected}"
    rs = np.random.default_rng(42).choice(len(t), size=40000, replace=False)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    hb = ax.hexbin(t.price.to_numpy()[rs], ph[rs], gridsize=70, xscale="log", yscale="log", bins="log", mincnt=1, cmap="Blues")
    lim = [2e4, 6e6]
    ax.plot(lim, lim, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(3.2e6, 1.9e6, "perfect\nprediction", fontsize=8.5, color=INK2, ha="center", rotation=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.xaxis.set_major_formatter(FuncFormatter(money)); ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.set_xlabel("recorded price (logarithmic scale)"); ax.set_ylabel("predicted price (logarithmic scale)")
    ax.set_title("Hybrid model: predicted versus recorded price", pad=22)
    ax.text(0, 1.015, f"test split, random sample of 40,000 of {len(t):,} homes; full-test MAE ${mae:,.0f}", transform=ax.transAxes, fontsize=9, color=INK2)
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("homes per hexagon (logarithmic scale)", color=INK2)
    finish(fig, "results_actual_vs_predicted_hybrid.png", "models/house_price_model.joblib + models/zip_model.joblib applied to the test split of data/processed/sold_clean.parquet")


def fig_star_schema():
    """Static diagram of the warehouse. Tables, columns and relationships are exactly those of sql/create_warehouse.sql
    (key columns and the columns used in the report; see the DDL for the complete list)."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13.4, 8.4))
    ax.set_xlim(0, 13.4); ax.set_ylim(0, 8.4); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, title, lines, head, schema):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#ffffff", ec=GREY, lw=1.2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.5), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc=head, ec=head))
        ax.text(x + 0.12, y + h - 0.25, title, color="#ffffff", fontsize=10.5, fontweight="bold", va="center")
        ax.text(x + w - 0.12, y + h - 0.25, schema, color="#ffffff", fontsize=8.5, va="center", ha="right")
        for i, (txt, bold) in enumerate(lines):
            ax.text(x + 0.14, y + h - 0.82 - i * 0.3, txt, fontsize=8.8, va="center", color=INK, fontweight="bold" if bold else "normal")

    LX, RX, W, FX, FW = 0.3, 9.9, 3.2, 4.95, 3.5          # left/right dimension x, dimension width, fact x and width
    box(LX, 5.0, W, 2.9, "dim_location", [("location_key  PK", True), ("state", False), ("city", False), ("zip_code  CHAR(5)", False),
                                          ("is_us_territory", False), ("unique (state, city, zip_code)", False)], BLUE, "dw")
    box(RX, 5.0, W, 2.9, "dim_property", [("property_dim_key  PK", True), ("bedrooms", False), ("bathrooms", False), ("house_size_band", False),
                                          ("lot_size_band", False), ("unique (all four columns)", False)], BLUE, "dw")
    box(LX, 0.3, W, 3.5, "dim_date", [("date_key  PK  (yyyymmdd)", True), ("full_date", False), ("year, quarter, month", False),
                                      ("month_name, day, iso_week", False), ("day_of_week, day_name", False), ("is_weekend", False),
                                      ("built from prev_sold_date only", False)], BLUE, "dw")
    box(RX, 0.3, W, 3.5, "audit.etl_run", [("run_id  PK", True), ("status, started_at, finished_at", False), ("raw_sha256", False),
                                           ("clean_parquet_sha256", False), ("fact_rows_inserted/updated/deleted", False),
                                           ("error_message", False), ("(also referenced by", False), (" audit.quarantine_record)", False)], GREY, "audit")
    box(FX, 0.3, FW, 7.6, "fact_property_listing", [
        ("raw_row_id  PK", True), ("(grain: one retained source record)", False), ("location_key  FK", True), ("property_dim_key  FK", True),
        ("prev_sold_date_key  FK (nullable)", True), ("etl_run_id  FK", True), ("", False), ("price  NUMERIC(14,2)", False),
        ("house_size_sqft  INTEGER", False), ("acre_lot  NUMERIC(10,4), nullable", False), ("price_per_sqft  NUMERIC(12,2)", False), ("", False),
        ("acre_lot_was_zero, repeated_property", False), ("property_key, property_group", False), ("brokered_by_code, street_code", False),
        ("source_status  = 'sold'", False), ("", False), ("CHECK: price > 0, house_size_sqft > 0", False)], ORANGE, "dw")
    # foreign keys: the arrow starts at the fact table's foreign key and ends at the dimension's primary key
    for (x1, x2, y, label) in ((FX, LX + W, 6.0, "location_key"), (FX + FW, RX, 6.0, "property_dim_key"),
                               (FX, LX + W, 1.9, "prev_sold_date_key"), (FX + FW, RX, 1.9, "etl_run_id")):
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))
        ax.text((x1 + x2) / 2, y + 0.16, label, fontsize=7.4, color=INK2, ha="center")
    ax.text(6.7, 8.2, "Warehouse star schema (arrows run from a foreign key in the fact table to the primary key it references)", ha="center", fontsize=10.5,
            color=INK, fontweight="bold")
    finish(fig, "warehouse_star_schema.png", "sql/create_warehouse.sql (key and report-relevant columns only; the staging tables are not shown)")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (fig_status_missing, fig_price_distribution, fig_state_coverage, fig_median_by_state, fig_model_comparison,
              fig_coverage_by_band, fig_seen_unseen, fig_actual_vs_predicted, fig_star_schema):
        f()
