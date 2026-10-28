# House Price Recommendation System

A Data Warehouse and Data Mining final project. It cleans a U.S. real-estate listing dataset, loads it into a
PostgreSQL star schema, trains and evaluates price models, and serves estimates with calibrated 80% and 90% prediction
ranges in a Streamlit web app.

> **What the estimate means.** The model predicts the *recorded price* of `sold`-status records from the
> USA Real Estate dataset (Realtor.com listings), dated about **October 2021 to May 2022**. The source does not document
> these as verified completed-sale prices. Results are a historical reference, **not an appraisal or a current market value**.

## Objective and scope

The project shows data-warehousing and data-mining methods end to end: profile and clean a public listing dataset, model
it as a star schema in PostgreSQL, compare regression models, and serve validated estimates with calibrated ranges. The
scope is U.S. `sold`-status records with complete bedroom, bathroom and house-size data. Other listing statuses are kept
in the raw file but not modelled. Current market valuation, appraisal and forecasting are out of scope.

## Data and cleaning

- **Source:** the "USA Real Estate Dataset" on Kaggle, uploaded by Ahmed Shahriar Sakib (version 25, last updated 2024-03-30):
  <https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset>. It was downloaded as
  `data/raw/realtor-data.zip.csv.zip`; it has 2,226,382 rows and 12 columns, and its SHA-256 is recorded in
  `reports/cleaning_summary.json`. The raw file is never modified.
  - **Licence and terms (from Kaggle's public metadata, retrieved 2026-10-28 with a web-fetch tool):** the licence field reads
    "Other (specified in description)", so the terms are the description's. It says: "The data and information in the data set
    provided here are intended to use for educational purposes only." and "I do not own any data, and all rights are reserved to
    the respective owners." It names Realtor.com (operated by Move, Inc., a News Corp subsidiary) as the source of the listings.
    **Use this project for coursework only, and do not use it commercially.** The raw data are not committed to this repository.
  - **Please recheck in a browser:** the quotes above are as the fetch tool returned them, and the Kaggle page itself could not be
    rendered. Also unverified: the data dictionary (meaning of `price`, `status` and `prev_sold_date`), the update schedule, and the
    exact file list. The Kaggle description refers to a 10-column file `realtor-data.csv`, while the downloaded file has 12 columns,
    so it comes from a later version; the version of the download was not recorded.
  - **Repository note:** this repository publishes derived statistics and trained models, not the raw data. Whether publishing
    those fits the educational-use terms was not confirmed; keep the repository private if in doubt. Kaggle may require a free
    login to download the data.
- **Target:** the recorded `price` of `sold`-status records. The dataset's documentation for this field was not found, so it
  is not described as a verified completed-sale price. For sold rows, `prev_sold_date` falls between 2021-10-18 and
  2022-05-06. It is probably the sale date, but that is not verified, so it is not used as a feature.
- **Cleaning** (`src/data_cleaning.py`, documented rules, counts per step in `reports/cleaning_waterfall.csv`):
  - Location: 1,606 rows without a state, city or ZIP are excluded. One invalid state and 99 rows whose state disagrees
    with their ZIP are quarantined. ZIP codes are restored to 5-character strings, keeping leading zeros.
  - Status: 811,892 `sold` rows remain in scope.
  - Missing values: 245 rows with no price are excluded; prices are never imputed. 114,698 sold rows that are missing
    bedrooms, bathrooms or house size are excluded from the complete-case data. The 114,401 of them whose present values
    are all valid are kept separately for the missing-data experiment.
  - Implausible values are quarantined rather than deleted: price under $1,000 or over $100M, more than 20 bedrooms or
    bathrooms, house size outside 200-50,000 sq ft, price per sq ft outside $5-$10,000, and lots over 1,000 acres.
    Lot size 0 is treated as unknown.
  - 58 exact duplicate sales are removed.
  - Result: **696,287 clean rows** and 704 quarantined rows with reasons (`data/processed/`, `reports/cleaning_report.md`).
- **Known data-quality limits:**
  - Missing values are heavy in the raw data: bedrooms 21.6% and house size 25.5% across all rows.
  - Coverage by state is uneven, and Guam has no complete sold rows.
  - The property key (street code plus ZIP) is imperfect.
  - The source's own ZIP-city pairs are not independently verified.

## Data warehouse

PostgreSQL 17 star schema (`sql/create_warehouse.sql`, `reports/warehouse_schema.md`):
- **Fact table:** `dw.fact_property_listing` has one row per retained sold record (grain = one source row; key
  `raw_row_id`) and holds the price, size and lot measures.
- **Dimensions:** `dw.dim_location` (state, city, ZIP), `dw.dim_property` (bedrooms, bathrooms, size bands) and
  `dw.dim_date` (`prev_sold_date`, neutral naming because its meaning is unverified).
- **Staging and audit:** `staging.*` tables and `audit.etl_run` / `audit.quarantine_record` provide traceability.

`src/warehouse_etl.py` loads in one transaction and is safe to rerun (a second load changes 0 rows). It checks row counts,
NOT NULL, key uniqueness, foreign keys, row-level staging-to-fact equality, measure totals and constraint probes. On the
local PostgreSQL 17.2 server it passed **52/52 checks**, with 696,287 fact rows, 22,128 locations, 1,939 property types,
201 dates and 704 quarantine records (`reports/warehouse_validation.md`).

## Modelling and evaluation

- **Splits:** each property group is hashed into train 60%, validation 10%, calibration 10% or test 20% (418,165 /
  69,769 / 69,253 / 139,100 rows). Every row of a property shares a split, and tests assert this.
  - Validation chose the models and settings and set the location-support policy.
  - Calibration set the interval widths.
  - Test was used only for reporting.
- **Leakage safeguards:** no target-derived features (for example price per sq ft). Location encodings are learned on
  training rows only and cross-fitted. Imputation is fitted on training rows. Identifiers and the unverified date are
  excluded.
- **Model comparison** (test split, reported after selection on validation MAE; `reports/model_comparison.md`):

  | Model | MAE | RMSE | R2 | Median % error |
  |---|---:|---:|---:|---:|
  | Median baseline | $282,014 | $712,215 | -0.041 | 43.8% |
  | Ridge (log price) | $124,699 | $424,927 | 0.629 | 17.5% |
  | Random forest (log price) | $111,134 | $424,167 | 0.631 | 15.1% |
  | **Gradient boosting (log price), v1** | **$106,982** | **$383,269** | **0.699** | **14.9%** |
  | Gradient boosting (raw price) | $113,117 | $381,184 | 0.702 | 15.7% |

- **ZIP model (v2):** adding ZIP as a third location level helped (`reports/zip_feature_experiment.md`).
  - A plain ZIP model failed a pre-registered check on unseen ZIPs.
  - The chosen *hybrid* uses the ZIP model only for ZIPs with at least 5 training rows and v1 otherwise. It reaches
    test MAE $98,281, R2 0.745 and a median error of 13.8%.
- **Ranges:** normalised split-conformal prediction gives 80% and 90% ranges. Test coverage is in the results summary below.
- **Geography:** cities held out of training entirely had a median error of 33.3%, against 15.0% for seen cities.
  Unseen cities therefore get no estimate.

## Project structure

```text
app/streamlit_app.py          Streamlit web app "HomeValue AI" (Phase 5)
app/styles.css                App styling (local; inline SVG icons, no images or web fonts)
src/data_cleaning.py          Phase 2: raw CSV -> cleaned Parquet + quarantine + cleaning reports
src/warehouse_etl.py          Phase 3: Parquet -> PostgreSQL star schema, with 52 validation checks
src/model_common.py           Phase 4: shared features, location encoder, metrics
src/train_model.py            Phase 4: model comparison, location support, conformal intervals, artifacts
src/predict.py                Validated predictions used by the app (v2: optional ZIP, hybrid routing)
src/zip_experiment.py         ZIP-code experiment (validation-first, one test evaluation)
src/integrate_zip_model.py    Builds the v2 production artifacts from the approved ZIP candidate
sql/create_warehouse.sql      Warehouse DDL (staging, dw, audit schemas)
sql/load_warehouse.sql        Staging -> warehouse transformation
sql/example_analytical_query.sql  Four example analytical queries over the star schema (run against the live warehouse)
models/                       v2 artifacts (see 'Model version'); backups/v1_no_zip/ holds the verified v1 files
reports/                      Profiling, cleaning, warehouse, model and location reports (+ figures/);
                              final_report.md is the academic report, and figures/make_report_figures.py
                              regenerates its charts from the project's own files
notebooks/                    Exploratory analysis
tests/                        pytest suites for the model pipeline and the web app
data/raw/                     realtor-data.zip.csv.zip (downloaded source, never modified)
data/processed/               Cleaned outputs (regenerated by data_cleaning.py)
.streamlit/config.toml        Theme colours; hides error details from app users
```

## Installation

Python 3.10 was used.

```bash
pip install -r requirements.txt
```

## Running the web app

The app needs only the files in `models/`; the accuracy and coverage figures it quotes are read from
`models/model_metadata.json`. It does **not** connect to PostgreSQL and does not retrain anything. Run from the project root:

```bash
streamlit run app/streamlit_app.py
```

Then open the URL it prints (normally http://localhost:8501).

The interface is branded **HomeValue AI** ("Smarter Insights for Every Home"). Its look comes from the `[theme]`
section of `.streamlit/config.toml` and from `app/styles.css`; both are local, so the app renders offline, and it still
works (with the default look) if the stylesheet is missing. The page is deliberately minimal: a header, a **Property
Details** card, an **Estimated Home Value** card, and three collapsed panels (*About this estimate*, *Location coverage*,
*Model limitations*) that hold the longer explanations. The **Estimate Home Value** button is disabled while the shown
estimate matches the inputs, so the same request is not submitted twice.

### Inputs

| Input | Rule |
|---|---|
| State | Only states with at least one supported or limited-coverage city are listed |
| City | Filtered by state, loaded from `models/location_support.csv`; limited-coverage cities are labelled "(limited)" |
| ZIP code | Optional; only ZIPs recorded for the chosen city in the dataset are listed (5-character strings, leading zeros kept). ZIPs with fewer than 5 training records are marked "(few records)" and use the city-level estimate |
| Bedrooms | Whole number, 1-20 |
| Bathrooms | 1-20 (the training data contains whole numbers only, so fractions trigger a warning) |
| House size | 200-50,000 sq ft; unusual sizes trigger a warning |
| Lot size | Optional: tick *Include lot size* to enter it (more than 0 and at most 1,000 acres) |

Cities with too little reliable data are **never** given an estimate. The *Location coverage* panel explains the
policy and can look up any city's status. The policy is described in `reports/location_policy.md`.

### Reading the result

- **Estimated Home Value:** the model's central estimate (the recorded price), rounded to the nearest $1,000, with a one-line
  note that it is based on historical prices and is not an appraisal. The full disclosure appears once, in *About this estimate*.
- **Model used** (in *About this estimate*): "city-level" when no ZIP is chosen, "ZIP-level detail" when the ZIP has at least 5
  training records, or "city-level" with a notice beside the estimate when a chosen ZIP has too few records.
- **80% and 90% prediction ranges:** split-conformal ranges calibrated on held-out homes. On the test split, 80.1% / 90.2%
  of homes fell inside their 80% / 90% range without a ZIP, and 80.4% / 90.2% with the home's ZIP. That is an average over
  many homes, **not a guarantee for any one property**. Coverage of the 80% range was weaker for homes whose recorded price
  turned out under $100,000 (55% without ZIP, 57% with) or over $2 million (66% / 70%) and in limited-coverage cities
  (79%); the app warns about price extremes and limited coverage beside the estimate. Displayed range ends are rounded outward to whole thousands.

## Reproducing the pipeline

```bash
python src/data_cleaning.py                    # Phase 2: about 2 minutes; writes data/processed and reports/cleaning_*
cp .env.example .env                           # Phase 3: then fill in PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
python src/warehouse_etl.py all --create-db    # create + load + validate the warehouse (needs CREATEDB)
python src/train_model.py                      # Phase 4: about 2 minutes; models/ and reports/model_comparison.md
python src/train_model.py --no-test            # development run that never touches the test split
python src/zip_experiment.py                   # ZIP experiment, stage 1 (validation only)
python src/zip_experiment.py --test            # stage 2: one test evaluation of the stage-1 choice (refuses a second run)
python src/integrate_zip_model.py              # v2: backup v1, calibrate ZIP-path ranges, rebuild location support
```

`.env` is git-ignored; never commit real credentials. `train_model.py` reads the warehouse and cross-checks it against
the Parquet file, and falls back to the Parquet file if the database is unreachable.

### Reproducibility requirements

- **Software:** Python 3.10.10 with pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.7.2 and joblib 1.4.2. These are recorded in
  `models/model_metadata.json`; the saved models should be loaded with the same scikit-learn version. PostgreSQL 17 is
  needed for the warehouse only. Streamlit 1.45 was used for the app.
- **Data:** the raw file must match the SHA-256 in `reports/cleaning_summary.json`.
- **Determinism:** the seed is 42 throughout, and splits come from a deterministic hash of `property_group`. Reruns
  reproduce the cleaned files and the v2 artifacts byte for byte (checked for `src/data_cleaning.py` and
  `src/integrate_zip_model.py`).
- **Not in version control:** large data files (`data/raw`, `data/processed`) are git-ignored and must be supplied or
  regenerated. `.env` is created per machine from `.env.example`.

## Tests

```bash
python -m pytest tests -q
```

- `tests/test_model_pipeline.py`: leakage controls, inference matching training, and input validation (37 tests).
- `tests/test_streamlit_app.py`: app startup, location choices, policy enforcement, both ranges shown, invalid input,
  failure handling without leaking internal details, and the redesigned UI (34 tests, using `streamlit.testing.v1.AppTest`).
- `tests/test_zip_experiment.py`: ZIP normalisation, leading zeros, encoder fallback and cross-fitting, split integrity,
  v1 preservation (34 tests).
- `tests/test_zip_integration.py`: v1 backup and rollback, no-ZIP results identical to v1, ZIP-city consistency,
  fallback routing, calibration provenance, location policy v2, app ZIP behaviour and safe errors (33 tests).

**Real-browser check of the final HomeValue AI interface (2026-10-28).** Playwright drove Chromium at 1366 px and at a
390 px mobile width against the running app. All **100 checks passed**: 89 for the desktop and mobile functional and
layout flows, and 11 for the model-load failure page, which was served from a temporary copy with a deliberately broken
model file. Results are in `reports/figures/app_screenshots/browser_check_results.json`, tagged with the SHA-256 of the
app files that were tested, and next to the screenshots. To repeat the check (the script is `tests/browser_check.py`,
and is not run by pytest):

```bash
pip install playwright && python -m playwright install chromium
streamlit run app/streamlit_app.py --server.port 8601 --server.headless true     # in a second terminal
python tests/browser_check.py http://127.0.0.1:8601 reports/figures/app_screenshots full
```

An earlier run, on the pre-restyle app, found two display defects that the in-process tests cannot see, and both were
fixed:
- Price ranges were rendered as LaTeX math, because Streamlit reads `$...$` as math.
- Reset left the previous state visible in the browser.

Known limits of this check:
- Chromium only. No Firefox, Safari or real phone was used. The mobile view is Chromium emulation at 390 px.
- Colour contrast was not measured.
- Invalid and wrong-city ZIP codes are blocked by the ZIP dropdown, which lists only the chosen city's ZIP codes. The
  browser check confirms they cannot be selected. Server-side rejection of invalid or mismatched ZIPs is covered by the
  automated tests, not by the browser.
- On a phone-width screen the two range rows stack their label above the value; the three expanders open full width.

## Model version

The installed model is **2.0.0-zip-hybrid**. Without a ZIP it is exactly the v1 model with its v1 ranges. With a ZIP it
uses a ZIP-level model when the ZIP has at least 5 training records, and the v1 model otherwise; the ZIP-path ranges were
recalibrated on the calibration split. Details and decisions: `reports/zip_feature_experiment.md`.

## Results summary (held-out test split, 139,100 homes)

| | Without ZIP (v1 path) | With the home's ZIP (v2 hybrid) |
|---|---|---|
| Model | HistGradientBoostingRegressor on log price | ZIP-level model, v1 fallback for rare ZIPs |
| MAE / RMSE | $106,982 / $383,269 | $98,281 / $352,288 |
| R2 | 0.699 | 0.745 |
| Median absolute error | $51,125 (14.9%) | $47,738 (13.8%) |
| 80% / 90% range coverage | 80.1% / 90.2% | 80.4% / 90.2% |

The test split informed none of the choices above; it has now been evaluated for reporting in Phase 4, the ZIP experiment
and the integration check. Details: `reports/model_comparison.md`, `reports/zip_feature_experiment.md`. Warehouse design: `reports/warehouse_schema.md` and `reports/warehouse_etl.md`.

## Limitations

- **Historical, unverified target.** The data are recorded prices from late 2021 to mid-2022, not current values or
  verified sale prices.
- **Location coverage.** 770 cities are supported and 1,665 have limited coverage (v2 policy, validation evidence). Many cities, and some whole states, have
  too few complete records, and cities never seen in training were estimated far less accurately (about 33% median
  error against 15%), so they are refused.
- **Uncertainty.** The model knows only location, bedrooms, bathrooms, size and lot. Condition, age, exact address and
  neighbourhood are unknown, so individual errors can be large.
- **ZIP codes.** Only ZIPs recorded with the chosen city in this dataset are accepted, and the source's ZIP-city pairs are
  not independently verified. A wrong but listed ZIP produces a confidently wrong ZIP-level estimate.
- **Not an appraisal.** Do not use the output for financial or legal decisions.

## Rollback to v1 (no ZIP)

1. Copy the four files in `models/backups/v1_no_zip/` back into `models/` and check them against `MANIFEST.json`.
2. Move `models/zip_model.joblib` and `models/zip_city_map.csv` out of `models/`. The app then hides the ZIP field and
   behaves exactly like v1.
3. Run `python -m pytest tests -q`. The v2 integration tests skip themselves once `zip_model.joblib` is absent.
   Their skip message, "run src/integrate_zip_model.py first", is expected after a rollback.

**Verified on 2026-10-28 in an isolated copy** of the project. The live `models/` folder was not changed.
- The restored files matched `MANIFEST.json`, and the tests gave 99 passed and 33 skipped.
- v1's test MAE ($106,982.27) and R2 (0.698577) were reproduced exactly, and Houston (3 bed / 2 bath / 1,800 sq ft)
  gave $327,418 [$215,781 - $496,811].
- On 200 cities at both range levels, estimates and ranges were identical to the live no-ZIP path.
- A real browser confirmed the ZIP field is hidden and the v1 values are shown.
