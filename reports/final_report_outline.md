# Final report outline: House Price Recommendation System

This is an outline for the academic write-up. Each section lists the evidence that already exists in this repository;
use those figures directly rather than restating them from memory. A section marked **To write** has no prose yet.
The technical reports listed are complete; summarise them, don't duplicate them.

## 1. Introduction and objective

- **Points:**
  - Goal: estimate the *recorded price* of U.S. homes with honest uncertainty ranges.
  - Course aim: show warehousing and data-mining methods end to end.
  - Scope: sold-status records with complete core features.
  - Out of scope: market valuation and appraisal.
- **Evidence:** `README.md` (Objective and scope).
- **To write:** motivation and research questions.

## 2. Dataset and target definition

- **Points:**
  - Source: Realtor.com USA Real Estate dataset, 2,226,382 rows and 12 columns.
  - Why the target is "recorded price of sold-status records", not a verified sale price.
  - The date window is 2021-10-18 to 2022-05-06.
- **Evidence:**
  - `reports/phase1_schema_missingness.csv`, `phase1_status_summary.csv` and `phase1_price_distribution.png`
  - `notebooks/01_data_exploration.ipynb`
  - `reports/cleaning_summary.json`

## 3. Data cleaning and quality

- **Points:**
  - The rule-by-rule waterfall from 2,226,382 rows to 696,287 clean rows.
  - Quarantine instead of deletion (704 rows).
  - Missing-value strategy (no target imputation).
  - Leading-zero ZIP restoration.
  - Known quality limits.
- **Evidence:**
  - `reports/cleaning_report.md`, `cleaning_waterfall.csv`, `cleaning_exclusions_by_reason.csv`, `cleaning_missingness.csv`
  - `reports/cleaning_quarantine_samples.csv`

## 4. Data warehouse design and ETL

- **Points:**
  - Star schema: fact grain, three dimensions, staging and audit layers.
  - Transactional, rerunnable load.
  - 52 validation checks passed on the local PostgreSQL 17.2 server.
- **Evidence:**
  - `reports/warehouse_schema.md` (includes an ER diagram in Mermaid)
  - `reports/warehouse_etl.md`
  - `reports/warehouse_validation.md` / `.json`
  - `sql/create_warehouse.sql`, `sql/load_warehouse.sql`
- **To write:**
  - One example analytical SQL query over the star schema, if the course expects OLAP-style queries.
  - Its output. This has not been produced yet.

## 5. Methodology

- **Points:**
  - Property-grouped hash split (60/10/10/20) and what each split was used for.
  - Leakage safeguards.
  - Size-adjusted hierarchical location encoding, including the Florida bias that motivated it.
  - Log target.
  - Candidate models.
  - Split-conformal prediction ranges.
  - Location-support policy.
- **Evidence:**
  - `reports/model_comparison.md` (design, leakage controls and development note)
  - `reports/experiment_summary.json` (configuration and seed)
  - `src/model_common.py`
  - `reports/location_policy.md`

## 6. Results

- **Points:**
  - The model comparison table (MAE, RMSE, R2, median % error).
  - Performance by price band, state and city size.
  - Geographic hold-out (33.3% median error on unseen cities against 15.0% on seen).
  - Missing-data experiment.
  - Interval coverage (80.1% / 90.2%) overall and by group.
- **Evidence:**
  - `reports/model_comparison.md`
  - `reports/figures/actual_vs_predicted_and_residuals.png`, `reports/figures/interval_coverage_and_geographic_holdout.png`
  - `reports/location_evaluation.csv`

## 7. Model improvement: ZIP code (v2)

- **Points:**
  - The pre-registered decision rule.
  - Why the plain ZIP model failed on unseen ZIPs.
  - The hybrid design.
  - Test results for the hybrid: MAE $98,281, R2 0.745, median error 13.8%, coverage 80.4% / 90.2%.
  - Integration decisions and rollback.
- **Evidence:**
  - `reports/zip_feature_experiment.md` (experiment and integration)
  - `reports/zip_experiment_validation.json`, `reports/zip_integration.json`

## 8. Web application

- **Points:**
  - Inputs and validation.
  - Location and ZIP policy in the UI.
  - How results and uncertainty are presented.
  - Safe error handling.
  - Test coverage.
- **Evidence:**
  - `app/streamlit_app.py`
  - `tests/test_streamlit_app.py`, `tests/test_zip_integration.py`
  - `reports/figures/app_screenshots/` (browser check results and screenshots)
- **To do:** retake the screenshots after the final UI restyle. The current ones show the pre-restyle look.

## 9. Testing and validation

- **Points:**
  - The automated suite (count and what it covers).
  - The real-browser check and the two defects it found.
  - The isolated rollback verification.
- **Evidence:**
  - `README.md` (Tests; Rollback)
  - `reports/figures/app_screenshots/browser_check_results.json`

## 10. Limitations and ethics

- **Points:**
  - Historical, unverified target.
  - Uneven geographic coverage.
  - Weaker range coverage for very cheap and very expensive homes.
  - ZIP-city pairs are not verified.
  - The test split has been viewed several times for reporting, though it was never used for a decision.
  - Not an appraisal.
- **Evidence:** the `README.md` limitations, `reports/zip_feature_experiment.md` (limitations sections) and
  `reports/model_comparison.md`.

## 11. Conclusion and future work

- **To write:**
  - A summary.
  - Next steps: a fresh hold-out set or newer data, property detail features, and a verified ZIP-city reference.

## Appendix

- Reproduction commands and requirements (`README.md`).
- Environment versions (`models/model_metadata.json`).
