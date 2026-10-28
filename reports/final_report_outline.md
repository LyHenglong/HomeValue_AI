# Final report: status, evidence and remaining work

Last updated 2026-10-28. The academic report is `reports/final_report.md` (about 14,400 words, 12 figures, 23 numbered
tables, references and appendices A to F). It is a **draft for the student to review**: names, dates and the
academic-integrity statement are placeholders, and a few facts about the dataset could not be verified (below). Numbers in
the report come from the project's own result files and were checked by a script (`reports/verify_final_report.py`).

## 1. Section status

| Report section | Status | Notes |
|---|---|---|
| Title page | **Placeholders** | Student name and ID, instructor, institution, submission date |
| Abstract | Written | |
| 1. Introduction | Written | Research questions were formulated while writing; the report says they were not pre-registered |
| 2. Dataset and data preparation | Written | Owner, version, licence field and educational-use terms **verified via Kaggle's public metadata** (as returned by a fetch tool; recheck in a browser). Data dictionary still unverified |
| 3. Data warehouse and ETL | Written | Schema diagram as a PNG (Figure 2) plus the Mermaid source; DDL in Appendix A |
| 4. Exploratory data analysis | Written | Figures regenerate from project files; the notebook has no stored outputs |
| 5. ML methodology | Written | |
| 6. Results and evaluation | Written | Discloses repeated use of the test split (Section 6.8) |
| 7. Application | Written | Final HomeValue AI screenshots (Figures 10 to 12) |
| 8. Testing and validation | Written | Chromium-only; no contrast measurement |
| 9. Limitations, ethics, risks | Written | |
| 10. Conclusion and future work | Written | Separates completed work from proposed work |
| References | Written | 14 entries, each checked against a web source; some without DOI (see Section 3 below) |
| Appendices A to E | Written | A schema DDL, B four analytical SQL queries with live results, C reproducibility, D further tables, E evidence map |
| Appendix F | **Placeholder** | Academic-integrity / AI-use statement; your course's wording is needed |

## 2. What you still need to do

1. Fill the bracketed placeholders (title page, Appendix F) and delete the draft box at the top of the report.
2. Write the AI-use statement your course requires. This project was built with an AI coding assistant, and the report cannot
   make that declaration for you.
3. Open the dataset's Kaggle page in a browser and confirm the licence wording quoted in Section 2.1 and the README (the page could
   not be rendered by the tools used). Note which dataset version you downloaded, and confirm that `realtor-data.zip.csv` is offered
   there. Decide whether publishing derived statistics and models fits the "educational purposes only" terms, and keep the
   repository private if in doubt.
4. Read the whole report and make sure you can explain every section. Adapt its length and format to your course template,
   and convert to PDF or Word if required. The Mermaid block renders only in some viewers; the PNG diagram works everywhere.
5. Optional: run `notebooks/01_data_exploration.ipynb` so the notebook stores its outputs. The report does not depend on them.

## 3. Facts that could not be verified (the report says so wherever they appear)

- **Dataset provenance, partly verified.** From Kaggle's public metadata endpoint (retrieved 2026-10-28 with a fetch tool): owner
  Ahmed Shahriar Sakib, version 25, last updated 2024-03-30, licence "Other (specified in description)", the description's
  "educational purposes only" and "I do not own any data, and all rights are reserved to the respective owners" statements, and
  Realtor.com (Move, Inc.) named as the source. **Not verified:** the data dictionary, the update schedule, the exact file list, which
  version was downloaded, and the wording itself in a rendered page.
- **Meaning of `price` and `prev_sold_date`** for sold-status records.
- **ZIP-to-city and state/ZIP mappings:** derived from the dataset itself, not an official source.
- **References:** DOIs were not verified for Hoerl & Kennard (1970) and Kaufman et al. (2012), so none is given. McKinney (2010)
  is cited by its proceedings URL, with no page range, because neither the DOI nor the end page could be confirmed.
- **Rollback browser run:** an earlier 12-check browser run of the rolled-back app (previous interface) was not saved and is not relied on.
- **Real-device and cross-browser behaviour, and colour contrast:** not tested.

## 4. Evidence summary (actual results, 2026-10-28)

- **Automated tests:** `python -m pytest tests -q -p no:warnings` gave **138 passed, 0 failed, 0 skipped**. By file: model
  pipeline 37, application 34, ZIP experiment 34, ZIP integration 33.
- **Real browser (Chromium via Playwright), final interface:** **89 of 89** functional and layout checks (56 desktop at 1366 px,
  33 mobile at 390 px) and **11 of 11** model-load failure checks, in
  `reports/figures/app_screenshots/browser_check_results.json`, tagged with the SHA-256 of the app files that were tested.
- **Model integrity:** the 11 production and backup model files have the same SHA-256 hashes as before the interface restyle and
  before the report work.
- **Rollback (isolated copy, live `models/` untouched):** **105 passed, 33 skipped** (the 33 v2 integration tests skip
  themselves). The restored v1 reproduces the recorded test MAE ($106,982.27) and R2 (0.698577) exactly. All 400 requests
  (200 cities at two range levels) give estimates and ranges identical to the live no-ZIP path; 6 differ only in status label. Rollback
  was re-verified against the final interface.
- **Analytical SQL:** the four queries in `sql/example_analytical_query.sql` ran on the live PostgreSQL 17.2 warehouse, and each
  result matched an independent pandas calculation on the cleaned Parquet file. Results are saved in
  `reports/example_query_results/`.
- **Report numbers:** 48 claim checks against the source files (headline metrics, cleaning counts, splits, coverage, the
  ZIP experiment, warehouse checks and test counts), plus checks for broken tables, figure links, placeholders, secrets and
  machine-specific paths, all passed.

## 5. Where the evidence lives

| Report content | Source in the project |
|---|---|
| Raw profile, cleaning | `reports/phase1_*.csv`, `reports/cleaning_*.csv`, `reports/cleaning_summary.json` |
| Warehouse | `sql/create_warehouse.sql`, `sql/load_warehouse.sql`, `reports/warehouse_schema.md`, `reports/warehouse_etl.md`, `reports/warehouse_validation.json` |
| Models, intervals, geography | `reports/experiment_summary.json`, `reports/model_comparison.md` |
| ZIP experiment and integration | `reports/zip_experiment_validation.json`, `reports/zip_integration.json`, `reports/zip_feature_experiment.md`, `reports/location_policy.md`, `models/model_metadata.json` |
| Application and tests | `app/`, `tests/`, `reports/figures/app_screenshots/` |
| Figures 1 to 9 | `reports/figures/make_report_figures.py` (run from the project root) |
| Screenshots (Figures 10 to 12 and the rest) | `reports/figures/app_screenshots/`, captured with `tests/browser_check.py`: desktop `desktop_01` to `desktop_06` and `desktop_error_model_load_failure`; mobile `mobile_01` to `mobile_05_expanders_open` and `mobile_error_model_load_failure` |
