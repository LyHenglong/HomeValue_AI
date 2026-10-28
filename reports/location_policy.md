# Supported-location policy

> **Current version: v2 (model 2.0.0-zip-hybrid).** `models/location_support.csv` is now written by
> `python src/integrate_zip_model.py`. The thresholds below are unchanged, but two things differ from the v1 text
> that follows:
> - **Evidence:** held-out evidence is the **validation split only**.
> - **Error used:** the **worse** of the no-ZIP and with-ZIP paths, so a city is supported only if both are reliable.
>
> v2 counts are 770 supported cities, 1,665 limited and 13,655 unsupported. 32 cities moved from supported to limited and
> 10 from limited to supported. Of the 32 downgrades, 25 are because validation error exceeded 25%. That is the stricter
> worse-of-both-paths rule plus noisier validation-only estimates. The other 7 are because the city has fewer than 10
> validation rows.
>
> A ZIP code, when given, must be recorded for the chosen city, and an unsupported city is refused with or without a ZIP.
> See "Integration into production" in `reports/zip_feature_experiment.md`. The v1 policy file is preserved in
> `models/backups/v1_no_zip/`.

*The rest of this document describes the original Phase 4 (v1) policy.*

Source of truth for v1: `models/location_support.csv` (written by `python src/train_model.py`). The web app builds its
dropdowns from this file and never from a hardcoded city list. Per-city detail, including test-split confirmation
columns, is in `reports/location_evaluation.csv`.

## Rules

A location is a `(state, city)` pair. Status is decided from the **training-row count** and the **held-out error** on the
validation and calibration splits only (never the test split):

| Status | Rule | App behaviour |
|---|---|---|
| **supported** | at least 100 training rows, at least 10 held-out rows, held-out median absolute % error <= 25% | estimate and range |
| **limited** | at least 30 training rows but not supported (fewer than 100 rows, too few held-out rows, or error above 25%) | estimate and range **with a visible warning** that includes the stored reason |
| **unsupported** | fewer than 30 training rows | **no estimate**, with the reason |
| not in the table | the city never appeared in the cleaned data | **no estimate** |

No validated fallback exists for unsupported or unseen cities: when whole cities were held out of training, the median
error was 33.3% (against 15.0% for seen cities) and R2 was 0.44 (against 0.67), so a state-level fallback is not offered.

## Why not sample count alone

Candidate thresholds of 30, 100 and 500 training rows were examined (`reports/model_comparison.md`). Held-out error did
not fall steadily with more rows: cities with at least 500 training rows had a *higher* median error (16.4%) than those
with at least 100 (14.1%), because the largest cities contain very mixed neighbourhoods. The 100-row floor therefore
combines with an explicit error ceiling. Example: Miami, Florida has plenty of data but a held-out error of 27%, so it is
`limited`, not `supported`.

## Resulting coverage (Oct 2021 - May 2022 sold-status records)

| Status | Cities | Cleaned rows |
|---|---:|---:|
| supported | 792 | 401,851 |
| limited | 1,643 | 179,078 |
| unsupported | 13,655 | 115,358 |

Test-split confirmation (rows not used to set the policy): median absolute % error 13.5% for supported cities, 15.8% for
limited and 19.3% for unsupported, so the status ordering holds on unseen rows, though the gaps are modest.

Thirteen states or territories have **no** supported city (Connecticut, Hawaii, Indiana, Nebraska, New Hampshire, New Jersey,
North Dakota, Puerto Rico, South Dakota, Utah, Vermont, Virgin Islands, Wyoming), and several large cities are unsupported
simply because the dataset holds very few complete sold records for them (for example Columbus, Ohio has 3).
The app must say so plainly rather than offering an estimate.

## Input validation (enforced in `src/predict.py`)

- Bedrooms: whole number 1-20. Bathrooms: 1-20 (fractions allowed with a warning). House size: 200-50,000 sq ft.
  Lot size: optional, above 0 and at most 1,000 acres.
- House size outside the training 0.5-99.5 percentile range, or a very large lot, returns a warning.
- Missing bedrooms, bathrooms or house size are **rejected**: imputing them gave 25-33% median error on the held-out incomplete
  rows (against 15% for complete rows) and did not improve complete-case accuracy.
- Booleans, strings, NaN and infinite values are rejected.
