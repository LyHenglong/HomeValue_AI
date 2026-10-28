# Warehouse ETL process (Phase 3)

## Local configuration

PostgreSQL must be running. Credentials are never stored in code; they come from environment variables, optionally via a
project-root `.env` file (git-ignored).

1. Copy `.env.example` to `.env`.
2. Set `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.
3. Install the dependencies: `pip install -r requirements.txt`.
4. Make sure the Phase 2 outputs exist (`python src/data_cleaning.py`).

```
python src/warehouse_etl.py all --create-db   # create database if missing, create tables, load, validate
python src/warehouse_etl.py init              # tables only (or: psql -f sql/create_warehouse.sql)
python src/warehouse_etl.py load              # refresh the warehouse from data/processed/
python src/warehouse_etl.py validate          # checks only; writes reports/warehouse_validation.{json,md}
```

`--create-db` needs a role with the CREATEDB privilege. Without it, create the database once (`CREATE DATABASE house_price_dw;`) and run
`init`. Docker is not required. The tested warehouse is about 300 MB, which is well inside the 2 GB Docker limit if you
prefer a container.

## Data flow

```
data/raw/realtor-data.zip.csv.zip   (never modified; SHA-256 recorded)
        |  src/data_cleaning.py (Phase 2)
        v
data/processed/sold_clean.parquet + quarantine.parquet
        |  COPY, 100,000-row batches (never the whole file in memory)
        v
staging.sold_clean, staging.quarantine          (TRUNCATE + reload each run)
        |  sql/load_warehouse.sql
        v
dw.dim_date, dim_location, dim_property -> dw.fact_property_listing      audit.quarantine_record
```

## Transformations (`sql/load_warehouse.sql`)

1. **Guards:** refuse to run if the run id is unset, staging is empty, a staged row is not `sold`, or `raw_row_id` repeats.
2. **`dim_date`:** every calendar day between the earliest and latest `prev_sold_date`; existing days are untouched.
3. **`dim_location`:** distinct `(state, city, zip_code)`; `is_us_territory` derived from state.
4. **`dim_property`:** distinct `(bed, bath, house-size band, lot-size band)`.
5. **Fact upsert:** keyed on `raw_row_id`. New rows are inserted. Existing rows are updated only if some value differs,
   so an unchanged rerun writes nothing. The type casts used are `price` to `NUMERIC(14,2)`, `house_size` to `INTEGER`,
   `acre_lot` to `NUMERIC(10,4)` and the source codes to `BIGINT`.
6. **Deletes:** fact rows missing from the new snapshot are removed, then dimension rows with no fact rows.
   `dim_date` is a calendar and is kept.
7. **Quarantine audit:** `audit.quarantine_record` is replaced with the current quarantined rows.
8. **Audit counts:** inserted / updated / deleted / staged row counts are written to `audit.etl_run`.

## Missing and invalid records

Handled before the warehouse, in Phase 2 (`reports/cleaning_report.md`). The warehouse accepts only complete-case sold
records: price, bedrooms, bathrooms, house size, state, city and zip are all present and plausible. Only these may be NULL:
`acre_lot` (87,989 rows, unknown lot), `prev_sold_date_key` (1 row), `property_key`, `brokered_by_code`, `street_code`.
CHECK, NOT NULL and FK constraints are a second line of defence, and the validator proves they reject bad data.

## Transactions and safe reruns

- The staging `COPY`, the transform and the quarantine refresh run in **one transaction**. Any error rolls back everything.
- The `audit.etl_run` row is written in a separate committed transaction, so a failed load is still recorded
  (`status = 'failed'`, with the error text).
- Reruns are idempotent: the warehouse always mirrors the latest snapshot.

## Refreshing with new data

1. Replace or add the raw file and rerun `python src/data_cleaning.py`.
2. Run `python src/warehouse_etl.py load`, then `validate`.

**Limitation:** `raw_row_id` is the row's *position* in the raw file. Re-running on the same file is exact. A new download
whose row order differs would assign different ids to the same listings, so the load would update or replace rows by
position rather than matching them by identity. The dataset has no true listing id, so for a different dataset version
treat the load as a fresh snapshot (drop and recreate the `dw` and `staging` schemas first) instead of an incremental
update.

## What was tested

**Real server (2026-10-27).** After the `postgres` role's password was reset, the warehouse was created and loaded on
the PostgreSQL 17.2 service at `localhost:5432` (database `house_price_dw`, which did not exist beforehand; the server
held only the `postgres` database). `python src/warehouse_etl.py all --create-db` passed **52/52** validation checks,
with these live counts: fact 696,287; `dim_location` 22,128; `dim_property` 1,939; `dim_date` 201;
`audit.quarantine_record` 704; database size 302 MB. A second `load` changed nothing (0 inserted, 0 updated,
0 deleted) and a second `validate` again passed 52/52. `reports/warehouse_validation.md` was written by that run.
The tampering, forced-failure and rollback tests below were run only on the temporary cluster, not on the real server.

The temporary-cluster tests below ran against a **disposable PostgreSQL 17.2 cluster created by me on port 5433**
(trust authentication, removed afterwards), before credentials for the real service were available.

| Test | Result |
|---|---|
| `init` + first `load` of 696,287 rows | succeeded in about 2 minutes; 0 updated, 0 deleted |
| `validate` | 52/52 checks passed (`reports/warehouse_validation.md`) |
| Second `load` with unchanged data | 0 inserted, 0 updated, 0 deleted |
| Tampered warehouse (one price changed, one row deleted, one stray row added), then `load` | repaired: 1 inserted, 1 updated, 1 deleted |
| `load` made to fail mid-way (a temporary CHECK constraint rejected a fact row) | rolled back; fact and staging row counts unchanged; run recorded as `failed` with the error |
| Constraint probes: price <= 0, unknown `location_key`, duplicate `raw_row_id`, non-sold status, malformed zip | all rejected |

The `validate` command checks:
- **Row counts:** parquet vs staging vs fact vs quarantine.
- **NOT NULL:** declared and respected.
- **Keys:** primary and natural-key uniqueness; all four foreign keys, with no orphans in either direction.
- **Consistency:** every loaded column compared row by row against staging; totals and per-state counts recomputed independently from the Parquet file in pandas.
- **Audit:** the run's source and Parquet hashes match the files on disk.

**Not tested:**
- A PostgreSQL version other than 17.2.
- A `--create-db` call by a role without the CREATEDB privilege.
- Concurrent loads, and loads against a non-empty warehouse built from a different dataset.

Validation notes: one check was initially written too strictly and was corrected. Phase 2 rounded `price_per_sqft` with
Python (half-to-even) and PostgreSQL rounds exact halves up, so 1,362 rows differ from a PostgreSQL recomputation by
exactly $0.01. The loaded value equals the Phase 2 value in all rows. The check now allows half a cent.
