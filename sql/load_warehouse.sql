-- Phase 3: staging -> warehouse transformation.
-- Prerequisite: staging.sold_clean and staging.quarantine hold the current Phase 2 snapshot, and the
-- session setting etl.run_id names the audit.etl_run row for this load:
--     SELECT set_config('etl.run_id', '<run_id>', true);
-- src/warehouse_etl.py runs this whole file in one transaction together with the staging COPY, so a
-- failure anywhere rolls back every change. Rerunning with an unchanged snapshot changes nothing
-- (fact rows are upserted only when a value differs; the report shows 0 inserted / 0 updated / 0 deleted).
-- Semantics: the warehouse mirrors the latest snapshot, so fact rows that are no longer present in
-- staging are deleted, and dimension rows no fact row references are removed.

-- 0. Guards ----------------------------------------------------------------------------------------
DO $$
BEGIN
    IF current_setting('etl.run_id', true) IS NULL OR current_setting('etl.run_id', true) = '' THEN
        RAISE EXCEPTION 'etl.run_id is not set';
    END IF;
    IF (SELECT count(*) FROM staging.sold_clean) = 0 THEN
        RAISE EXCEPTION 'staging.sold_clean is empty; refusing to sync the warehouse to an empty snapshot';
    END IF;
    IF EXISTS (SELECT 1 FROM staging.sold_clean WHERE raw_row_id IS NULL
                OR status IS DISTINCT FROM 'sold') THEN
        RAISE EXCEPTION 'staging.sold_clean contains rows with NULL raw_row_id or status other than sold';
    END IF;
    IF (SELECT count(*) FROM staging.sold_clean) <> (SELECT count(DISTINCT raw_row_id) FROM staging.sold_clean) THEN
        RAISE EXCEPTION 'staging.sold_clean has duplicate raw_row_id values';
    END IF;
END $$;

-- 1. dim_date: every calendar day between the earliest and latest recorded prev_sold_date -----------
INSERT INTO dw.dim_date (date_key, full_date, year, quarter, month, month_name, day,
                         iso_week, day_of_week, day_name, is_weekend)
SELECT to_char(d, 'YYYYMMDD')::int, d::date,
       extract(year FROM d)::smallint, extract(quarter FROM d)::smallint,
       extract(month FROM d)::smallint, trim(to_char(d, 'Month')),
       extract(day FROM d)::smallint, extract(week FROM d)::smallint,
       extract(isodow FROM d)::smallint, trim(to_char(d, 'Day')),
       extract(isodow FROM d) IN (6, 7)
FROM (SELECT generate_series(min(prev_sold_date)::timestamp, max(prev_sold_date)::timestamp, interval '1 day') AS d
      FROM staging.sold_clean) g
WHERE d IS NOT NULL
ON CONFLICT (date_key) DO NOTHING;

-- 2. dim_location ----------------------------------------------------------------------------------
INSERT INTO dw.dim_location (state, city, zip_code, is_us_territory)
SELECT DISTINCT state, city, zip_code, state IN ('Puerto Rico', 'Guam', 'Virgin Islands')
FROM staging.sold_clean
ON CONFLICT ON CONSTRAINT uq_dim_location DO NOTHING;

-- 3. dim_property: bands are defined here and documented in reports/warehouse_schema.md ------------
CREATE TEMP TABLE tmp_property AS
SELECT bed::smallint AS bedrooms,
       bath::numeric(4,1) AS bathrooms,
       CASE WHEN house_size <  1000 THEN '1) under 1,000 sqft'
            WHEN house_size <  1500 THEN '2) 1,000-1,499 sqft'
            WHEN house_size <  2000 THEN '3) 1,500-1,999 sqft'
            WHEN house_size <  3000 THEN '4) 2,000-2,999 sqft'
            WHEN house_size <  4000 THEN '5) 3,000-3,999 sqft'
            ELSE                         '6) 4,000+ sqft' END AS house_size_band,
       CASE WHEN acre_lot IS NULL THEN '0) unknown'
            WHEN acre_lot <  0.25 THEN '1) under 0.25 acre'
            WHEN acre_lot <  0.5  THEN '2) 0.25-0.49 acre'
            WHEN acre_lot <  1    THEN '3) 0.5-0.99 acre'
            WHEN acre_lot <  5    THEN '4) 1-4.99 acres'
            WHEN acre_lot <  20   THEN '5) 5-19.99 acres'
            ELSE                       '6) 20+ acres' END AS lot_size_band,
       raw_row_id
FROM staging.sold_clean;

INSERT INTO dw.dim_property (bedrooms, bathrooms, house_size_band, lot_size_band)
SELECT DISTINCT bedrooms, bathrooms, house_size_band, lot_size_band FROM tmp_property
ON CONFLICT ON CONSTRAINT uq_dim_property DO NOTHING;

-- 4. fact upsert: insert new rows, update only rows whose values changed ---------------------------
WITH up AS (
    INSERT INTO dw.fact_property_listing AS f
        (raw_row_id, location_key, property_dim_key, prev_sold_date_key, price, house_size_sqft,
         acre_lot, price_per_sqft, acre_lot_was_zero, repeated_property, property_key, property_group,
         brokered_by_code, street_code, source_status, etl_run_id)
    SELECT s.raw_row_id, l.location_key, p.property_dim_key,
           CASE WHEN s.prev_sold_date IS NULL THEN NULL ELSE to_char(s.prev_sold_date, 'YYYYMMDD')::int END,
           s.price::numeric(14,2), s.house_size::int, s.acre_lot::numeric(10,4), s.price_per_sqft::numeric(12,2),
           s.acre_lot_was_zero, s.repeated_property, s.property_key, s.property_group,
           s.brokered_by::bigint, s.street::bigint, s.status,
           current_setting('etl.run_id')::bigint
    FROM staging.sold_clean s
    JOIN tmp_property t USING (raw_row_id)
    JOIN dw.dim_location l ON l.state = s.state AND l.city = s.city AND l.zip_code = s.zip_code
    JOIN dw.dim_property p ON p.bedrooms = t.bedrooms AND p.bathrooms = t.bathrooms
                          AND p.house_size_band = t.house_size_band AND p.lot_size_band = t.lot_size_band
    ON CONFLICT (raw_row_id) DO UPDATE SET
        location_key       = EXCLUDED.location_key,
        property_dim_key   = EXCLUDED.property_dim_key,
        prev_sold_date_key = EXCLUDED.prev_sold_date_key,
        price              = EXCLUDED.price,
        house_size_sqft    = EXCLUDED.house_size_sqft,
        acre_lot           = EXCLUDED.acre_lot,
        price_per_sqft     = EXCLUDED.price_per_sqft,
        acre_lot_was_zero  = EXCLUDED.acre_lot_was_zero,
        repeated_property  = EXCLUDED.repeated_property,
        property_key       = EXCLUDED.property_key,
        property_group     = EXCLUDED.property_group,
        brokered_by_code   = EXCLUDED.brokered_by_code,
        street_code        = EXCLUDED.street_code,
        source_status      = EXCLUDED.source_status,
        etl_run_id         = EXCLUDED.etl_run_id
    WHERE (f.location_key, f.property_dim_key, f.prev_sold_date_key, f.price, f.house_size_sqft, f.acre_lot,
           f.price_per_sqft, f.acre_lot_was_zero, f.repeated_property, f.property_key, f.property_group,
           f.brokered_by_code, f.street_code, f.source_status)
          IS DISTINCT FROM
          (EXCLUDED.location_key, EXCLUDED.property_dim_key, EXCLUDED.prev_sold_date_key, EXCLUDED.price,
           EXCLUDED.house_size_sqft, EXCLUDED.acre_lot, EXCLUDED.price_per_sqft, EXCLUDED.acre_lot_was_zero,
           EXCLUDED.repeated_property, EXCLUDED.property_key, EXCLUDED.property_group,
           EXCLUDED.brokered_by_code, EXCLUDED.street_code, EXCLUDED.source_status)
    RETURNING (xmax = 0) AS inserted
)
UPDATE audit.etl_run
SET fact_rows_inserted = (SELECT count(*) FILTER (WHERE inserted) FROM up),
    fact_rows_updated  = (SELECT count(*) FILTER (WHERE NOT inserted) FROM up)
WHERE run_id = current_setting('etl.run_id')::bigint;

-- 5. remove fact rows that are no longer in the snapshot -------------------------------------------
WITH del AS (
    DELETE FROM dw.fact_property_listing f
    WHERE NOT EXISTS (SELECT 1 FROM staging.sold_clean s WHERE s.raw_row_id = f.raw_row_id)
    RETURNING 1
)
UPDATE audit.etl_run
SET fact_rows_deleted = (SELECT count(*) FROM del)
WHERE run_id = current_setting('etl.run_id')::bigint;

-- 6. remove dimension rows nothing references any more (dim_date is a calendar and is kept) --------
DELETE FROM dw.dim_location l
WHERE NOT EXISTS (SELECT 1 FROM dw.fact_property_listing f WHERE f.location_key = l.location_key);
DELETE FROM dw.dim_property p
WHERE NOT EXISTS (SELECT 1 FROM dw.fact_property_listing f WHERE f.property_dim_key = p.property_dim_key);

-- 7. quarantine audit table: replaced by the current snapshot ---------------------------------------
DELETE FROM audit.quarantine_record;
INSERT INTO audit.quarantine_record
    (raw_row_id, etl_run_id, quarantine_reason, all_flags, status, price, bed, bath, house_size,
     acre_lot, city, state, zip_code, prev_sold_date, property_key)
SELECT raw_row_id, current_setting('etl.run_id')::bigint, quarantine_reason, all_flags, status, price, bed,
       bath, house_size, acre_lot, city, state, zip_code, prev_sold_date, property_key
FROM staging.quarantine;

-- 8. audit counts -------------------------------------------------------------------------------------
UPDATE audit.etl_run
SET staged_clean_rows      = (SELECT count(*) FROM staging.sold_clean),
    staged_quarantine_rows = (SELECT count(*) FROM staging.quarantine)
WHERE run_id = current_setting('etl.run_id')::bigint;

DROP TABLE tmp_property;
