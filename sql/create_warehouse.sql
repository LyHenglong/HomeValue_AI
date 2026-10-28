-- Phase 3: warehouse DDL for the House Price Recommendation project.
-- Idempotent: safe to run repeatedly (CREATE ... IF NOT EXISTS). It never drops data.
-- Run with:  python src/warehouse_etl.py init     (or psql -f sql/create_warehouse.sql)
--
-- Layers
--   staging.*  : verbatim copies of the Phase 2 Parquet outputs, replaced on every load
--   dw.*       : star schema (one fact table, three dimensions)
--   audit.*    : ETL run log and the quarantined records kept for traceability

CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS dw;
CREATE SCHEMA IF NOT EXISTS audit;

-- ---------------------------------------------------------------------------------------------
-- audit.etl_run : one row per load attempt
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit.etl_run (
    run_id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    started_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at          TIMESTAMPTZ,
    status               TEXT NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'succeeded', 'failed')),
    raw_file             TEXT,
    raw_sha256           TEXT,            -- from reports/cleaning_summary.json (Phase 2)
    clean_parquet_sha256 TEXT,
    quarantine_parquet_sha256 TEXT,
    staged_clean_rows    BIGINT,
    staged_quarantine_rows BIGINT,
    fact_rows_inserted   BIGINT,
    fact_rows_updated    BIGINT,
    fact_rows_deleted    BIGINT,
    error_message        TEXT
);

-- ---------------------------------------------------------------------------------------------
-- staging.sold_clean : mirrors data/processed/sold_clean.parquet (no constraints on purpose)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS staging.sold_clean (
    raw_row_id        BIGINT,
    property_key      TEXT,
    property_group    TEXT,
    repeated_property BOOLEAN,
    status            TEXT,
    price             DOUBLE PRECISION,
    bed               INTEGER,
    bath              DOUBLE PRECISION,
    house_size        DOUBLE PRECISION,
    acre_lot          DOUBLE PRECISION,
    acre_lot_was_zero BOOLEAN,
    city              TEXT,
    state             TEXT,
    zip_code          TEXT,
    prev_sold_date    DATE,
    sold_year         SMALLINT,
    price_per_sqft    DOUBLE PRECISION,
    brokered_by       DOUBLE PRECISION,
    street            DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS ix_staging_sold_clean_raw_row_id ON staging.sold_clean (raw_row_id);

-- staging.quarantine : mirrors data/processed/quarantine.parquet
CREATE TABLE IF NOT EXISTS staging.quarantine (
    raw_row_id        BIGINT,
    brokered_by       DOUBLE PRECISION,
    status            TEXT,
    price             DOUBLE PRECISION,
    bed               DOUBLE PRECISION,
    bath              DOUBLE PRECISION,
    acre_lot          DOUBLE PRECISION,
    street            DOUBLE PRECISION,
    city              TEXT,
    state             TEXT,
    zip_code          TEXT,
    house_size        DOUBLE PRECISION,
    prev_sold_date    DATE,
    property_key      TEXT,
    acre_lot_was_zero BOOLEAN,
    quarantine_reason TEXT,
    all_flags         TEXT
);

-- ---------------------------------------------------------------------------------------------
-- audit.quarantine_record : typed, run-stamped copy of the quarantined rows
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit.quarantine_record (
    raw_row_id        BIGINT PRIMARY KEY,
    etl_run_id        BIGINT NOT NULL REFERENCES audit.etl_run (run_id),
    quarantine_reason TEXT NOT NULL,
    all_flags         TEXT,
    status            TEXT,
    price             DOUBLE PRECISION,
    bed               DOUBLE PRECISION,
    bath              DOUBLE PRECISION,
    house_size        DOUBLE PRECISION,
    acre_lot          DOUBLE PRECISION,
    city              TEXT,
    state             TEXT,
    zip_code          TEXT,
    prev_sold_date    DATE,
    property_key      TEXT
);

-- ---------------------------------------------------------------------------------------------
-- dw.dim_location : one row per distinct (state, city, zip_code)
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_location (
    location_key     INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    state            TEXT    NOT NULL,
    city             TEXT    NOT NULL,
    zip_code         CHAR(5) NOT NULL CHECK (zip_code ~ '^[0-9]{5}$'),
    is_us_territory  BOOLEAN NOT NULL,
    CONSTRAINT uq_dim_location UNIQUE (state, city, zip_code)
);
CREATE INDEX IF NOT EXISTS ix_dim_location_state_city ON dw.dim_location (state, city);

-- ---------------------------------------------------------------------------------------------
-- dw.dim_property : descriptive property attributes with low cardinality.
-- Exact house_size and acre_lot stay on the fact table as measures; this dimension holds
-- bedrooms, bathrooms and size bands so users can slice by property type.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_property (
    property_dim_key INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    bedrooms         SMALLINT     NOT NULL CHECK (bedrooms BETWEEN 1 AND 20),
    bathrooms        NUMERIC(4,1) NOT NULL CHECK (bathrooms >= 0.5 AND bathrooms <= 20),
    house_size_band  TEXT         NOT NULL,
    lot_size_band    TEXT         NOT NULL,
    CONSTRAINT uq_dim_property UNIQUE (bedrooms, bathrooms, house_size_band, lot_size_band)
);

-- ---------------------------------------------------------------------------------------------
-- dw.dim_date : calendar dimension. Its only use is the source column prev_sold_date, whose exact
-- meaning for 'sold' records is NOT verified; the fact table therefore uses the neutral name
-- prev_sold_date_key and nothing downstream may treat it as a confirmed closing date.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.dim_date (
    date_key     INTEGER  PRIMARY KEY,           -- yyyymmdd
    full_date    DATE     NOT NULL UNIQUE,
    year         SMALLINT NOT NULL,
    quarter      SMALLINT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    month        SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name   TEXT     NOT NULL,
    day          SMALLINT NOT NULL CHECK (day BETWEEN 1 AND 31),
    iso_week     SMALLINT NOT NULL,
    day_of_week  SMALLINT NOT NULL CHECK (day_of_week BETWEEN 1 AND 7),  -- ISO: 1 = Monday
    day_name     TEXT     NOT NULL,
    is_weekend   BOOLEAN  NOT NULL,
    CONSTRAINT ck_dim_date_key CHECK (date_key = to_char(full_date, 'YYYYMMDD')::int)
);

-- ---------------------------------------------------------------------------------------------
-- dw.fact_property_listing
-- Grain: one retained source record (one row of the raw CSV with status = 'sold' that passed
-- every Phase 2 rule). raw_row_id is the 0-based position of that row in the raw file.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dw.fact_property_listing (
    raw_row_id          BIGINT PRIMARY KEY,
    location_key        INTEGER NOT NULL REFERENCES dw.dim_location (location_key),
    property_dim_key    INTEGER NOT NULL REFERENCES dw.dim_property (property_dim_key),
    prev_sold_date_key  INTEGER          REFERENCES dw.dim_date (date_key),  -- NULL when the source date is missing
    price               NUMERIC(14,2) NOT NULL CHECK (price > 0),
    house_size_sqft     INTEGER       NOT NULL CHECK (house_size_sqft > 0),
    acre_lot            NUMERIC(10,4)          CHECK (acre_lot > 0),           -- NULL = unknown
    price_per_sqft      NUMERIC(12,2) NOT NULL CHECK (price_per_sqft > 0),
    acre_lot_was_zero   BOOLEAN       NOT NULL,
    repeated_property   BOOLEAN       NOT NULL,
    property_key        TEXT,                          -- '<street code>|<zip>' ; NULL if street unknown
    property_group      TEXT          NOT NULL,        -- leakage-safe grouping id for train/test splits
    brokered_by_code    BIGINT,                        -- anonymised source code, NULL if unknown
    street_code         BIGINT,                        -- anonymised source code, NULL if unknown
    source_status       TEXT          NOT NULL CHECK (source_status = 'sold'),
    etl_run_id          BIGINT        NOT NULL REFERENCES audit.etl_run (run_id)  -- run that last inserted or changed the row
);
CREATE INDEX IF NOT EXISTS ix_fact_location       ON dw.fact_property_listing (location_key);
CREATE INDEX IF NOT EXISTS ix_fact_property_dim   ON dw.fact_property_listing (property_dim_key);
CREATE INDEX IF NOT EXISTS ix_fact_prev_sold_date ON dw.fact_property_listing (prev_sold_date_key);
CREATE INDEX IF NOT EXISTS ix_fact_property_group ON dw.fact_property_listing (property_group);
CREATE INDEX IF NOT EXISTS ix_fact_etl_run        ON dw.fact_property_listing (etl_run_id);

COMMENT ON TABLE dw.fact_property_listing IS
  'One retained source record per row. price is the recorded price of a sold-status record; it is not a verified completed-sale price.';
COMMENT ON COLUMN dw.fact_property_listing.prev_sold_date_key IS
  'Source column prev_sold_date as recorded. Meaning for sold-status rows is unverified.';
COMMENT ON TABLE dw.dim_property IS
  'Bedrooms, bathrooms and size bands. Exact sizes live on the fact table.';
COMMENT ON TABLE audit.quarantine_record IS
  'Records held out of the warehouse by a Phase 2 rule; project-specific thresholds, kept for review.';
