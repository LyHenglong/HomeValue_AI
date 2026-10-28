-- Example analytical queries for the HomeValue AI star schema (PostgreSQL 17).
--
-- Schema used (see sql/create_warehouse.sql and reports/warehouse_schema.md):
--   dw.fact_property_listing  one row per retained sold-status record
--       (price, house_size_sqft, price_per_sqft, location_key, property_dim_key, prev_sold_date_key, ...)
--   dw.dim_location           state, city, zip_code
--   dw.dim_property           bedrooms, bathrooms, house_size_band, lot_size_band
--   dw.dim_date               full_date, year, month, month_name, ...
--
-- How to run (from the project root; credentials are read from your own .env or environment, never from this file):
--   psql -h localhost -U <your_user> -d house_price_dw -f sql/example_analytical_query.sql
--
-- Caveats that apply to every result below:
--   * price is the RECORDED price of a sold-status record. The dataset does not document it as a verified
--     completed-sale price, so these are descriptive statistics of recorded prices, not market values.
--   * The warehouse holds only the complete-case records that passed the Phase 2 cleaning rules (696,287 rows),
--     so groups can differ from the population of all U.S. sales.
--   * Each query uses a minimum-observation threshold (HAVING COUNT(*) >= n) so that a group is reported only
--     when it has enough records for a median to be meaningful.

-- ---------------------------------------------------------------------------------------------------------
-- QUERY 1. Typical recorded price by state
-- Business question: which states have the highest and lowest typical recorded prices, and how much evidence
-- stands behind each figure?
-- Interpretation: the MEDIAN is the headline statistic because recorded prices are strongly right-skewed (a few
-- very expensive homes pull the mean up; compare mean_price with median_price). `homes` is the sample size.
-- States with fewer than 1,000 records are omitted.
-- ---------------------------------------------------------------------------------------------------------
SELECT l.state,
       COUNT(*)                                                                       AS homes,
       ROUND(AVG(f.price))                                                            AS mean_price,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price)::numeric, 0)        AS median_price,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price_per_sqft)::numeric, 0) AS median_price_per_sqft,
       ROUND(AVG(f.house_size_sqft))                                                  AS mean_house_size_sqft
FROM dw.fact_property_listing AS f
JOIN dw.dim_location          AS l USING (location_key)
GROUP BY l.state
HAVING COUNT(*) >= 1000
ORDER BY median_price DESC;

-- ---------------------------------------------------------------------------------------------------------
-- QUERY 2. Typical recorded price by house-size band
-- Business question: how does the typical recorded price (and price per square foot) change with house size?
-- Interpretation: price should rise with size while price per square foot usually falls, because small homes
-- carry a larger share of the land and location value. Bands are defined in sql/load_warehouse.sql.
-- ---------------------------------------------------------------------------------------------------------
SELECT p.house_size_band,
       COUNT(*)                                                                       AS homes,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price)::numeric, 0)        AS median_price,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price_per_sqft)::numeric, 0) AS median_price_per_sqft
FROM dw.fact_property_listing AS f
JOIN dw.dim_property          AS p USING (property_dim_key)
GROUP BY p.house_size_band
HAVING COUNT(*) >= 1000
ORDER BY p.house_size_band;

-- ---------------------------------------------------------------------------------------------------------
-- QUERY 3. Cities with the highest typical recorded price (minimum 200 records)
-- Business question: among cities with a reasonable number of records, where are recorded prices highest?
-- Interpretation: a city's median is reliable only with enough records; the 200-record floor is a project
-- choice, not a statistical guarantee. Ties between near-equal medians should not be over-interpreted.
-- ---------------------------------------------------------------------------------------------------------
SELECT l.state,
       l.city,
       COUNT(*)                                                                AS homes,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price)::numeric, 0) AS median_price
FROM dw.fact_property_listing AS f
JOIN dw.dim_location          AS l USING (location_key)
GROUP BY l.state, l.city
HAVING COUNT(*) >= 200
ORDER BY median_price DESC
LIMIT 10;

-- ---------------------------------------------------------------------------------------------------------
-- QUERY 4. Records by month of the recorded previous-sold-date field (a coverage check)
-- Business question: over what period do the records fall, and is the coverage even?
-- Interpretation: dw.dim_date is keyed on the source column prev_sold_date, whose meaning for sold-status
-- records is NOT verified (it is probably, but not confirmed to be, the sale date). Use this as a data-coverage
-- check, not as a time series of market prices. The one record with no date is excluded by the inner join.
-- ---------------------------------------------------------------------------------------------------------
SELECT d.year,
       d.month,
       d.month_name,
       COUNT(*)                                                                AS homes,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price)::numeric, 0) AS median_price
FROM dw.fact_property_listing AS f
JOIN dw.dim_date              AS d ON d.date_key = f.prev_sold_date_key
GROUP BY d.year, d.month, d.month_name
ORDER BY d.year, d.month;
