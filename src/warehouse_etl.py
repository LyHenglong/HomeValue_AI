"""Phase 3 ETL: load the Phase 2 Parquet outputs into the PostgreSQL star schema and validate it.

    python src/warehouse_etl.py init       create schemas/tables (sql/create_warehouse.sql)
    python src/warehouse_etl.py load       stage the Parquet files and transform into dw.* (one transaction)
    python src/warehouse_etl.py validate   run integrity/consistency checks, write reports/warehouse_validation.*
    python src/warehouse_etl.py all        init + load + validate

Connection settings come from environment variables (optionally a project-root .env file):
PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD. Nothing is hardcoded. Add --create-db to `init`/`all`
to create the database if it does not exist (needs the CREATEDB privilege).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import DBAPIError

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SQL_DIR = ROOT / "sql"
CLEAN_PARQUET = PROCESSED / "sold_clean.parquet"
QUARANTINE_PARQUET = PROCESSED / "quarantine.parquet"
CLEANING_SUMMARY = REPORTS / "cleaning_summary.json"

# Column order must match the staging tables in sql/create_warehouse.sql.
CLEAN_COLS = ["raw_row_id", "property_key", "property_group", "repeated_property", "status", "price", "bed",
              "bath", "house_size", "acre_lot", "acre_lot_was_zero", "city", "state", "zip_code",
              "prev_sold_date", "sold_year", "price_per_sqft", "brokered_by", "street"]
QUARANTINE_COLS = ["raw_row_id", "brokered_by", "status", "price", "bed", "bath", "acre_lot", "street", "city",
                   "state", "zip_code", "house_size", "prev_sold_date", "property_key", "acre_lot_was_zero",
                   "quarantine_reason", "all_flags"]
INT_COLS = {"raw_row_id", "bed", "sold_year"}
BATCH_ROWS = 100_000


# ------------------------------------------------------------------------------------------------
# Connection
# ------------------------------------------------------------------------------------------------
def connection_url(database: str | None = None) -> URL:
    load_dotenv(ROOT / ".env")
    missing = [k for k in ("PGUSER", "PGPASSWORD", "PGDATABASE") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    return URL.create("postgresql+psycopg2", username=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
                      host=os.environ.get("PGHOST", "localhost"), port=int(os.environ.get("PGPORT", "5432")),
                      database=database or os.environ["PGDATABASE"])


def get_engine() -> Engine:
    return create_engine(connection_url(), future=True)


def create_database_if_missing() -> None:
    target = os.environ.get("PGDATABASE") or connection_url().database
    admin = create_engine(connection_url("postgres"), isolation_level="AUTOCOMMIT", future=True)
    with admin.connect() as c:
        if not c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": target}).scalar():
            c.execute(text(f'CREATE DATABASE "{target}"'))  # name comes from our own env, quoted
            print(f"created database {target}")
    admin.dispose()


# ------------------------------------------------------------------------------------------------
# init
# ------------------------------------------------------------------------------------------------
def init_schema(engine: Engine) -> None:
    ddl = (SQL_DIR / "create_warehouse.sql").read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(ddl)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    print("schema ready (staging, dw, audit)")


# ------------------------------------------------------------------------------------------------
# load
# ------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_parquet(cur, path: Path, table: str, cols: list[str]) -> int:
    """Stream a Parquet file into a staging table with COPY, in batches, without loading it all."""
    pf = pq.ParquetFile(path)
    total = 0
    sql = f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')"
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=cols):
        df = batch.to_pandas()
        for c in INT_COLS & set(cols):
            df[c] = df[c].astype("Int64")
        if "prev_sold_date" in df:
            df["prev_sold_date"] = df["prev_sold_date"].dt.strftime("%Y-%m-%d")
        buf = io.StringIO()
        df[cols].to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        cur.copy_expert(sql, buf)
        total += len(df)
    return total


def run_load(engine: Engine) -> int:
    for p in (CLEAN_PARQUET, QUARANTINE_PARQUET, CLEANING_SUMMARY):
        if not p.exists():
            sys.exit(f"{p} not found; run `python src/data_cleaning.py` first")
    summary = json.loads(CLEANING_SUMMARY.read_text(encoding="utf-8"))
    expected_clean = pq.read_metadata(CLEAN_PARQUET).num_rows
    if summary["clean_rows"] != expected_clean:
        sys.exit("reports/cleaning_summary.json does not match sold_clean.parquet; rerun the cleaning pipeline")

    # The audit row is written in its own committed transaction so a failed load is still recorded.
    with engine.begin() as c:
        run_id = c.execute(text(
            "INSERT INTO audit.etl_run (raw_file, raw_sha256, clean_parquet_sha256, quarantine_parquet_sha256) "
            "VALUES (:f, :r, :c, :q) RETURNING run_id"),
            {"f": summary["raw_file"], "r": summary["raw_sha256"],
             "c": sha256(CLEAN_PARQUET), "q": sha256(QUARANTINE_PARQUET)}).scalar_one()
    print(f"etl run {run_id} started")

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:  # one transaction: staging + transform commit together or not at all
            cur.execute("TRUNCATE staging.sold_clean, staging.quarantine")
            n_clean = copy_parquet(cur, CLEAN_PARQUET, "staging.sold_clean", CLEAN_COLS)
            n_quar = copy_parquet(cur, QUARANTINE_PARQUET, "staging.quarantine", QUARANTINE_COLS)
            print(f"staged {n_clean:,} clean rows and {n_quar:,} quarantined rows")
            if n_clean != expected_clean:
                raise RuntimeError(f"staged {n_clean} rows but the Parquet file has {expected_clean}")
            cur.execute("SELECT set_config('etl.run_id', %s, true)", (str(run_id),))
            cur.execute((SQL_DIR / "load_warehouse.sql").read_text(encoding="utf-8"))
        raw.commit()
    except Exception as exc:
        raw.rollback()
        raw.close()
        with engine.begin() as c:
            c.execute(text("UPDATE audit.etl_run SET status='failed', finished_at=now(), error_message=:m "
                           "WHERE run_id=:i"), {"m": str(exc)[:2000], "i": run_id})
        print(f"load FAILED and was rolled back: {exc}")
        raise
    raw.close()
    with engine.begin() as c:
        c.execute(text("UPDATE audit.etl_run SET status='succeeded', finished_at=now() WHERE run_id=:i"), {"i": run_id})
        row = c.execute(text("SELECT fact_rows_inserted, fact_rows_updated, fact_rows_deleted "
                             "FROM audit.etl_run WHERE run_id=:i"), {"i": run_id}).one()
    print(f"etl run {run_id} succeeded: fact inserted={row[0]}, updated={row[1]}, deleted={row[2]}")
    return run_id


# ------------------------------------------------------------------------------------------------
# validate
# ------------------------------------------------------------------------------------------------
FACT_COLS_NOT_NULL = ["raw_row_id", "location_key", "property_dim_key", "price", "house_size_sqft",
                      "price_per_sqft", "acre_lot_was_zero", "repeated_property", "property_group",
                      "source_status", "etl_run_id"]


def run_validation(engine: Engine) -> bool:
    results: list[dict] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    def scalar(sql: str, **p):
        with engine.connect() as c:
            return c.execute(text(sql), p).scalar()

    clean = pd.read_parquet(CLEAN_PARQUET)
    quar_rows = pq.read_metadata(QUARANTINE_PARQUET).num_rows

    # 1. Row counts ---------------------------------------------------------------------------------
    n_stage = scalar("SELECT count(*) FROM staging.sold_clean")
    n_fact = scalar("SELECT count(*) FROM dw.fact_property_listing")
    n_quar = scalar("SELECT count(*) FROM audit.quarantine_record")
    check("fact rows == parquet rows", n_fact == len(clean), f"fact={n_fact:,} parquet={len(clean):,}")
    check("staging rows == parquet rows", n_stage == len(clean), f"staging={n_stage:,}")
    check("quarantine rows == parquet rows", n_quar == quar_rows, f"audit={n_quar:,} parquet={quar_rows:,}")
    check("fact rows == cleaning_summary clean_rows",
          n_fact == json.loads(CLEANING_SUMMARY.read_text(encoding="utf-8"))["clean_rows"])
    check("dim_location rows == distinct (state, city, zip) in parquet",
          scalar("SELECT count(*) FROM dw.dim_location") == clean.groupby(["state", "city", "zip_code"]).ngroups,
          f"dim={scalar('SELECT count(*) FROM dw.dim_location'):,}")

    # 2. Null constraints: declared NOT NULL and no nulls present -----------------------------------
    nn = pd.read_sql(text("SELECT column_name FROM information_schema.columns WHERE table_schema='dw' "
                          "AND table_name='fact_property_listing' AND is_nullable='NO'"), engine)["column_name"].tolist()
    check("fact required columns are declared NOT NULL", set(FACT_COLS_NOT_NULL) <= set(nn),
          f"missing: {sorted(set(FACT_COLS_NOT_NULL) - set(nn))}")
    for col in FACT_COLS_NOT_NULL:
        n = scalar(f"SELECT count(*) FROM dw.fact_property_listing WHERE {col} IS NULL")
        check(f"fact.{col} has no NULLs", n == 0, f"nulls={n}")
    check("only prev_sold_date_key may be NULL, matching parquet",
          scalar("SELECT count(*) FROM dw.fact_property_listing WHERE prev_sold_date_key IS NULL")
          == int(clean["prev_sold_date"].isna().sum()))

    # 3. Key uniqueness -------------------------------------------------------------------------------
    for tbl, key in [("dw.fact_property_listing", "raw_row_id"), ("dw.dim_location", "location_key"),
                     ("dw.dim_location", "state, city, zip_code"), ("dw.dim_property", "property_dim_key"),
                     ("dw.dim_property", "bedrooms, bathrooms, house_size_band, lot_size_band"),
                     ("dw.dim_date", "date_key"), ("audit.quarantine_record", "raw_row_id")]:
        dup = scalar(f"SELECT count(*) FROM (SELECT {key} FROM {tbl} GROUP BY {key} HAVING count(*) > 1) d")
        check(f"{tbl} ({key}) unique", dup == 0, f"duplicate keys={dup}")
    pk = pd.read_sql(text("SELECT conrelid::regclass::text AS tbl FROM pg_constraint WHERE contype='p' "
                          "AND connamespace IN ('dw'::regnamespace, 'audit'::regnamespace)"), engine)["tbl"].tolist()
    check("every dw/audit table has a primary key constraint",
          {"dw.fact_property_listing", "dw.dim_location", "dw.dim_property", "dw.dim_date", "audit.etl_run",
           "audit.quarantine_record"} <= set(pk), str(sorted(pk)))

    # 4. Foreign-key integrity ----------------------------------------------------------------------
    for name, sql in {
        "fact -> dim_location": "SELECT count(*) FROM dw.fact_property_listing f LEFT JOIN dw.dim_location d USING (location_key) WHERE d.location_key IS NULL",
        "fact -> dim_property": "SELECT count(*) FROM dw.fact_property_listing f LEFT JOIN dw.dim_property d USING (property_dim_key) WHERE d.property_dim_key IS NULL",
        "fact -> dim_date": "SELECT count(*) FROM dw.fact_property_listing f LEFT JOIN dw.dim_date d ON d.date_key = f.prev_sold_date_key WHERE f.prev_sold_date_key IS NOT NULL AND d.date_key IS NULL",
        "fact -> audit.etl_run": "SELECT count(*) FROM dw.fact_property_listing f LEFT JOIN audit.etl_run r ON r.run_id = f.etl_run_id WHERE r.run_id IS NULL",
        "dim_location rows with no facts": "SELECT count(*) FROM dw.dim_location d WHERE NOT EXISTS (SELECT 1 FROM dw.fact_property_listing f WHERE f.location_key = d.location_key)",
        "dim_property rows with no facts": "SELECT count(*) FROM dw.dim_property d WHERE NOT EXISTS (SELECT 1 FROM dw.fact_property_listing f WHERE f.property_dim_key = d.property_dim_key)",
    }.items():
        n = scalar(sql)
        check(f"referential integrity: {name}", n == 0, f"orphans={n}")
    fk = scalar("SELECT count(*) FROM pg_constraint WHERE contype='f' AND conrelid = 'dw.fact_property_listing'::regclass")
    check("fact table has 4 declared foreign keys", fk == 4, f"declared={fk}")

    # 5. Row-level consistency between staging and the loaded warehouse ----------------------------
    mism = scalar("""
        SELECT count(*) FROM staging.sold_clean s
        JOIN dw.fact_property_listing f USING (raw_row_id)
        JOIN dw.dim_location l ON l.location_key = f.location_key
        JOIN dw.dim_property p ON p.property_dim_key = f.property_dim_key
        LEFT JOIN dw.dim_date d ON d.date_key = f.prev_sold_date_key
        WHERE s.price::numeric(14,2) IS DISTINCT FROM f.price
           OR s.house_size::int IS DISTINCT FROM f.house_size_sqft
           OR s.acre_lot::numeric(10,4) IS DISTINCT FROM f.acre_lot
           OR s.price_per_sqft::numeric(12,2) IS DISTINCT FROM f.price_per_sqft
           OR s.bed IS DISTINCT FROM p.bedrooms OR s.bath::numeric(4,1) IS DISTINCT FROM p.bathrooms
           OR s.state IS DISTINCT FROM l.state OR s.city IS DISTINCT FROM l.city OR s.zip_code IS DISTINCT FROM l.zip_code
           OR s.prev_sold_date IS DISTINCT FROM d.full_date
           OR s.property_group IS DISTINCT FROM f.property_group OR s.property_key IS DISTINCT FROM f.property_key
           OR s.repeated_property IS DISTINCT FROM f.repeated_property
           OR s.acre_lot_was_zero IS DISTINCT FROM f.acre_lot_was_zero
           OR s.brokered_by::bigint IS DISTINCT FROM f.brokered_by_code OR s.street::bigint IS DISTINCT FROM f.street_code""")
    check("row-level: staging vs fact/dimensions (all columns) has 0 mismatches", mism == 0, f"mismatching rows={mism}")
    # Phase 2 rounded with Python (half-to-even); PostgreSQL's round() on numeric is half-up, so exact
    # halves can differ by one cent. Anything larger than 0.01 is a real inconsistency.
    ppsf = scalar("SELECT count(*) FROM dw.fact_property_listing "
                  "WHERE abs(price_per_sqft - price / house_size_sqft) > 0.0050001")
    check("price_per_sqft within half a cent of price / house_size_sqft", ppsf == 0, f"violations={ppsf}")

    # 6. Measures vs the processed Parquet, computed independently in pandas ------------------------
    agg = pd.read_sql(text("""SELECT count(*) n, sum(price) total, min(price) pmin, max(price) pmax,
                                     sum(house_size_sqft) sqft, count(acre_lot) lots, sum(acre_lot) acres
                              FROM dw.fact_property_listing"""), engine).iloc[0]
    for label, sql_v, pd_v in [("count", agg["n"], len(clean)),
                               ("sum(price)", agg["total"], Decimal(int(clean["price"].sum()))),
                               ("min(price)", agg["pmin"], Decimal(int(clean["price"].min()))),
                               ("max(price)", agg["pmax"], Decimal(int(clean["price"].max()))),
                               ("sum(house_size)", agg["sqft"], int(clean["house_size"].sum())),
                               ("count(acre_lot)", agg["lots"], int(clean["acre_lot"].notna().sum()))]:
        check(f"measure {label} matches parquet", Decimal(sql_v) == Decimal(pd_v), f"warehouse={sql_v} parquet={pd_v}")
    acres_pd = Decimal(str(round(clean["acre_lot"].sum(), 4)))
    check("sum(acre_lot) matches parquet (4-dp tolerance 0.01)", abs(Decimal(agg["acres"]) - acres_pd) < Decimal("0.01"),
          f"warehouse={agg['acres']} parquet={acres_pd}")
    by_state = pd.read_sql(text("""SELECT l.state, count(*) n, sum(f.price) total FROM dw.fact_property_listing f
                                   JOIN dw.dim_location l USING (location_key) GROUP BY l.state"""), engine)
    by_state = by_state.astype({"state": object, "total": float}).set_index("state").sort_index()
    exp = clean.groupby("state").agg(n=("price", "size"), total=("price", "sum"))
    exp.index = exp.index.astype(object)
    exp = exp.sort_index()
    cmp_ok = (list(by_state.index) == list(exp.index)
              and (by_state["n"].to_numpy() == exp["n"].to_numpy()).all()
              and (by_state["total"].to_numpy() == exp["total"].to_numpy()).all())  # prices are whole dollars: exact in float64
    check("per-state row counts and price totals match parquet", cmp_ok, f"states={len(by_state)}")

    # 7. Constraints actually reject bad data (probes run in a transaction that is always rolled back)
    probes = {
        "price <= 0 rejected": "UPDATE dw.fact_property_listing SET price = 0 WHERE raw_row_id = (SELECT min(raw_row_id) FROM dw.fact_property_listing)",
        "unknown location_key rejected": "UPDATE dw.fact_property_listing SET location_key = -1 WHERE raw_row_id = (SELECT min(raw_row_id) FROM dw.fact_property_listing)",
        "duplicate raw_row_id rejected": "INSERT INTO dw.fact_property_listing SELECT * FROM dw.fact_property_listing WHERE raw_row_id = (SELECT min(raw_row_id) FROM dw.fact_property_listing)",
        "non-sold status rejected": "UPDATE dw.fact_property_listing SET source_status = 'for_sale' WHERE raw_row_id = (SELECT min(raw_row_id) FROM dw.fact_property_listing)",
        "bad zip format rejected": "UPDATE dw.dim_location SET zip_code = 'abcde' WHERE location_key = (SELECT min(location_key) FROM dw.dim_location)",
    }
    for name, sql in probes.items():
        raw = engine.raw_connection()
        try:
            try:
                with raw.cursor() as cur:
                    cur.execute(sql)
                check(f"constraint probe: {name}", False, "statement was accepted")
            except Exception as exc:  # psycopg2 IntegrityError / CheckViolation
                check(f"constraint probe: {name}", True, type(exc).__name__)
        finally:
            raw.rollback()
            raw.close()
    check("probes left data unchanged", scalar("SELECT count(*) FROM dw.fact_property_listing WHERE price <= 0") == 0)

    # 8. Audit trail ---------------------------------------------------------------------------------
    last = pd.read_sql(text("SELECT * FROM audit.etl_run ORDER BY run_id DESC LIMIT 1"), engine).iloc[0]
    check("latest etl_run succeeded", last["status"] == "succeeded", f"run_id={last['run_id']}")
    check("etl_run source hash equals cleaning_summary raw_sha256",
          last["raw_sha256"] == json.loads(CLEANING_SUMMARY.read_text(encoding="utf-8"))["raw_sha256"])
    check("etl_run parquet hash equals current sold_clean.parquet", last["clean_parquet_sha256"] == sha256(CLEAN_PARQUET))

    ok = all(r["passed"] for r in results)
    write_validation_report(results, ok, last)
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    print(f"\n{sum(r['passed'] for r in results)}/{len(results)} checks passed")
    return ok


def write_validation_report(results: list[dict], ok: bool, last: pd.Series) -> None:
    REPORTS.mkdir(exist_ok=True)
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "all_passed": ok,
               "latest_etl_run": {k: (None if pd.isna(v) else str(v)) for k, v in last.items()}, "checks": results}
    (REPORTS / "warehouse_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Warehouse validation (generated by `python src/warehouse_etl.py validate`)", "",
             f"Generated: {payload['generated_utc']} UTC. Result: **{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}** "
             f"({sum(r['passed'] for r in results)}/{len(results)}).", "",
             "| Result | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if r['passed'] else 'FAIL'} | {r['check']} | {r['detail']} |" for r in results]
    (REPORTS / "warehouse_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "load", "validate", "all"])
    ap.add_argument("--create-db", action="store_true", help="create the database first if it does not exist")
    args = ap.parse_args()
    if args.create_db and args.command in ("init", "all"):
        create_database_if_missing()
    engine = get_engine()
    try:
        if args.command in ("init", "all"):
            init_schema(engine)
        if args.command in ("load", "all"):
            run_load(engine)
        if args.command in ("validate", "all"):
            if not run_validation(engine):
                sys.exit(1)
    except DBAPIError as exc:
        sys.exit(f"database error: {exc.orig}")


if __name__ == "__main__":
    main()
