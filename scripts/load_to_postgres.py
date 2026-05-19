"""
scripts/load_to_postgres.py
Task 3 — Database Insertion Script

Reads the analyzed review CSV and loads it into a PostgreSQL database
named 'bank_reviews' using psycopg2.

Prerequisites:
    1. PostgreSQL must be running locally.
    2. The database 'bank_reviews' must exist:
           createdb bank_reviews
    3. The schema must be applied:
           psql -U postgres -d bank_reviews -f scripts/schema.sql
    4. Set environment variables for database credentials:
           DB_NAME     (default: bank_reviews)
           DB_USER     (default: postgres)
           DB_PASSWORD (default: postgres)
           DB_HOST     (default: localhost)
           DB_PORT     (default: 5432)

Usage:
    python scripts/load_to_postgres.py
"""

import os
import pandas as pd
import psycopg2
from psycopg2 import sql


def get_db_connection():
    """
    Creates a connection to the PostgreSQL database using environment variables.
    Falls back to sensible local development defaults.
    """
    conn = psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "bank_reviews"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", "postgres"),
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432")
    )
    return conn


def create_tables(conn):
    """
    Creates the banks and reviews tables if they don't already exist.
    Reads and executes the schema.sql file.
    """
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    print("[+] Database schema applied successfully.")


def get_bank_id_map(conn):
    """
    Retrieves the bank_name → bank_id mapping from the banks table.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT bank_id, bank_name FROM banks;")
        rows = cur.fetchall()
    return {name: bid for bid, name in rows}


def insert_reviews(conn, df, bank_id_map):
    """
    Inserts review records into the reviews table in batch.

    Parameters
    ----------
    conn : psycopg2 connection
    df : pd.DataFrame
        Analyzed review data with columns matching the reviews table.
    bank_id_map : dict
        Mapping of bank_name → bank_id from the banks table.
    """
    insert_query = """
        INSERT INTO reviews (bank_id, review_text, rating, review_date,
                             sentiment_label, sentiment_score, identified_theme, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    inserted = 0
    skipped = 0

    with conn.cursor() as cur:
        for _, row in df.iterrows():
            bank_name = row.get('bank', '')
            bank_id = bank_id_map.get(bank_name)

            if bank_id is None:
                skipped += 1
                continue

            values = (
                bank_id,
                str(row.get('review', '')),
                int(row.get('rating', 0)),
                row.get('date', None),
                row.get('sentiment_label', None),
                float(row.get('sentiment_score', 0)) if pd.notna(row.get('sentiment_score')) else None,
                row.get('identified_theme', None),
                row.get('source', 'Google Play')
            )

            cur.execute(insert_query, values)
            inserted += 1

    conn.commit()
    print(f"[+] Inserted {inserted} reviews into the database.")
    if skipped:
        print(f"[!] Skipped {skipped} rows (bank name not found in banks table).")


def run_verification_queries(conn):
    """
    Executes verification queries to validate data integrity after insertion.
    """
    print("\n--- Verification Queries ---\n")

    with conn.cursor() as cur:
        # 1. Count reviews per bank
        cur.execute("""
            SELECT b.bank_name, COUNT(r.review_id) AS review_count
            FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id
            GROUP BY b.bank_name
            ORDER BY b.bank_name;
        """)
        print("Reviews per bank:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} reviews")

        # 2. Average rating per bank
        cur.execute("""
            SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating
            FROM banks b JOIN reviews r ON b.bank_id = r.bank_id
            GROUP BY b.bank_name
            ORDER BY b.bank_name;
        """)
        print("\nAverage rating per bank:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]}")

        # 3. Check for nulls in key columns
        cur.execute("""
            SELECT
                COUNT(*) AS total_reviews,
                COUNT(*) FILTER (WHERE review_text IS NULL) AS null_text,
                COUNT(*) FILTER (WHERE rating IS NULL) AS null_rating,
                COUNT(*) FILTER (WHERE sentiment_label IS NULL) AS null_sentiment
            FROM reviews;
        """)
        result = cur.fetchone()
        print(f"\nData integrity check:")
        print(f"  Total reviews:    {result[0]}")
        print(f"  Null text:        {result[1]}")
        print(f"  Null rating:      {result[2]}")
        print(f"  Null sentiment:   {result[3]}")

        # 4. Sentiment distribution
        cur.execute("""
            SELECT b.bank_name, r.sentiment_label, COUNT(*) AS count
            FROM banks b JOIN reviews r ON b.bank_id = r.bank_id
            GROUP BY b.bank_name, r.sentiment_label
            ORDER BY b.bank_name, r.sentiment_label;
        """)
        print("\nSentiment distribution:")
        for row in cur.fetchall():
            print(f"  {row[0]} — {row[1]}: {row[2]}")


if __name__ == "__main__":
    input_path = "data/raw/analyzed_reviews.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Missing '{input_path}'. Run the analysis pipeline first:\n"
            "  python scripts/scrape_reviews.py\n"
            "  python src/preprocessing.py\n"
            "  python src/nlp_analysis.py"
        )

    print(f"[*] Loading analyzed reviews from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"    Loaded {len(df)} reviews.")

    print("[*] Connecting to PostgreSQL...")
    conn = get_db_connection()

    try:
        print("[*] Applying database schema...")
        create_tables(conn)

        print("[*] Retrieving bank ID mapping...")
        bank_id_map = get_bank_id_map(conn)
        print(f"    Found banks: {bank_id_map}")

        print("[*] Inserting reviews...")
        insert_reviews(conn, df, bank_id_map)

        run_verification_queries(conn)

        print("\n[SUCCESS] Task 3 Complete — Data loaded into PostgreSQL 'bank_reviews' database.")
    except Exception as e:
        print(f"\n[ERROR] Database operation failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        print("[*] Database connection closed.")
