"""
src/preprocessing.py
Task 1 — Data Preprocessing

Cleans raw scraped review data by removing nulls, deduplicating,
and normalizing date formats. Produces a clean, analysis-ready CSV.

Input:  data/raw/raw_reviews.csv  (output of scripts/scrape_reviews.py)
Output: data/raw/cleaned_reviews.csv
"""

import os
import pandas as pd


def preprocess_data(df):
    """
    Cleans raw review data through three sequential operations:

    1. Null Value Filter — Drops rows missing essential 'review' text or 'rating'.
    2. De-duplication — Removes exact duplicate entries sharing the same
       review text and bank name to eliminate spam and re-submissions.
    3. Date Normalization — Converts inconsistent timestamp formats into
       uniform YYYY-MM-DD strings.

    Parameters
    ----------
    df : pd.DataFrame
        Raw review data with columns: review, rating, date, bank, source.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with the same column structure.
    """
    if df.empty:
        print("[!] Received empty DataFrame — nothing to preprocess.")
        return df

    initial_count = len(df)

    # 1. Drop rows missing core fields
    null_before = df[['review', 'rating']].isnull().any(axis=1).sum()
    df = df.dropna(subset=['review', 'rating'])
    print(f"    Null filter: dropped {null_before} rows missing review/rating.")

    # 2. Remove duplicate reviews within the same bank
    dup_before = len(df)
    df = df.drop_duplicates(subset=['review', 'bank'])
    dup_dropped = dup_before - len(df)
    print(f"    De-duplication: dropped {dup_dropped} duplicate rows.")

    # 3. Normalize dates to YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

    final_count = len(df)
    total_dropped = initial_count - final_count
    print(f"    Summary: {initial_count} → {final_count} rows ({total_dropped} removed, "
          f"{total_dropped / initial_count * 100:.1f}% loss)")

    return df


if __name__ == "__main__":
    input_path = "data/raw/raw_reviews.csv"
    output_path = "data/raw/cleaned_reviews.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Missing '{input_path}'. Run 'python scripts/scrape_reviews.py' first."
        )

    print(f"[*] Loading raw reviews from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"    Loaded {len(df)} raw reviews.")

    print("[*] Running preprocessing pipeline...")
    df = preprocess_data(df)

    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Task 1b Complete — Cleaned reviews saved to: {output_path}")
    print(f"          Final dataset: {len(df)} reviews across {df['bank'].nunique()} banks.")
