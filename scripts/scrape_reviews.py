"""
scripts/scrape_reviews.py
Task 1 — Data Collection

Connects to the Google Play Store via google-play-scraper to extract
English-language customer reviews for three major Ethiopian banking apps.

Implements a multi-region fallback: tries the Ethiopian store first (country='et'),
and if zero results are returned, expands to the US store (country='us') to
bypass regional data availability constraints.

Output: data/raw/raw_reviews.csv (unprocessed scraped reviews)
"""

import os
import pandas as pd
from google_play_scraper import Sort, reviews_all


# Official Google Play app package IDs for the target Ethiopian banks
BANKS = {
    "CBE": "com.cbe.cbebirr",               # Commercial Bank of Ethiopia (CBE Birr)
    "BOA": "com.boamobilebanking.boa",       # Bank of Abyssinia
    "Dashen": "com.dashen.amole"             # Dashen Bank (Amole)
}


def scrape_bank_reviews(app_id, bank_name):
    """
    Extracts all accessible English reviews for a given banking application
    from the Google Play Store.

    Uses a multi-region fallback strategy:
      1. First attempts the Ethiopian market (country='et').
      2. If zero reviews are returned, retries with the US market (country='us')
         to bypass geographic data restrictions.

    Parameters
    ----------
    app_id : str
        The Google Play Store package identifier for the target app.
    bank_name : str
        Human-readable name of the bank (used as a label in the output).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: review, rating, date, bank, source.
        Returns an empty DataFrame if scraping fails entirely.
    """
    regions = [
        ("et", "Ethiopian"),
        ("us", "US (fallback)")
    ]

    for country_code, region_label in regions:
        print(f"[*] Fetching reviews for {bank_name} ({app_id}) — {region_label} store...")
        try:
            scraped_reviews = reviews_all(
                app_id,
                lang='en',
                country=country_code,
                sort=Sort.NEWEST
            )

            df = pd.DataFrame(scraped_reviews)

            if df.empty:
                print(f"[!] No reviews found in {region_label} store for {bank_name}.")
                continue  # Try next region

            # Isolate and rename target columns
            df = df[['content', 'score', 'at']].copy()
            df.columns = ['review', 'rating', 'date']
            df['bank'] = bank_name
            df['source'] = 'Google Play'

            print(f"[+] Extracted {len(df)} reviews for {bank_name} from {region_label} store.")
            return df

        except Exception as e:
            print(f"[-] Error scraping {bank_name} in {region_label} store: {e}")
            continue

    # All regions exhausted
    print(f"[!!] Failed to retrieve any reviews for {bank_name} across all regions.")
    return pd.DataFrame()


if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs("data/raw", exist_ok=True)

    all_banks_data = []

    for bank_name, app_id in BANKS.items():
        raw_df = scrape_bank_reviews(app_id, bank_name)
        if not raw_df.empty:
            all_banks_data.append(raw_df)

    if all_banks_data:
        master_df = pd.concat(all_banks_data, ignore_index=True)
        output_path = "data/raw/raw_reviews.csv"
        master_df.to_csv(output_path, index=False)
        print(f"\n[SUCCESS] Task 1a Complete — Raw reviews saved to: {output_path}")
        print(f"          Total reviews collected: {len(master_df)}")
        print(f"          Reviews per bank:")
        print(master_df['bank'].value_counts().to_string(header=False))
    else:
        print("\n[FAIL] No reviews were collected for any bank. Check network and app IDs.")
