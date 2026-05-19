# Scripts

This directory contains executable pipeline scripts for the fintech-review-analytics project. Run them in order to execute the full data engineering cycle.

## Execution Order

| Step | Script | Description |
|------|--------|-------------|
| 1 | `scrape_reviews.py` | Scrapes raw reviews from the Google Play Store for CBE, BOA, and Dashen Bank. Outputs `data/raw/raw_reviews.csv`. |
| 2 | `../src/preprocessing.py` | Cleans the raw data (nulls, duplicates, date normalization). Outputs `data/raw/cleaned_reviews.csv`. |
| 3 | `../src/nlp_analysis.py` | Runs DistilBERT sentiment analysis and TF-IDF thematic classification. Outputs `data/raw/analyzed_reviews.csv`. |
| 4 | `load_to_postgres.py` | Loads the analyzed data into a PostgreSQL `bank_reviews` database. Requires a running PostgreSQL instance. |

## Quick Start

```bash
# Step 1: Scrape reviews
python scripts/scrape_reviews.py

# Step 2: Clean and preprocess
python src/preprocessing.py

# Step 3: Sentiment + thematic analysis
python src/nlp_analysis.py

# Step 4: Load to PostgreSQL (ensure DB is running)
python scripts/load_to_postgres.py
```

## Database Schema

The `schema.sql` file contains the PostgreSQL DDL for the `banks` and `reviews` tables. It can be run independently via:

```bash
psql -U postgres -d bank_reviews -f scripts/schema.sql
```
