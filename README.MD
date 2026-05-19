# Fintech Review Analytics

A data engineering and text analytics pipeline designed to scrape, clean, analyze, and store Google Play Store customer reviews for major Ethiopian banking applications. This project transforms unstructured user feedback into structured, actionable business intelligence to help product teams improve user retention, prioritize features, and manage customer complaints.

---

## Project Architecture

```text
fintech-review-analytics/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── raw/
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── nlp_analysis.py
├── tests/
│   └── __init__.py
└── scripts/
    ├── __init__.py
    ├── README.md
    ├── scrape_reviews.py
    ├── schema.sql
    └── load_to_postgres.py
```

---

## 1. Data Collection and Preprocessing

### Scraping Methodology

Customer reviews are extracted automatically from the Google Play Store using the `google-play-scraper` Python library. The pipeline targets three major Ethiopian banks:

* **Commercial Bank of Ethiopia (CBE)** (App ID: `com.cbe.cbebirr`)
* **Bank of Abyssinia (BOA)** (App ID: `com.boamobilebanking.boa`)
* **Dashen Bank** (App ID: `com.dashen.amole`)

### Pipeline Fallback Mechanisms

To handle instances where local geographic account configurations mask data availability, the scraping script utilizes a multi-region fallback loop. If a localized search within the Ethiopian region (`country='et'`) returns zero rows, the script expands its filter parameters to pull from global store databases (`country='us'`). This approach bypasses strict network limitations and ensures a robust sample size exceeding 400 clean reviews per target institution.

### Cleaning Operations

The raw text metrics undergo a systematic sanitization process inside `src/preprocessing.py`:

* **Null Value Filter**: Rows missing essential review text content or star ratings are identified and dropped.
* **De-duplication**: Absolute duplicate records sharing matching text body and bank name assignments are removed to eliminate spam.
* **Date Normalization**: Inconsistent timestamp formats are standardized to uniform `YYYY-MM-DD` strings.

---

## 2. Sentiment and Thematic Analysis

### Sentiment Modeling

The text processing pipeline utilizes a pre-trained Deep Learning transformer model: `distilbert-base-uncased-finetuned-sst-2-english`. This model reads user comments and assigns a sentiment label (`POSITIVE` or `NEGATIVE`) along with an algorithmic confidence score.

### Compute Execution Fallback

To ensure accessibility across varying developer environments, the pipeline initialization forces execution directly on the system CPU (`device=-1`). This parameters bypasses legacy CUDA version mismatch exceptions common on older dedicated graphics hardware (such as Nvidia Quadro cards) and guarantees consistent data mapping without crashing.

### TF-IDF Keyword Extraction

The pipeline uses scikit-learn's `TfidfVectorizer` to extract statistically significant unigrams and bigrams from each bank's review corpus. This data-driven approach surfaces the most distinctive terms per bank, supplementing rule-based theme classification with quantitative evidence.

### Theme Classification

Reviews are grouped into five core operational software themes using rule-based keyword matching heuristics enhanced with TF-IDF-derived insights:

* **Transaction Performance**: Captures speed issues, transfer delays, loading screens, and system timeouts.
* **Account Access Issues**: Tracks login failures, forgotten passwords, blocked profiles, and OTP delivery issues.
* **UI & Design**: Groups feedback regarding interface changes, color themes, menu navigation, and visual style.
* **Customer Support**: Filters mentions of helpline numbers, support agents, bank responses, and branch inquiries.
* **Feature Requests**: Captures user demands for fingerprint login, notifications, budgeting tools, and other enhancements.

---

## 3. Database Engineering (PostgreSQL)

The cleaned data is moved from flat file structures into a persistent relational database named `bank_reviews`. The database architecture utilizes a two-table relational structure optimized to enforce reference integrity and reduce data redundancy.

### Schema Blueprint

#### Banks Table

Stores core institutional application meta-properties.

* `bank_id` (SERIAL, PRIMARY KEY): Unique identifier.
* `bank_name` (VARCHAR, UNIQUE): Short bank identifier (CBE, BOA, Dashen).
* `app_name` (VARCHAR): Full application name listed on the marketplace.

#### Reviews Table

Stores the finalized text logs, scores, and categorical themes.

* `review_id` (SERIAL, PRIMARY KEY): Unique tracking index.
* `bank_id` (INT, FOREIGN KEY): Maps back to the `banks` table with an automatic `ON DELETE CASCADE` constraint.
* `review_text` (TEXT): The customer message string.
* `rating` (INT): Stars awarded, protected by a check constraint (`CHECK (rating >= 1 AND rating <= 5)`).
* `review_date` (DATE): Timestamp of submission.
* `sentiment_label` (VARCHAR): `POSITIVE` or `NEGATIVE` tags.
* `sentiment_score` (NUMERIC): Probability score generated by the model.
* `identified_theme` (VARCHAR): Extracted theme category.
* `source` (VARCHAR): Marketplace identifier, defaulting to 'Google Play'.

---

## 4. Setup and Installation

### Prerequisites

* Python 3.10 or higher
* PostgreSQL instance running locally or via Docker

### Environment Setup

Clone the repository and build your localized virtual environment:

```bash
git clone https://github.com/yourusername/fintech-review-analytics.git
cd fintech-review-analytics
python3 -m venv venv
source venv/bin/activate
```

Install project dependencies and download the English spaCy language model pack:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 5. Pipeline Execution Workflow

Run the following scripts in order to execute the full data engineering cycle:

1. **Scrape Raw Data**:
```bash
python scripts/scrape_reviews.py
```

2. **Clean and Sanitize Data**:
```bash
python src/preprocessing.py
```

3. **Run Sentiment and Topic Modeling**:
```bash
python src/nlp_analysis.py
```

4. **Ingest to PostgreSQL**:
*(Ensure you update your database connection configurations via environment variables or in `scripts/load_to_postgres.py` before running this step).*
```bash
python scripts/load_to_postgres.py
```

### Database Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_NAME` | `bank_reviews` | PostgreSQL database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `postgres` | Database password |
| `DB_HOST` | `localhost` | Database host |
| `DB_PORT` | `5432` | Database port |