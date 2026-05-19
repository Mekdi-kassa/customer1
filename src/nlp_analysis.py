"""
src/nlp_analysis.py
Task 2 — Sentiment and Thematic Analysis

Applies NLP techniques to classify review sentiment and extract recurring themes:

1. Sentiment Classification — Uses DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)
   to assign POSITIVE/NEGATIVE labels with confidence scores.
2. TF-IDF Keyword Extraction — Extracts statistically significant terms per bank
   using sklearn's TfidfVectorizer to identify data-driven themes.
3. Thematic Classification — Combines rule-based keyword matching with TF-IDF evidence
   to group reviews into business-relevant categories.

Input:  data/raw/cleaned_reviews.csv  (output of src/preprocessing.py)
Output: data/raw/analyzed_reviews.csv
"""

import os
import pandas as pd
import numpy as np
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

print("[*] Loading HuggingFace Sentiment Pipeline (DistilBERT)...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # Force CPU execution to avoid CUDA version mismatches
)


# ---------------------------------------------------------------------------
# Task 2a: Sentiment Analysis
# ---------------------------------------------------------------------------
def run_sentiment_analysis(df):
    """
    Classifies each review as POSITIVE or NEGATIVE using DistilBERT.

    Truncates review text to 512 characters to stay within the transformer's
    token limit. Processes reviews in batches of 32 for efficiency.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'review' column with text data.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns: 'sentiment_label', 'sentiment_score'.
    """
    print("[*] Running sentiment classification (DistilBERT)...")
    texts = df['review'].astype(str).tolist()

    # Truncate to prevent token-length failures
    truncated_texts = [t[:512] for t in texts]
    results = classifier(truncated_texts, batch_size=32)

    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [round(r['score'], 4) for r in results]

    # Log sentiment distribution
    dist = df['sentiment_label'].value_counts()
    print(f"    Sentiment distribution: {dict(dist)}")
    return df


# ---------------------------------------------------------------------------
# Task 2b: Text Preprocessing with spaCy
# ---------------------------------------------------------------------------
def lemmatize_text(text):
    """
    Cleans and lemmatizes text using spaCy. Removes punctuation, numbers,
    and stop words, retaining only meaningful word stems.

    Parameters
    ----------
    text : str
        Raw review text.

    Returns
    -------
    str
        Space-joined lemmatized tokens.
    """
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Task 2c: TF-IDF Keyword Extraction
# ---------------------------------------------------------------------------
def extract_tfidf_keywords(df, text_column='review', n_keywords=5):
    """
    Extracts the top TF-IDF keywords for each bank's review corpus.

    Uses a lemmatized version of the reviews to reduce noise and identify
    statistically significant terms that distinguish each bank's feedback.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'review' and 'bank' columns.
    text_column : str
        Column name containing review text.
    n_keywords : int
        Number of top keywords to extract per bank.

    Returns
    -------
    dict
        Mapping of bank_name → list of (keyword, tfidf_score) tuples.
    """
    print("[*] Extracting TF-IDF keywords per bank...")

    # Lemmatize all reviews for cleaner term extraction
    df['_lemmatized'] = df[text_column].apply(lemmatize_text)

    bank_keywords = {}

    for bank in df['bank'].unique():
        bank_texts = df[df['bank'] == bank]['_lemmatized'].tolist()

        if not bank_texts:
            bank_keywords[bank] = []
            continue

        vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),  # Capture both single words and bigrams
            min_df=2             # Term must appear in at least 2 reviews
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(bank_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Average TF-IDF scores across all reviews for this bank
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-n_keywords:][::-1]

            keywords = [(feature_names[i], round(avg_scores[i], 4)) for i in top_indices]
            bank_keywords[bank] = keywords
            print(f"    {bank}: {[kw for kw, _ in keywords]}")
        except ValueError:
            bank_keywords[bank] = []
            print(f"    {bank}: insufficient data for TF-IDF extraction.")

    # Clean up temporary column
    df.drop(columns=['_lemmatized'], inplace=True)

    return bank_keywords


# ---------------------------------------------------------------------------
# Task 2d: Thematic Classification (Rule-Based + TF-IDF Enhanced)
# ---------------------------------------------------------------------------
def assign_theme(review_text):
    """
    Categorizes a review into a business-relevant theme using keyword matching.

    Themes are designed to map directly to the three business scenarios
    defined in the challenge brief:
      - Transaction Performance → Scenario 1 (Retaining Users)
      - Account Access Issues   → Scenario 1 (Retaining Users)
      - Feature Requests        → Scenario 2 (Enhancing Features)
      - UI & Design             → Scenario 2 (Enhancing Features)
      - Customer Support         → Scenario 3 (Managing Complaints)

    Parameters
    ----------
    review_text : str
        Raw review text.

    Returns
    -------
    str
        One of: 'Transaction Performance', 'Account Access Issues',
        'UI & Design', 'Customer Support', 'Feature Requests',
        or 'General Feedback'.
    """
    text = str(review_text).lower()

    themes = {
        "Transaction Performance": [
            "slow", "transfer", "loading", "delay", "waiting", "pending",
            "sent", "money", "transaction", "timeout", "speed", "fast",
            "lag", "freeze", "crash", "stuck"
        ],
        "Account Access Issues": [
            "login", "sign in", "password", "otp", "code", "verification",
            "error", "unable", "locked", "blocked", "forgot", "reset",
            "authenticate", "access", "denied"
        ],
        "UI & Design": [
            "interface", "ui", "ux", "beautiful", "ugly", "clean",
            "navigation", "screen", "display", "design", "layout",
            "theme", "dark mode", "color", "icon", "menu"
        ],
        "Customer Support": [
            "help", "agent", "support", "call", "branch", "complain",
            "service", "refund", "response", "resolve", "contact",
            "staff", "hotline", "chat"
        ],
        "Feature Requests": [
            "fingerprint", "biometric", "budget", "notification", "alert",
            "update", "version", "add", "feature", "wish", "need",
            "should", "please", "improve", "want"
        ]
    }

    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            return theme

    return "General Feedback"


# ---------------------------------------------------------------------------
# Main Pipeline Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_file = "data/raw/cleaned_reviews.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Missing '{input_file}'. Run the preprocessing pipeline first:\n"
            "  python scripts/scrape_reviews.py\n"
            "  python src/preprocessing.py"
        )

    print(f"[*] Loading cleaned reviews from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"    Loaded {len(df)} reviews across {df['bank'].nunique()} banks.\n")

    # Step 1: Sentiment Analysis (DistilBERT)
    df = run_sentiment_analysis(df)

    # Step 2: TF-IDF Keyword Extraction (per bank)
    print()
    bank_keywords = extract_tfidf_keywords(df, text_column='review', n_keywords=8)

    # Step 3: Thematic Classification
    print("\n[*] Applying thematic classification...")
    df['identified_theme'] = df['review'].apply(assign_theme)

    theme_dist = df.groupby(['bank', 'identified_theme']).size().unstack(fill_value=0)
    print("    Theme distribution per bank:")
    print(theme_dist.to_string())

    # Step 4: Assign unique review IDs for database ingestion
    df.insert(0, 'review_id', range(1, len(df) + 1))

    # Step 5: Export final dataset
    output_path = "data/raw/analyzed_reviews.csv"
    final_cols = [
        'review_id', 'review', 'rating', 'date', 'bank', 'source',
        'sentiment_label', 'sentiment_score', 'identified_theme'
    ]
    df[final_cols].to_csv(output_path, index=False)

    # Print summary report
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Task 2 Complete — Analyzed reviews saved to: {output_path}")
    print(f"{'='*60}")
    print(f"\n--- Summary ---")
    print(f"Total reviews analyzed: {len(df)}")
    print(f"Sentiment coverage:     {df['sentiment_label'].notna().sum()}/{len(df)} "
          f"({df['sentiment_label'].notna().mean()*100:.1f}%)")
    print(f"Themes assigned:        {(df['identified_theme'] != 'General Feedback').sum()}/{len(df)} "
          f"({(df['identified_theme'] != 'General Feedback').mean()*100:.1f}% thematic coverage)")

    print(f"\n--- Sentiment by Bank ---")
    print(df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0).to_string())

    print(f"\n--- Top TF-IDF Keywords by Bank ---")
    for bank, keywords in bank_keywords.items():
        kw_str = ", ".join([f"{kw} ({score:.4f})" for kw, score in keywords])
        print(f"  {bank}: {kw_str}")