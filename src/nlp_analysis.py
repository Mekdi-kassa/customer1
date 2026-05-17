import pandas as pd
import numpy as np
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize spaCy and HuggingFace pipelines
nlp = spacy.load("en_core_web_sm")
print("Loading HuggingFace Sentiment Pipeline (DistilBERT)...")
classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1 # set to 0 if executing on a GPU enabled environment
)

def run_sentiment_analysis(df):
    """
    Applies DistilBERT to classify sentiment.
    Handles long text cuts to match transformer token constraints (max 512 tokens).
    """
    print("Running Sentiment Classification...")
    labels = []
    scores = []
    
    # Deduplicate processing array for memory safety
    texts = df['review'].astype(str).tolist()
    
    # Run predictions in batches for optimization
    # Truncate strings to prevent token length failures
    truncated_texts = [t[:512] for t in texts]
    results = classifier(truncated_texts, batch_size=32)
    
    for res in results:
        labels.append(res['label'])
        scores.append(res['score'])
        
    df['sentiment_label'] = labels
    df['sentiment_score'] = scores
    return df

def lemmatize_text(text):
    """
    Cleans structural syntax noise from text using spaCy
    """
    doc = nlp(str(text).lower())
    # Keep only words, discard punctuation, numbers, and stop words
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def assign_rule_based_theme(review_text):
    """
    Categorizes reviews into business themes based on target fintech keyword indicators.
    """
    text = str(review_text).lower()
    
    # Domain keyword groups mapping directly to our 3 Core Business Scenarios
    themes = {
        "Transaction Performance": ["slow", "transfer", "loading", "delay", "waiting", "pending", "sent", "money"],
        "Account Access Issues": ["login", "sign in", "password", "otp", "code", "verification", "error", "unable"],
        "UI & Design": ["interface", "ui", "ux", "beautiful", "ugly", "clean", "navigation", "screen", "display"],
        "Customer Support": ["help", "agent", "support", "call", "branch", "complain", "service", "refund"]
    }
    
    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            return theme
            
    return "General Feedback"

if __name__ == "__main__":
    input_file = "data/raw/cleaned_reviews.csv"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}. Please run Task 1 script first!")
        
    df = pd.read_csv(input_file)
    
    # Step 1: Execute Sentiment Analysis
    df = run_sentiment_analysis(df)
    
    # Step 2: Lemmatize for advanced keyword extraction metrics 
    print("Lemmatizing reviews for keyword tracking...")
    df['cleaned_review_tokens'] = df['review'].apply(lemmatize_text)
    
    # Step 3: Run Thematic Categorization
    print("Applying structural thematic grouping...")
    df['identified_theme'] = df['review'].apply(assign_rule_based_theme)
    
    # Add unique ID for Database loading sequence safely
    df.insert(0, 'review_id', range(1, len(df) + 1))
    
    # Final Output Data Export
    output_path = "data/raw/analyzed_reviews.csv"
    # Keep final format structured for Task 3 DB insertion rules
    final_cols = ['review_id', 'review', 'rating', 'date', 'bank', 'source', 'sentiment_label', 'sentiment_score', 'identified_theme']
    df[final_cols].to_csv(output_path, index=False)
    
    print(f"\nTask 2 Successful! Data rich processing file exported directly to: {output_path}")
    
    # Simple metrics generation validation
    print("\n--- Summary Performance Matrix ---")
    print(df.groupby(['bank', 'sentiment_label']).size().unstack(fill_value=0))