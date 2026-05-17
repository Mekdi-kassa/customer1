import os
import pandas as pd
from google_play_scraper import Sort, reviews_all

# Official Google Play app package IDs for the target Ethiopian banks
BANKS = {
    "CBE": "com.cbe.cbebirr",                    # Commercial Bank of Ethiopia (CBE Birr)
    "BOA": "com.boamobilebanking.boa",          # Bank of Abyssinia
    "Dashen": "com.dashen.amole"                 # Dashen Bank (Amole)
}

def scrape_bank_reviews(app_id, bank_name):
    """
    Connects to the Google Play Store to extract all accessible English reviews 
    for a given banking application.
    """
    print(f"[*] Fetching raw reviews for {bank_name} ({app_id})...")
    try:
        scraped_reviews = reviews_all(
            app_id,
            lang='en',      # Restrict to English reviews for reliable downstream NLP parsing
            country='et',   # Target the Ethiopian App Store market space
            sort=Sort.NEWEST
        )
        
        df = pd.DataFrame(scraped_reviews)
        if df.empty:
            print(f"[!] Warning: No reviews retrieved for {bank_name}.")
            return pd.DataFrame()
            
        # Isolate and clean target data elements required by the analytical brief
        df = df[['content', 'score', 'at']].copy()
        df.columns = ['review', 'rating', 'date']
        df['bank'] = bank_name
        df['source'] = 'Google Play'
        
        print(f"[+] Successfully extracted {len(df)} reviews for {bank_name}.")
        return df
    except Exception as e:
        print(f"[-] Error occurred while scraping {bank_name}: {e}")
        return pd.DataFrame()

def preprocess_data(df):
    """
    Cleans raw review payloads by removing empty entries, deduplicating, 
    and normalizing dates to standard YYYY-MM-DD strings.
    """
    if df.empty:
        return df
    
    initial_row_count = len(df)
    
    # 1. Eliminate structural records missing core values
    df = df.dropna(subset=['review', 'rating'])
    
    # 2. Drop historical duplicate comments submitted for the same banking product
    df = df.drop_duplicates(subset=['review', 'bank'])
    
    # 3. Format timestamp entities into clean YYYY-MM-DD configurations
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    final_row_count = len(df)
    print(f"[~] Preprocessing: Dropped {initial_row_count - final_row_count} rows. Retained: {final_row_count}")
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    all_banks_data = []
    
    for bank, app_id in BANKS.items():
        raw_df = scrape_bank_reviews(app_id, bank)
        cleaned_df = preprocess_data(raw_df)
        all_banks_data.append(cleaned_df)
        
    master_df = pd.concat(all_banks_data, ignore_index=True)
    
    output_path = "data/raw/cleaned_reviews.csv"
    master_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Task 1 Complete. Dataset saved to: {output_path} ({len(master_df)} records)")