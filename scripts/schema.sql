-- =============================================================================
-- scripts/schema.sql
-- Task 3 — PostgreSQL Schema for bank_reviews Database
--
-- Creates a two-table relational structure:
--   1. banks   — Stores institutional metadata for each banking application.
--   2. reviews — Stores processed review data with sentiment and theme labels.
--
-- Usage:
--   createdb bank_reviews
--   psql -U postgres -d bank_reviews -f scripts/schema.sql
-- =============================================================================

-- Drop existing tables if re-running (development only)
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS banks CASCADE;

-- -----------------------------------------------------------------------------
-- Banks Table
-- Stores core institutional application meta-properties.
-- -----------------------------------------------------------------------------
CREATE TABLE banks (
    bank_id   SERIAL       PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name  VARCHAR(255) NOT NULL
);

-- Seed the three target Ethiopian banks
INSERT INTO banks (bank_name, app_name) VALUES
    ('CBE',    'CBE Birr'),
    ('BOA',    'Bank of Abyssinia Mobile'),
    ('Dashen', 'Amole by Dashen Bank');

-- -----------------------------------------------------------------------------
-- Reviews Table
-- Stores the finalized review text, ratings, sentiment scores, and themes.
-- References the banks table via a foreign key with cascade delete.
-- -----------------------------------------------------------------------------
CREATE TABLE reviews (
    review_id       SERIAL       PRIMARY KEY,
    bank_id         INTEGER      NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text     TEXT         NOT NULL,
    rating          INTEGER      NOT NULL CHECK (rating >= 1 AND rating <= 5),
    review_date     DATE,
    sentiment_label VARCHAR(20),
    sentiment_score NUMERIC(5, 4),
    identified_theme VARCHAR(100),
    source          VARCHAR(50)  DEFAULT 'Google Play'
);

-- Create index on bank_id for faster joins and filtering
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);

-- Create index on sentiment_label for aggregation queries
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);

-- =============================================================================
-- Verification Queries (run after data insertion to validate integrity)
-- =============================================================================

-- Count reviews per bank
-- SELECT b.bank_name, COUNT(r.review_id) AS review_count
-- FROM banks b LEFT JOIN reviews r ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name;

-- Average rating per bank
-- SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating
-- FROM banks b JOIN reviews r ON b.bank_id = r.bank_id
-- GROUP BY b.bank_name;

-- Check for nulls in key columns
-- SELECT
--     COUNT(*) FILTER (WHERE review_text IS NULL) AS null_text,
--     COUNT(*) FILTER (WHERE rating IS NULL) AS null_rating,
--     COUNT(*) FILTER (WHERE sentiment_label IS NULL) AS null_sentiment
-- FROM reviews;
