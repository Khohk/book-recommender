"""
Script 00: Prepare Clean Datasets
Process raw Parquet files into 2 clean datasets:
1. Master Books Dataset (1 row per book) - for NLP Retrieval
2. Ratings Dataset (many rows per book) - for CF
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

# ========================================
# CONFIGURATION
# ========================================

RAW_REVIEWS_FILE = "data/raw/reviews.parquet"  # Your reviews dataset
RAW_BOOKS_FILE = "data/raw/books_details.parquet"  # Your books details dataset

OUTPUT_BOOKS = "data/processed/master_books.csv"  # Clean books (1 row/book)
OUTPUT_RATINGS = "data/processed/ratings.csv"  # Clean ratings (many rows/book)

# ========================================
# LOAD RAW DATA
# ========================================

print("="*80)
print("[STEP 1] LOADING RAW DATA")
print("="*80)

print(f"\n[*] Loading reviews from: {RAW_REVIEWS_FILE}")
reviews_df = pd.read_parquet(RAW_REVIEWS_FILE)
print(f"[OK] Loaded {len(reviews_df)} reviews")
print("\n[INFO] Reviews columns:")
print(reviews_df.info())

print(f"\n[*] Loading books details from: {RAW_BOOKS_FILE}")
books_df = pd.read_parquet(RAW_BOOKS_FILE)
print(f"[OK] Loaded {len(books_df)} books")
print("\n[INFO] Books columns:")
print(books_df.info())

# ========================================
# CREATE MASTER BOOKS DATASET
# ========================================

print("\n" + "="*80)
print("[STEP 2] CREATING MASTER BOOKS DATASET")
print("="*80)

# Start with books_details as base (1 row per book)
master_books = books_df.copy()

# Rename columns for consistency
master_books = master_books.rename(columns={
    'Title': 'title',
    'description': 'description',
    'authors': 'authors',
    'publisher': 'publisher',
    'publishedDate': 'published_date',
    'categories': 'categories',
    'ratingsCount': 'ratings_count'
})

# Create book_id from Title (use as unique identifier)
# Clean title to create consistent IDs
master_books['book_id'] = master_books['title'].str.strip().str.lower()
master_books['book_id'] = master_books['book_id'].str.replace(r'[^\w\s]', '', regex=True)
master_books['book_id'] = master_books['book_id'].str.replace(r'\s+', '_', regex=True)

# Add prefix to ensure uniqueness
master_books['book_id'] = 'book_' + master_books.index.astype(str)

print(f"\n[*] Created book_id for {len(master_books)} unique books")

# Aggregate review statistics per book
print("\n[*] Aggregating review statistics with SMART FILTERING...")

def aggregate_reviews_smart(group):
    """
    Smart review aggregation with quality filtering
    Priority: helpful_vote > rating > length
    """
    # Parse helpful votes (format: "numerator/denominator")
    def parse_helpful(x):
        try:
            if pd.isna(x) or str(x) == 'nan':
                return 0
            num, denom = str(x).split('/')
            return int(num) if int(denom) > 0 else 0
        except:
            return 0
    
    group['helpful_score'] = group['review/helpfulness'].apply(parse_helpful)
    group['review_length'] = group['review/text'].astype(str).str.len()
    
    # Filter quality reviews (rating >= 3, length >= 50 chars)
    quality_reviews = group[
        (group['review/score'] >= 3.0) & 
        (group['review_length'] >= 50)
    ].copy()
    
    if len(quality_reviews) == 0:
        quality_reviews = group  # Fallback to all reviews
    
    # Sort by: helpful_score DESC, rating DESC, length DESC
    quality_reviews = quality_reviews.sort_values(
        by=['helpful_score', 'review/score', 'review_length'],
        ascending=[False, False, False]
    )
    
    # Take top 10 reviews
    top_reviews = quality_reviews.head(10)
    
    # Aggregate stats
    return pd.Series({
        'avg_rating': group['review/score'].mean(),
        'rating_std': group['review/score'].std(),
        'num_ratings': len(group),
        'top_reviews_text': ' '.join(top_reviews['review/text'].astype(str).tolist()),
        'top_reviews_summary': ' '.join(top_reviews['review/summary'].fillna('').astype(str).tolist()),
        'avg_helpful_score': top_reviews['helpful_score'].mean()
    })

review_stats = reviews_df.groupby('Title').apply(aggregate_reviews_smart).reset_index()

print(f"[OK] Aggregated reviews for {len(review_stats)} books with smart filtering")

# Merge review stats with master books
master_books = master_books.merge(
    review_stats,
    left_on='title',
    right_on='Title',
    how='left'
).drop(columns=['Title'])  # Drop duplicate title column

# Rename aggregated columns
master_books = master_books.rename(columns={
    'top_reviews_text': 'aggregated_reviews',
    'top_reviews_summary': 'aggregated_summaries'
})

# Fill missing values
master_books['avg_rating'] = master_books['avg_rating'].fillna(0)
master_books['num_ratings'] = master_books['num_ratings'].fillna(0)
master_books['avg_helpful_score'] = master_books['avg_helpful_score'].fillna(0)
master_books['aggregated_reviews'] = master_books['aggregated_reviews'].fillna('')
master_books['aggregated_summaries'] = master_books['aggregated_summaries'].fillna('')

# Select final columns for master books
final_books_columns = [
    'book_id',
    'title',
    'authors',
    'description',
    'categories',
    'publisher',
    'published_date',
    'avg_rating',
    'rating_std',
    'num_ratings',
    'avg_helpful_score',
    'aggregated_reviews',
    'aggregated_summaries'
]

master_books = master_books[final_books_columns]

# Remove duplicates (if any)
original_len = len(master_books)
master_books = master_books.drop_duplicates(subset=['title'], keep='first')
if len(master_books) < original_len:
    print(f"[WARNING] Removed {original_len - len(master_books)} duplicate books")

print(f"\n[OK] Master Books Dataset created: {len(master_books)} unique books")
print("\n[INFO] Sample master book:")
print(master_books.head(1).T)

# Save master books
os.makedirs(os.path.dirname(OUTPUT_BOOKS), exist_ok=True)
master_books.to_csv(OUTPUT_BOOKS, index=False)
print(f"\n[SAVE] Master books saved to: {OUTPUT_BOOKS}")

# ========================================
# CREATE RATINGS DATASET
# ========================================

print("\n" + "="*80)
print("[STEP 3] CREATING RATINGS DATASET (for CF)")
print("="*80)

# Create book_id mapping (Title -> book_id)
book_id_map = dict(zip(master_books['title'], master_books['book_id']))

# Extract ratings from reviews
ratings_df = reviews_df[['User_id', 'Title', 'review/score', 'review/time']].copy()

# Rename columns
ratings_df = ratings_df.rename(columns={
    'User_id': 'user_id',
    'Title': 'title',
    'review/score': 'rating',
    'review/time': 'timestamp'
})

# Map title to book_id
ratings_df['book_id'] = ratings_df['title'].map(book_id_map)

# Remove ratings for books not in master dataset
ratings_df = ratings_df.dropna(subset=['book_id'])

# Remove rows with missing user_id or rating
ratings_df = ratings_df.dropna(subset=['user_id', 'rating'])

# Select final columns
ratings_df = ratings_df[['user_id', 'book_id', 'rating', 'timestamp']]

# Convert timestamp to integer
ratings_df['timestamp'] = ratings_df['timestamp'].astype(int)

print(f"\n[OK] Ratings Dataset created: {len(ratings_df)} ratings")
print(f"   Unique users: {ratings_df['user_id'].nunique()}")
print(f"   Unique books: {ratings_df['book_id'].nunique()}")
print(f"   Avg ratings per user: {len(ratings_df) / ratings_df['user_id'].nunique():.1f}")
print(f"   Avg ratings per book: {len(ratings_df) / ratings_df['book_id'].nunique():.1f}")

print("\n[INFO] Rating distribution:")
print(ratings_df['rating'].value_counts().sort_index())

print("\n[INFO] Sample ratings:")
print(ratings_df.head(10))

# Save ratings
ratings_df.to_csv(OUTPUT_RATINGS, index=False)
print(f"\n[SAVE] Ratings saved to: {OUTPUT_RATINGS}")

# ========================================
# DATA QUALITY CHECK
# ========================================

print("\n" + "="*80)
print("[STEP 4] DATA QUALITY CHECK")
print("="*80)

# Check 1: All books in ratings exist in master books
ratings_books = set(ratings_df['book_id'].unique())
master_books_ids = set(master_books['book_id'].unique())

orphan_books = ratings_books - master_books_ids
if orphan_books:
    print(f"[WARNING] {len(orphan_books)} books in ratings not found in master books")
else:
    print("[OK] All books in ratings exist in master books")

# Check 2: Text availability
text_cols = ['description', 'aggregated_reviews', 'aggregated_summaries']
for col in text_cols:
    non_empty = (master_books[col].notna() & (master_books[col].str.len() > 0)).sum()
    print(f"[INFO] Books with {col}: {non_empty}/{len(master_books)} ({non_empty/len(master_books)*100:.1f}%)")

# Check 3: Detect potential duplicates by title similarity
print("\n[*] Checking for similar titles (potential duplicates)...")
titles = master_books['title'].str.lower().str.strip()
title_counts = Counter(titles)
duplicates = {title: count for title, count in title_counts.items() if count > 1}

if duplicates:
    print(f"[WARNING] Found {len(duplicates)} potential duplicate titles:")
    for title, count in list(duplicates.items())[:5]:
        print(f"   '{title}': {count} times")
else:
    print("[OK] No duplicate titles found")

# ========================================
# SUMMARY
# ========================================

print("\n" + "="*80)
print("[COMPLETE] DATA PREPARATION FINISHED")
print("="*80)

print("\n[SUMMARY] Generated Datasets:")
print(f"  1. Master Books: {OUTPUT_BOOKS}")
print(f"     - Total books: {len(master_books)}")
print(f"     - Avg text length: {master_books['description'].str.len().mean():.0f} chars")
print(f"  2. Ratings: {OUTPUT_RATINGS}")
print(f"     - Total ratings: {len(ratings_df)}")
print(f"     - Unique users: {ratings_df['user_id'].nunique()}")
print(f"     - Unique books: {ratings_df['book_id'].nunique()}")

print("\n[NEXT] Run the following commands:")
print("  1. python scripts/01_preprocess_and_generate_embeddings.py")
print("  2. python scripts/02_test_retrieval.py")

print("\n" + "="*80)