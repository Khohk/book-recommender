"""
Configuration file for Book Recommendation System
Centralized settings for paths, models, and hyperparameters
"""

import os

class Config:
    # ========================================
    # PATHS
    # ========================================
    
    # Base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directories
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    FEATURES_DIR = os.path.join(DATA_DIR, "features")
    
    # Raw data files (YOUR DATA - PARQUET)
    RAW_REVIEWS_FILE = os.path.join(RAW_DATA_DIR, "reviews.parquet")
    RAW_BOOKS_FILE = os.path.join(RAW_DATA_DIR, "books_details.parquet")
    
    # Processed data files
    MASTER_BOOKS = os.path.join(PROCESSED_DATA_DIR, "master_books.csv")
    RATINGS = os.path.join(PROCESSED_DATA_DIR, "ratings.csv")
    BOOKS_PROCESSED = os.path.join(PROCESSED_DATA_DIR, "books_with_combined_text.csv")
    RATINGS_MATRIX = os.path.join(PROCESSED_DATA_DIR, "ratings_matrix.npz")
    
    # Feature files (precomputed)
    BOOK_EMBEDDINGS = os.path.join(FEATURES_DIR, "book_embeddings_5field.npz")
    BOOK_EMBEDDINGS_2 = os.path.join(FEATURES_DIR, "book_embeddings.npy")
    BOOK_IDS = os.path.join(FEATURES_DIR, "book_ids.pkl")
    BOOK_KEYWORDS = os.path.join(FEATURES_DIR, "book_keywords.pkl")
    BOOK_ENTITIES = os.path.join(FEATURES_DIR, "book_entities.pkl")
    IMPACT_SCORES = os.path.join(FEATURES_DIR, "book_impact_scores.csv")
    
    # ========================================
    # COLUMN MAPPING (YOUR DATA STRUCTURE)
    # ========================================
    
    COLUMN_MAPPING = {
        'book_id': 'Id',
        'title': 'Title',
        'user_id': 'User_id',
        'rating': 'review/score',
        'review_text': 'review/text',
        'review_summary': 'review/summary',
        'description': 'description',
        'authors': 'authors',
        'publisher': 'publisher',
        'published_date': 'publishedDate',
        'categories': 'categories',
        'ratings_count': 'ratingsCount'
    }
    
    # Columns needed for combined text
    TEXT_COLUMNS = ['title', 'description', 'review_text', 'review_summary', 'categories']
    
    # ========================================
    # MODEL CONFIGURATIONS
    # ========================================
    
    # NLP Model
    EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'  # Supports Vietnamese
    EMBEDDING_DIM = 384  # Dimension of embeddings
    EMBEDDING_BATCH_SIZE = 16
    
    # Keyword Extraction
    KEYWORD_TOP_N = 5
    KEYWORD_NGRAM_RANGE = (1, 2)
    
    # NER Model
    SPACY_MODEL = "en_core_web_sm"
    ENTITY_TYPES = ['PERSON', 'GPE', 'ORG', 'WORK_OF_ART']
    
    # ========================================
    # RETRIEVAL PARAMETERS
    # ========================================
    
    # Stage 1: Candidate Generation
    N_CANDIDATES = 100  # Retrieve top-100 candidates
    
    # Stage 2: Re-ranking
    TOP_K_RESULTS = 10  # Return top-10 to user
    
    # Hybrid weights for scoring
    WEIGHTS_NLP_ONLY = {
        'similarity': 0.70,   # Text similarity (SBERT)
        'keywords': 0.20,     # Keyword overlap
        'entities': 0.10,     # Entity matching
    }
    
    WEIGHTS_COLD_START = {
        'similarity': 0.80,   # Text similarity
        'impact': 0.20,       # Popularity/quality
    }
    
    WEIGHTS_PERSONALIZED = {
        'similarity': 0.50,   # Text similarity
        'cf_score': 0.30,     # Collaborative filtering
        'impact': 0.20,       # Popularity/quality
    }
    
    # ========================================
    # TEXT PREPROCESSING
    # ========================================
    
    # Text cleaning
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 5000
    
    # Title repetition (for weight)
    TITLE_REPEAT_TIMES = 2
    
    # Review aggregation
    MAX_REVIEWS_PER_BOOK = 10
    
    # ========================================
    # DISPLAY SETTINGS
    # ========================================
    
    DESCRIPTION_PREVIEW_LENGTH = 200
    
    # ========================================
    # LOGGING
    # ========================================
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Singleton instance
config = Config()