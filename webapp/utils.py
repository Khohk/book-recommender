"""
Backend Loader for Streamlit App
Handles encoding and model loading
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import io

# ============================================
# FIX ENCODING (CRITICAL!)
# ============================================
# Force UTF-8 for all I/O operations
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Set default encoding for file operations
import locale
locale.setlocale(locale.LC_ALL, '')

# Thêm đường dẫn root để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.models.hybrid_scorer import HybridScorer
from src.models.item_similarity import ItemSimilarityCalculator
from src.config import config

@st.cache_resource
def load_data():
    """Load dataframes và cache lại"""
    try:
        # Force UTF-8 when reading CSV
        books_df = pd.read_csv(config.BOOKS_PROCESSED, encoding='utf-8')
        
        try:
            ratings_path = 'data/processed/ratings.csv'
            ratings_df = pd.read_csv(ratings_path, encoding='utf-8')
        except:
            ratings_df = pd.DataFrame()
            st.warning("Ratings file not found - cross-sell may be limited")
            
        return books_df, ratings_df
    except Exception as e:
        st.error(f"Lỗi load data: {e}")
        return None, None

@st.cache_resource
def load_models():
    """Load và khởi tạo các model (Search, Scorer, Similarity)"""
    
    # 1. Init Search & Scorer
    retrieval = NLPRetrieval()
    
    # Load embeddings cho Search engine
    embeddings_path = config.BOOK_EMBEDDINGS
    
    if os.path.exists(embeddings_path):
        books_df, _ = load_data()
        retrieval.df = books_df
        
        # Load model
        from sentence_transformers import SentenceTransformer
        retrieval.model = SentenceTransformer(retrieval.model_name)
        
        # Load embeddings
        retrieval.load_embeddings(embeddings_path)
        st.success("✅ Search engine loaded")
    else:
        st.error(f"Không tìm thấy embeddings: {embeddings_path}")
        st.stop()

    scorer = HybridScorer(retrieval)

    # 2. Init Cross-sell Engine (Item Similarity)
    similarity_calc = ItemSimilarityCalculator()
    
    try:
        # Check which embeddings file to use
        if embeddings_path.endswith('.npz'):
            # Multi-field embeddings
            data = np.load(embeddings_path)
            embeddings = data['combined']  # Use combined field
        else:
            # Single embedding
            embeddings = np.load(embeddings_path)
        
        # Load book IDs
        book_ids_path = 'data/features/book_ids.pkl'
        with open(book_ids_path, 'rb') as f:
            book_ids = pickle.load(f)
        
        # Ensure matching lengths
        if len(book_ids) != embeddings.shape[0]:
            min_len = min(len(book_ids), embeddings.shape[0])
            book_ids = book_ids[:min_len]
            embeddings = embeddings[:min_len]
            st.warning(f"Adjusted to {min_len} books for similarity calculation")

        _, ratings_df = load_data()
        
        # Fit similarity calculator
        similarity_calc.fit(
            embeddings=embeddings,
            book_ids=book_ids,
            ratings_df=ratings_df,
            method='hybrid',
            content_weight=0.6,
            cf_weight=0.4,
            min_common_users=3
        )
        st.success("✅ Cross-sell engine loaded")
        
    except Exception as e:
        st.warning(f"Cross-sell engine không khả dụng: {e}")
        similarity_calc = None

    return retrieval, scorer, similarity_calc

def get_cover_image(book_row):
    """Get book cover image URL"""
    if isinstance(book_row, dict):
        book_row = pd.Series(book_row)
    
    if 'image_url' in book_row and pd.notna(book_row['image_url']):
        return book_row['image_url']
    
    # Placeholder
    return "https://via.placeholder.com/150x220.png?text=No+Cover"

def safe_str(text, max_length=None):
    """Safely convert to string and truncate if needed"""
    if pd.isna(text):
        return "N/A"
    
    text = str(text)
    
    if max_length and len(text) > max_length:
        return text[:max_length] + "..."
    
    return text

@st.cache_data
def cache_book_selection(book_id, book_data):
    """Cache selected book to prevent loss during navigation"""
    return book_id, book_data

def get_cached_book():
    """Retrieve cached book"""
    if 'cached_book_id' in st.session_state:
        return st.session_state['cached_book_id']
    return None