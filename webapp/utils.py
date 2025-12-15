import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

# Thêm đường dẫn root để import được src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.models.hybrid_scorer import HybridScorer
from src.models.item_similarity import ItemSimilarityCalculator
from src.config import config

@st.cache_resource
def load_data():
    """Load dataframes và cache lại để không phải đọc đĩa nhiều lần"""
    try:
        books_df = pd.read_csv(config.BOOKS_PROCESSED)
        # Load ratings nếu cần cho CF, nếu không có thì bỏ qua hoặc load file rating
        try:
            ratings_df = pd.read_csv(config.RATINGS)
        except:
            ratings_df = pd.DataFrame() # Fallback nếu chưa có rating
            
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
    if os.path.exists(config.BOOK_EMBEDDINGS):
        # Trick: Load dataframe vào retrieval trước
        books_df, _ = load_data()
        retrieval.df = books_df
        # Load pre-computed embeddings
        retrieval.load_embeddings(config.BOOK_EMBEDDINGS)
    else:
        st.error("Không tìm thấy file embeddings cho Search!")

    scorer = HybridScorer(retrieval)

    # 2. Init Cross-sell Engine (Item Similarity)
    similarity_calc = ItemSimilarityCalculator()
    
    # Load các dữ liệu feature riêng cho Similarity
    try:
        embeddings = np.load(config.BOOK_EMBEDDINGS_2)
        with open(config.BOOK_IDS, 'rb') as f:
            book_ids = pickle.load(f)
        
        # Đảm bảo kích thước khớp nhau (fix bug tiềm ẩn)
        if len(book_ids) != embeddings.shape[0]:
            min_len = min(len(book_ids), embeddings.shape[0])
            book_ids = book_ids[:min_len]
            embeddings = embeddings[:min_len]

        _, ratings_df = load_data()
        
        # Fit model (tính toán/cache ma trận)
        similarity_calc.fit(
            embeddings=embeddings,
            book_ids=book_ids,
            ratings_df=ratings_df,
            method='hybrid', # Mặc định dùng hybrid
            content_weight=0.6,
            cf_weight=0.4
        )
    except Exception as e:
        st.warning(f"Không thể khởi tạo Cross-sell engine: {e}")
        similarity_calc = None

    return retrieval, scorer, similarity_calc

# Helper function để lấy ảnh bìa (giả lập nếu không có cột image_url)
def get_cover_image(book_row):
    # Nếu data của bạn có cột 'image_url'
    if 'image_url' in book_row and pd.notna(book_row['image_url']):
        return book_row['image_url']
    # Placeholder image
    return "https://via.placeholder.com/150x220.png?text=No+Cover"