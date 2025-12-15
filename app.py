"""
Smart Book Discovery - Streamlit Web App
Main entry point with search, details, and recommendations
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.models.hybrid_scorer import HybridScorer
from src.config import config

# Page config
st.set_page_config(
    page_title="📚 Smart Book Discovery",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Book card styling */
    .book-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    
    .book-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .book-author {
        font-size: 1rem;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .book-rating {
        font-size: 1.2rem;
        color: #ffd700;
        margin-bottom: 1rem;
    }
    
    /* Score bars */
    .score-container {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    
    .score-label {
        color: white;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .matched-keywords {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        color: white;
        font-size: 0.9rem;
    }
    
    /* Search box */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        font-size: 1.1rem;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'retrieval' not in st.session_state:
    st.session_state.retrieval = None
if 'scorer' not in st.session_state:
    st.session_state.scorer = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_book' not in st.session_state:
    st.session_state.selected_book = None

@st.cache_resource
def load_models():
    """Load models once and cache"""
    with st.spinner("🔄 Loading AI models... (This takes ~30 seconds first time)"):
        # Load data
        df = pd.read_csv(config.BOOKS_PROCESSED)
        
        # Initialize retrieval
        retrieval = NLPRetrieval()
        retrieval.df = df
        
        # Load embeddings
        from sentence_transformers import SentenceTransformer
        retrieval.model = SentenceTransformer(retrieval.model_name)
        retrieval.load_embeddings(config.BOOK_EMBEDDINGS)
        
        # Initialize scorer
        scorer = HybridScorer(retrieval)
        
        return retrieval, scorer, df

def display_score_bar(label, score, color):
    """Display a visual score bar"""
    percentage = int(score * 100)
    bar_length = int(score * 10)
    bar = "█" * bar_length + "░" * (10 - bar_length)
    
    st.markdown(f"""
    <div class="score-label">
        {label}: <strong>{bar}</strong> {percentage}%
    </div>
    """, unsafe_allow_html=True)

def display_book_card(book, rank):
    """Display a book result card - matches 03_test_hybrid.py format"""
    
    with st.container():
        st.markdown(f"""
        <div class="book-card">
            <div class="book-title">#{rank} {book['title']}</div>
            <div class="book-author">👤 Author: {book.get('author', 'Unknown')}</div>
            <div class="book-author">📚 Categories: {book.get('categories', 'N/A')}</div>
            <div class="book-rating">⭐ Rating: {book.get('avg_rating', 'N/A')} ({book.get('num_ratings', 0)} ratings)</div>
        """, unsafe_allow_html=True)
        
        # Final Score (prominent display)
        final_score = book.get('final_score', 0)
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <div style='color: white; font-size: 1.1rem;'>
                <strong>🏆 Final Score: {final_score:.4f}</strong> ({final_score*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Matched keywords
        if book.get('matched_keywords'):
            keywords_list = list(book['matched_keywords'])[:5]
            keywords = ', '.join(keywords_list)
            st.markdown(f"""
            <div class="matched-keywords">
                💡 Matched Keywords: {keywords}
            </div>
            """, unsafe_allow_html=True)
        
        # Description
        desc = str(book.get('description', 'No description available'))[:200]
        st.markdown(f"<div style='color: white; margin-top: 1rem;'>📖 {desc}...</div>", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(f"📖 View Details", key=f"details_{rank}"):
                st.session_state.selected_book = book
                st.rerun()
        
        with col2:
            if st.button(f"🔗 Similar Books", key=f"similar_{rank}"):
                st.session_state.selected_book = book
                st.session_state.show_similar = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>📚 Smart Book Discovery</h1>
        <p style='font-size: 1.2rem; color: #666;'>
            AI-powered book search using advanced NLP & semantic understanding
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if st.session_state.retrieval is None:
        retrieval, scorer, df = load_models()
        st.session_state.retrieval = retrieval
        st.session_state.scorer = scorer
        st.session_state.df = df
        st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        top_k = st.slider("Number of results", 5, 20, 10)
        
        st.markdown("### 📊 Model Weights")
        st.markdown(f"""
        - **Semantic**: 50%
        - **Keywords**: 20%
        - **Category**: 15%
        - **Entities**: 5%
        - **Impact**: 10%
        """)
        
        st.markdown("### 📈 System Stats")
        st.metric("Total Books", f"{len(st.session_state.df):,}")
        st.metric("Categories", st.session_state.df['categories'].nunique())
        
        if st.button("🔄 Reset Search"):
            st.session_state.search_results = None
            st.session_state.selected_book = None
            st.rerun()
    
    # Main content
    if st.session_state.selected_book is not None:
        # Show book details
        show_book_details()
    else:
        # Show search interface
        show_search_interface(top_k)

def show_search_interface(top_k):
    """Display search interface"""
    
    # Search box
    st.markdown("### 🔍 What are you looking for?")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "Describe the book you want (e.g., 'adventure book with magic for children')",
            height=100,
            placeholder="Enter your search query here..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Example queries
    st.markdown("**💡 Try these examples:**")
    example_cols = st.columns(4)
    
    examples = [
        "fantasy novel with dragons",
        "self-help about productivity",
        "mystery thriller detective",
        "biography of scientists"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                query = example
                search_button = True
    
    # Perform search
    if search_button and query:
        with st.spinner("🔎 Searching through 2700+ books..."):
            # Get candidates
            candidates = st.session_state.retrieval.search(query, top_k=50)
            
            # Hybrid scoring
            results = st.session_state.scorer.score(query, candidates)
            
            st.session_state.search_results = results[:top_k]
        
        st.success(f"✅ Found {len(st.session_state.search_results)} relevant books!")
    
    # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown(f"### 📚 Top {len(st.session_state.search_results)} Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = sum(r['final_score'] for r in st.session_state.search_results) / len(st.session_state.search_results)
            st.markdown(f"""
            <div class="metric-container">
                <h3>Avg Score</h3>
                <h2>{avg_score:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            top_score = st.session_state.search_results[0]['final_score']
            st.markdown(f"""
            <div class="metric-container">
                <h3>Top Score</h3>
                <h2>{top_score:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_rating = sum(float(r.get('avg_rating', 0) or 0) for r in st.session_state.search_results) / len(st.session_state.search_results)
            st.markdown(f"""
            <div class="metric-container">
                <h3>Avg Rating</h3>
                <h2>{avg_rating:.1f} ⭐</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display book cards
        for rank, book in enumerate(st.session_state.search_results, 1):
            display_book_card(book, rank)

def show_book_details():
    """Display detailed book view with similar recommendations"""
    
    book = st.session_state.selected_book
    
    # Back button
    if st.button("← Back to Search"):
        st.session_state.selected_book = None
        st.rerun()
    
    # Book details
    st.markdown(f"# 📖 {book['title']}")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; color: white;'>
            <h3>📖 Book Information</h3>
            <p><strong>👤 Author:</strong> {book.get('author', 'Unknown')}</p>
            <p><strong>📚 Categories:</strong> {book.get('categories', 'N/A')}</p>
            <p><strong>⭐ Rating:</strong> {book.get('avg_rating', 'N/A')} / 5</p>
            <p><strong>👥 Reviews:</strong> {book.get('num_ratings', 0):,}</p>
            <p><strong>🏆 Final Score:</strong> {book.get('final_score', 0):.4f} ({book.get('final_score', 0)*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Matched keywords
        if book.get('matched_keywords'):
            st.markdown("### 💡 Matched Keywords")
            keywords = list(book['matched_keywords'])[:10]
            st.markdown(" • ".join(f"`{kw}`" for kw in keywords))
    
    with col2:
        st.markdown("### 📝 Description")
        desc = book.get('description', 'No description available')
        st.write(desc if pd.notna(desc) else 'No description available')
    
    # Similar books section
    st.markdown("---")
    st.markdown("## 📚 You May Also Like")
    
    with st.spinner("Finding similar books..."):
        # Use SBERT similarity for recommendations
        book_id = book['book_id']
        book_idx = st.session_state.df[st.session_state.df['book_id'] == book_id].index[0]
        
        # Get similar books using embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        book_embedding = st.session_state.retrieval.combined_embeddings[book_idx]
        similarities = cosine_similarity(
            [book_embedding],
            st.session_state.retrieval.combined_embeddings
        )[0]
        
        # Get top 6 similar (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:7]
        
        similar_books = []
        for idx in similar_indices:
            similar_row = st.session_state.df.iloc[idx]
            similar_books.append({
                'book_id': similar_row['book_id'],
                'title': similar_row['title'],
                'author': similar_row.get('authors', 'Unknown'),
                'avg_rating': similar_row.get('avg_rating', 'N/A'),
                'categories': similar_row.get('categories', 'N/A'),
                'similarity': float(similarities[idx])
            })
    
    # Display similar books in grid
    cols = st.columns(3)
    
    for i, similar in enumerate(similar_books):
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;'>
                <h4>{similar['title'][:40]}...</h4>
                <p>👤 {similar['author']}</p>
                <p>⭐ {similar['avg_rating']}</p>
                <p>🎯 {similar['similarity']:.0%} match</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("View", key=f"view_similar_{i}"):
                # Find full book data
                full_book = st.session_state.df[st.session_state.df['book_id'] == similar['book_id']].iloc[0]
                st.session_state.selected_book = full_book.to_dict()
                st.rerun()

if __name__ == "__main__":
    main()