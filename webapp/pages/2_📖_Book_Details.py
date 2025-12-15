"""
Book Details Page with Cross-sell Recommendations
FIXED VERSION - No syntax errors
"""

import streamlit as st
import sys
import os
import pandas as pd

# Fix encoding
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except:
    pass

# Import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from utils import load_models, load_data, get_cover_image, safe_str
except ImportError as e:
    st.error(f"Cannot import utils: {e}")
    st.stop()

st.set_page_config(page_title="Book Details", page_icon="📖", layout="wide")

# Load CSS
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'style.css')
if os.path.exists(css_path):
    try:
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

# ============================================
# DEBUG SECTION
# ============================================
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.write("Session State Keys:", list(st.session_state.keys()))

if 'selected_book_id' in st.session_state:
    st.sidebar.success(f"Book ID: {st.session_state['selected_book_id']}")
else:
    st.sidebar.error("No book ID found!")

# ============================================
# GET BOOK ID
# ============================================
book_id = None

if 'selected_book_id' in st.session_state:
    book_id = st.session_state['selected_book_id']
    st.sidebar.info("Source: session_state")

if not book_id:
    st.error("❌ No book selected")
    st.warning("⚠️ Please go back to Search page and select a book")
    
    if st.button("⬅️ Back to Search"):
        st.switch_page("pages/1_🔍_Search.py")
    
    st.stop()

# ============================================
# LOAD DATA
# ============================================
try:
    books_df, _ = load_data()
    _, _, similarity_engine = load_models()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ============================================
# FIND BOOK INFO
# ============================================
book_info = books_df[books_df['book_id'] == book_id]

if book_info.empty:
    st.error(f"❌ Book not found: {book_id}")
    
    if st.button("⬅️ Back to Search"):
        st.switch_page("pages/1_🔍_Search.py")
    
    st.stop()

book_info = book_info.iloc[0]

# ============================================
# DISPLAY BOOK DETAILS
# ============================================

# Back button
if st.button("⬅️ Back to Search"):
    st.switch_page("pages/1_🔍_Search.py")

st.markdown("---")

# Title
title = safe_str(book_info.get('title', 'Unknown Title'))
st.title(title)

# Two columns layout
col1, col2 = st.columns([1, 2])

# LEFT COLUMN - Cover and metadata
with col1:
    # Cover image
    try:
        cover_url = get_cover_image(book_info)
        st.image(cover_url, width=250)
    except:
        st.markdown("## 📚")
    
    # Rating
    rating = book_info.get('avg_rating')
    num_ratings = book_info.get('num_ratings', 0)
    
    if rating and not pd.isna(rating):
        st.markdown(f"### ⭐ {float(rating):.1f}/5")
        st.caption(f"({int(num_ratings):,} ratings)")
    else:
        st.markdown("### ⭐ N/A")
    
    # Metadata
    st.markdown("---")
    
    author = safe_str(book_info.get('authors', 'Unknown'))
    st.markdown(f"**👤 Author:** {author}")
    
    categories = safe_str(book_info.get('categories', 'N/A'))
    st.markdown(f"**📚 Categories:** {categories}")
    
    publisher = safe_str(book_info.get('publisher', 'N/A'))
    st.markdown(f"**🏢 Publisher:** {publisher}")

# RIGHT COLUMN - Description
with col2:
    st.markdown("### 📖 Description")
    
    desc = book_info.get('description')
    if desc and not pd.isna(desc):
        st.write(safe_str(desc))
    else:
        st.info("No description available")
    
    # Review summary
# Review summary
if 'aggregated_summaries' in book_info.index:
    summary = book_info.get('aggregated_summaries')
    if summary and not pd.isna(summary):
        st.markdown("### 💬 Review Summary")
        st.info(safe_str(summary, max_length=300))


st.markdown("---")

# ============================================
# CROSS-SELL SECTION
# ============================================
st.markdown("## 📚 You May Also Like")

if similarity_engine is None:
    st.warning("⚠️ Recommendation engine not available")
    st.stop()

try:
    with st.spinner("🔄 Finding similar books..."):
        similar_books = similarity_engine.get_similar_items(
            book_id,
            top_k=5,
            method='hybrid'
        )
    
    if not similar_books or len(similar_books) == 0:
        st.info("💡 No similar books found")
    else:
        # Display in 3x2 grid
        cols = st.columns(3)
        
        for idx, (rec_id, score) in enumerate(similar_books):
            rec_info = books_df[books_df['book_id'] == rec_id]
            
            if not rec_info.empty:
                rec_row = rec_info.iloc[0]
                
                with cols[idx % 3]:
                    with st.container():
                        # Cover
                        try:
                            cover = get_cover_image(rec_row)
                            st.image(cover, use_container_width=True)
                        except:
                            st.markdown("### 📚")
                        
                        # Title
                        rec_title = safe_str(rec_row.get('title', 'Unknown'), max_length=40)
                        st.markdown(f"**{rec_title}**")
                        
                        # Author
                        rec_author = safe_str(rec_row.get('authors', 'Unknown'), max_length=30)
                        st.caption(f"👤 {rec_author}")
                        
                        # Match score
                        st.markdown(f"🎯 **{score*100:.0f}%** match")
                        
                        # Rating
                        rec_rating = rec_row.get('avg_rating')
                        if rec_rating and not pd.isna(rec_rating):
                            st.caption(f"⭐ {float(rec_rating):.1f}")
                        
                        # View button
                        button_key = f"rec_view_{idx}_{rec_id}"
                        if st.button("📖 View", key=button_key, use_container_width=True):
                            st.session_state['selected_book_id'] = rec_id
                            st.rerun()

except Exception as e:
    st.error(f"❌ Error finding similar books: {e}")
    import traceback
    with st.expander("Show error details"):
        st.code(traceback.format_exc())