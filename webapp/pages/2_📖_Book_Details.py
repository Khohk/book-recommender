import streamlit as st
import sys
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_models, load_data, get_cover_image

st.set_page_config(page_title="Book Details", page_icon="📖", layout="wide")

# Load CSS
with open('webapp/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Kiểm tra xem User đã chọn sách từ trang Search chưa
if 'selected_book_id' not in st.session_state:
    st.warning("⚠️ Bạn chưa chọn sách nào. Vui lòng quay lại trang Tìm kiếm.")
    if st.button("⬅️ Quay lại Tìm kiếm"):
        st.switch_page("pages/1_🔍_Search.py")
    st.stop()

# Lấy ID sách
book_id = st.session_state['selected_book_id']

# Load Data & Models
books_df, _ = load_data()
_, _, similarity_engine = load_models()

# Tìm thông tin sách trong DataFrame
book_info = books_df[books_df['book_id'] == book_id].iloc[0] if not books_df[books_df['book_id'] == book_id].empty else None

if book_info is None:
    st.error("Không tìm thấy thông tin sách này.")
    st.stop()

# --- Nút Back ---
if st.button("⬅️ Back to Search"):
    st.switch_page("pages/1_🔍_Search.py")

# --- UI Chi tiết sách ---
st.title(book_info['title'])

col1, col2 = st.columns([1, 2])

with col1:
    st.image(get_cover_image(book_info), width=250)
    st.markdown(f"### ⭐ {book_info.get('avg_rating', 'N/A')}/5")
    st.caption(f"({book_info.get('num_ratings', 0)} ratings)")

with col2:
    st.markdown(f"**Author:** {book_info.get('authors', 'Unknown')}")
    st.markdown(f"**Publisher:** {book_info.get('publisher', 'Unknown')}")
    st.markdown(f"**Categories:** {book_info.get('categories', 'N/A')}")
    
    st.markdown("### 📖 Description")
    st.write(book_info.get('description', 'No description available.'))

    st.markdown("### 💬 Reviews Summary")
    st.info(f"Summary: {book_info.get('review_summary', 'No summary available')}")

st.markdown("---")

# --- CROSS-SELL SECTION (Task 2 integration) ---
st.subheader("📚 You May Also Like (Similar Books)")

if similarity_engine:
    with st.spinner("Đang tìm sách tương tự..."):
        # Gọi hàm get_similar_items từ Person 2 Logic
        # method='hybrid' để lấy kết quả tốt nhất
        similar_books = similarity_engine.get_similar_items(
            item_id=book_id, 
            top_k=5, 
            method='hybrid' 
        )
    
    if similar_books:
        cols = st.columns(5)
        for idx, (rec_id, score) in enumerate(similar_books):
            # Lấy thông tin sách gợi ý
            rec_info = books_df[books_df['book_id'] == rec_id]
            if not rec_info.empty:
                rec_row = rec_info.iloc[0]
                with cols[idx]:
                    st.image(get_cover_image(rec_row), use_column_width=True)
                    # Cắt tên sách nếu quá dài
                    short_title = (rec_row['title'][:30] + '..') if len(rec_row['title']) > 30 else rec_row['title']
                    st.markdown(f"**{short_title}**")
                    st.caption(f"Sim: {int(score*100)}%")
                    
                    # Nút để xem chi tiết cuốn sách gợi ý này
                    if st.button("View", key=f"rec_{idx}"):
                        st.session_state['selected_book_id'] = rec_id
                        st.rerun() # Load lại trang với ID mới
    else:
        st.info("Chưa tìm thấy sách tương tự cho cuốn này.")
else:
    st.error("Engine gợi ý chưa sẵn sàng.")