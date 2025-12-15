import streamlit as st
import sys
import os

# Thêm đường dẫn để import utils từ thư mục cha của pages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_models, load_data, get_cover_image
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

st.set_page_config(page_title="Search Books", page_icon="🔍", layout="wide")

# Load CSS
with open('webapp/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Header
st.markdown("## 📚 Smart Book Discovery")
st.markdown("---")

# Load Resources
retrieval, scorer, _ = load_models()
books_df, _ = load_data()

# 1. Search UI
col1, col2 = st.columns([4, 1])
with col1:
    raw_query = st.text_input("Nhập mô tả sách bạn muốn tìm (VD: adventure magic, sách lịch sử chiến tranh...)", "")
    query = str(raw_query).strip() 
with col2:    
    st.write("") # Spacer
    st.write("")
    search_btn = st.button("🔍 Tìm kiếm", type="primary")

# 2. Process Search
if search_btn and query:
    with st.spinner('Đang phân tích và tìm kiếm...'):
        try:
            # Stage 1: Retrieval
            candidates = retrieval.search(query, top_k=20)
            # Stage 2: Ranking
            results = scorer.score(query, candidates)
            
            st.success(f"Tìm thấy {len(results)} kết quả phù hợp!")
            
            # 3. Display Results
            for i, result in enumerate(results[:10]):
                with st.container():
                    st.markdown(f'<div class="book-card">', unsafe_allow_html=True)
                    c1, c2, c3 = st.columns([1, 4, 1.5])
                    
                    # Cột 1: Ảnh bìa
                    with c1:
                        st.image(get_cover_image(result), width=100)
                    
                    # Cột 2: Thông tin chính
                    with c2:
                        st.subheader(f"#{i+1} {result['title']}")
                        st.markdown(f"**Author:** {result.get('author', 'Unknown')}")
                        st.markdown(f"**Categories:** {result.get('categories', 'N/A')}")
                        
                        # Show matched keywords
                        if 'matched_keywords' in result and result['matched_keywords']:
                            tags = " ".join([f"`{kw}`" for kw in result['matched_keywords']])
                            st.markdown(f"💡 **Matches:** {tags}")
                            
                        # Preview description
                        desc = str(result.get('description', ''))[:200] + "..."
                        st.caption(desc)
                    
                    # Cột 3: Score & Action
                    with c3:
                        st.metric("Final Score", f"{result['final_score']:.2f}")
                        if 'avg_rating' in result:
                            st.write(f"⭐ {result['avg_rating']} ({result.get('num_ratings',0)})")
                        
                        # Button chuyển sang trang chi tiết
                        # Sử dụng session_state để lưu book_id được chọn
                        if st.button("📄 Xem chi tiết", key=f"btn_{i}"):
                            st.session_state['selected_book_id'] = result['book_id']
                            st.switch_page("pages/2_📖_Book_Details.py")
                            
                    st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")

elif search_btn and not query:
    st.warning("Vui lòng nhập nội dung tìm kiếm.")