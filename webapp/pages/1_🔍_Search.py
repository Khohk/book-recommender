"""
Search Page - Smart Book Discovery
Handles Vietnamese and English queries
Persist search state correctly
"""

import streamlit as st
import sys
import os

# =========================
# Encoding fix
# =========================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# =========================
# Import utils
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_models, load_data, get_cover_image, safe_str

# =========================
# Page config
# =========================
st.set_page_config(page_title="Search Books", page_icon="🔍", layout="wide")

# =========================
# Load CSS
# =========================
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'style.css')
if os.path.exists(css_path):
    with open(css_path, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("## 📚 Smart Book Discovery")
st.markdown("---")

# =========================
# Load models & data
# =========================
retrieval, scorer, _ = load_models()
books_df, _ = load_data()

# =========================
# Session State Init
# =========================
st.session_state.setdefault("last_query", "")
st.session_state.setdefault("last_results", [])
st.session_state.setdefault("example_query", "")

# =========================
# Search UI
# =========================
col1, col2 = st.columns([4, 1])

# Decide input value priority
input_value = (
    st.session_state.example_query
    if st.session_state.example_query
    else st.session_state.last_query
)

with col1:
    query = st.text_input(
        "Nhập mô tả sách bạn muốn tìm",
        value=input_value,
        placeholder="VD: adventure magic for children, sách lịch sử chiến tranh...",
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔍 Tìm kiếm", type="primary", use_container_width=True)

# =========================
# Example Queries
# =========================
st.markdown("**💡 Thử các ví dụ:**")
example_cols = st.columns(4)

examples = [
    ("fantasy dragons", "🐉 Fantasy Dragons"),
    ("sách lịch sử chiến tranh", "📜 Lịch sử chiến tranh"),
    ("self-help productivity", "💪 Self-help"),
    ("mystery detective", "🔍 Mystery"),
]

for i, (q_text, label) in enumerate(examples):
    with example_cols[i]:
        if st.button(label, key=f"example_{i}", use_container_width=True):
            st.session_state.example_query = q_text
            st.rerun()

# =========================
# Handle Search Action
# =========================
if search_btn:
    if not query.strip():
        st.warning("⚠️ Vui lòng nhập nội dung tìm kiếm")
    else:
        st.session_state.example_query = ""
        st.session_state.last_query = query.strip()

        with st.spinner("🔄 Đang phân tích và tìm kiếm..."):
            try:
                # Stage 1: Retrieval
                candidates = retrieval.search(
                    st.session_state.last_query,
                    top_k=50,
                    strategy="auto"
                )

                # Stage 2: Hybrid Ranking
                results = scorer.score(
                    st.session_state.last_query,
                    candidates,
                    strategy="auto"
                )

                # Persist results
                st.session_state.last_results = results

            except Exception as e:
                st.error(f"❌ Đã xảy ra lỗi: {e}")
                st.stop()

# =========================
# Render Results (STATE-DRIVEN)
# =========================
if st.session_state.last_results:
    results = st.session_state.last_results
    st.success(f"✅ Tìm thấy {len(results)} kết quả phù hợp!")

    for i, result in enumerate(results[:10], 1):
        with st.container():
            st.markdown('<div class="book-card">', unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 4, 1.5])

            # Cover
            with c1:
                try:
                    st.image(get_cover_image(result), width=100)
                except:
                    st.markdown("📚")

            # Info
            with c2:
                title = safe_str(result.get("title", "Unknown Title"), 80)
                st.subheader(f"#{i} {title}")

                st.markdown(f"**👤 Author:** {safe_str(result.get('author', 'Unknown'))}")
                st.markdown(f"**📚 Categories:** {safe_str(result.get('categories', 'N/A'))}")

                if result.get("matched_keywords"):
                    kws = list(result["matched_keywords"])[:5]
                    st.markdown("💡 **Matched:** " + " ".join(f"`{k}`" for k in kws))

                desc = safe_str(result.get("description", ""), 200)
                st.caption(f"📖 {desc}")

            # Actions
            with c3:
                st.metric("🏆 Final Score", f"{result.get('final_score', 0):.2f}")

                rating = result.get("avg_rating")
                num_ratings = result.get("num_ratings", 0)
                if rating:
                    st.markdown(f"⭐ {rating:.1f} ({num_ratings:,} ratings)")

                if st.button("📄 View Details", key=f"view_{i}"):
                    st.session_state["selected_book_id"] = result["book_id"]
                    st.session_state["selected_book_data"] = result
                    st.session_state["navigation_source"] = "search_page"
                    st.switch_page("pages/2_📖_Book_Details.py")

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
