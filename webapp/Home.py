import streamlit as st
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Book Recommender System")

st.markdown("""
### Chào mừng bạn đến với hệ thống gợi ý sách!

Hệ thống cung cấp 2 tính năng chính:
1. **🔍 Smart Search:** Tìm sách theo mô tả ngữ nghĩa (Natural Language Search).
2. **🧩 Similarity & Cross-sell:** Xem chi tiết sách và nhận gợi ý các cuốn tương tự.

👈 **Hãy chọn 'Search' từ thanh menu bên trái để bắt đầu.**
""")

# Load css
with open('webapp/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)