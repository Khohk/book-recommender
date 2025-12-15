import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from keybert import KeyBERT
from src.config import config
import os

class NLPRetrieval:
    """
    6-Field SBERT Retrieval with Category Alignment for systematic Re-ranking.
    
    Fields:
    - 1-4: Title, Description, Summary, Review (for detailed content matching)
    - 5: Combined (for high Recall in retrieval)
    - 6: Category (for Intent Alignment and systematic filtering/boosting)
    """
    
    def __init__(self, model_name=None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.keybert = None
        self.nlp = None
        self.df = None
        
        # 6 separate embeddings
        self.title_embeddings = None
        self.description_embeddings = None
        self.summary_embeddings = None
        self.review_embeddings = None
        self.combined_embeddings = None
        self.category_embeddings = None # [NEW] Dành cho Intent Alignment
    
    def fit(self, df):
        """
        Fit retrieval model on dataframe and create 6 strategic embeddings.
        """
        self.df = df.copy()
        
        # Validate required columns
        required = ['title', 'categories', 'description', 
                    'aggregated_summaries', 'aggregated_reviews']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Load SBERT model
        print(f"[*] Loading SBERT model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("[OK] SBERT model loaded")
        
        # Create multi-field embeddings
        self._create_multifield_embeddings()
        
        return self
    
    def _create_multifield_embeddings(self):
        """
        Create separate embeddings for 6 strategic fields.
        """
        print("\n[*] Creating 6-field embeddings...")
        print("    This will take time based on the number of books.")
        
        # --- 1. Title + Categories (Broad matching) ---
        print("\n    [1/6] Title + Categories...")
        title_cat_texts = []
        for _, row in self.df.iterrows():
            title = str(row.get('title', ''))
            cats = str(row.get('categories', ''))
            # Title is only used for broad text in combined, not category alone
            text = f"{title}. {cats}".strip() 
            title_cat_texts.append(text)
        
        self.title_embeddings = self.model.encode(
            title_cat_texts, show_progress_bar=True, batch_size=32
        )
        print(f"      ✓ Title/Cat Shape: {self.title_embeddings.shape}")
        
        # --- 2. Description (Detailed content) ---
        print("\n    [2/6] Descriptions...")
        descriptions = self.df['description'].fillna('').astype(str).tolist()
        self.description_embeddings = self.model.encode(
            descriptions, show_progress_bar=True, batch_size=32
        )
        print(f"      ✓ Description Shape: {self.description_embeddings.shape}")
        
        # --- 3. Aggregated Summaries (Thematic essence) ---
        print("\n    [3/6] Aggregated Summaries...")
        summaries = self.df['aggregated_summaries'].fillna('').astype(str).tolist()
        self.summary_embeddings = self.model.encode(
            summaries, show_progress_bar=True, batch_size=32
        )
        print(f"      ✓ Summary Shape: {self.summary_embeddings.shape}")
        
        # --- 4. Aggregated Reviews (User intent/language) ---
        print("\n    [4/6] Aggregated Reviews...")
        reviews = self.df['aggregated_reviews'].fillna('').astype(str).tolist()
        self.review_embeddings = self.model.encode(
            reviews, show_progress_bar=True, batch_size=32
        )
        print(f"      ✓ Review Shape: {self.review_embeddings.shape}")

        # --- 6. [NEW] Category Embeddings (Intent Alignment Anchor) ---
        print("\n    [5/6] Category Embeddings (Intent Anchor)...")
        # Chỉ vector hóa cột categories
        categories = self.df['categories'].fillna('').astype(str).tolist()
        self.category_embeddings = self.model.encode(
            categories, show_progress_bar=True, batch_size=32
        )
        print(f"      ✓ Category Shape: {self.category_embeddings.shape}")
        
        # --- 5. Create weighted combination (Cho Content Score) ---
        print("\n    [6/6] Creating weighted COMBINED embedding (Content Score)...")
        # Loại bỏ categories khỏi Title_embeddings để category_embeddings là độc lập
        # Giả sử self.title_embeddings cũ chỉ là title
        self.combined_embeddings = (
            0.15 * self.title_embeddings +        # Title (broad topic)
            0.35 * self.description_embeddings +  # Official content (high weight)
            0.15 * self.summary_embeddings +      # Thematic
            0.35 * self.review_embeddings         # User language (high weight)
        )
        # Chuẩn hóa (Normalization) embedding kết hợp
        self.combined_embeddings = self.combined_embeddings / np.linalg.norm(self.combined_embeddings, axis=1, keepdims=True)

        print(f"      ✓ Combined Shape: {self.combined_embeddings.shape}")
        
        print("\n[OK] All 6 embeddings created!\n")
    
    def search(self, query, top_k=20, strategy='auto'):
        """
        Semantic Retrieval with Systematic Category Alignment Re-ranking.
        Ignores 'strategy' for actual retrieval, always uses the best: COMBINED + CATEGORY.
        """
        print(f"\n[SEARCH] Searching: '{query}'")
        
        # 1. Encode Query
        query_embedding = self.model.encode([query])[0]
        
        # 2. Tính Content Score (Semantic Similarity) - Dùng Combined
        content_scores = cosine_similarity([query_embedding], self.combined_embeddings)[0]
        
        # 3. Tính Category Alignment Score (Context Relevance)
        category_scores = cosine_similarity([query_embedding], self.category_embeddings)[0]
        
        # 4. [SYSTEMATIC RE-RANKING] Tính Final Score
        
        # Chuẩn hóa về [0, 1] để tránh các vấn đề về phân phối score
        norm_content = (content_scores + 1) / 2
        norm_category = (category_scores + 1) / 2
        
        # Công thức Re-ranking Multiplicative (Nhân tính)
        # Ý nghĩa: Nếu Content (norm_content) cao, nó sẽ được nhân thêm điểm Category.
        # Nếu Category sai (norm_category gần 0), điểm tổng sẽ bị kéo tụt rất mạnh.
        # W_c = 0.7 (Content weight), W_a = 0.3 (Alignment weight)
        W_c = 0.7 
        W_a = 0.3
        
        final_scores = (W_c * norm_content) * (1 + (W_a * norm_category))
        
        # 5. Lấy Top K (Sử dụng Top K*3 để đảm bảo Recall tốt, sau đó cắt)
        initial_k = top_k * 3 # Ví dụ: Lấy 60 ứng viên
        top_indices = np.argsort(final_scores)[::-1][:initial_k]
        
        # 6. Xây dựng kết quả cuối cùng
        candidates = []
        for idx in top_indices:
            book_row = self.df.iloc[idx]
            candidates.append({
                'book_id': book_row['book_id'],
                'similarity_score': float(final_scores[idx]), # Điểm đã Re-rank
                'raw_content_score': float(content_scores[idx]),
                'category_alignment_score': float(category_scores[idx]),
                'book_data': book_row.to_dict()
            })
        
        # Sắp xếp và cắt lấy Top K
        candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
        final_results = candidates[:top_k]

        print(f"[OK] Found {len(final_results)} candidates after Systematic Re-ranking")
        if final_results:
            print(f"    Top score: {final_results[0]['similarity_score']:.3f}")
            print(f"    Raw Content Score (Top 1): {final_results[0]['raw_content_score']:.3f}")
            print(f"    Category Alignment Score (Top 1): {final_results[0]['category_alignment_score']:.3f}")
        
        return final_results
    
    # [Các hàm khác không thay đổi logic cốt lõi của Re-ranking, giữ nguyên hoặc tối giản]
    def extract_keywords(self, text, top_n=5):

        """Extract keywords using KeyBERT"""

        if self.keybert is None:

            print("[*] Loading KeyBERT model...")

            self.keybert = KeyBERT()

            print("[OK] KeyBERT model loaded")

       

        try:

            keywords = self.keybert.extract_keywords(

                text,

                keyphrase_ngram_range=(1, 2),

                stop_words='english',

                top_n=top_n

            )

            return [kw[0] for kw in keywords]

        except:

            return []

   

    def extract_entities(self, text):

        """Extract named entities using spaCy"""

        if self.nlp is None:

            print("[*] Loading Spacy model: en_core_web_sm")

            self.nlp = spacy.load('en_core_web_sm')

            print("[OK] Spacy model loaded")

       

        doc = self.nlp(text[:1000])  # Limit length

        entities = {}

       

        for ent in doc.ents:

            if ent.label_ not in entities:

                entities[ent.label_] = []

            entities[ent.label_].append(ent.text)

       

        return entities
    def _get_field_embeddings(self, strategy):
        """Get embeddings for specified strategy (Dùng cho debug/test đơn lẻ)"""
        field_map = {
            'title': self.title_embeddings,
            'description': self.description_embeddings,
            'summary': self.summary_embeddings,
            'review': self.review_embeddings,
            'combined': self.combined_embeddings
        }
        return field_map.get(strategy, self.combined_embeddings)
    
    def _select_strategy(self, query):
        """
        Auto-select strategy now mainly serves as a hint, 
        as search() uses combined + category alignment for best results.
        """
        query_lower = query.lower()
        query_len = len(query.split())
        
        user_lang_keywords = ['review', 'recommend', 'like', 'love', 'best', 'favorite']
        if any(kw in query_lower for kw in user_lang_keywords):
            return 'review'
        
        genre_keywords = ['fiction', 'mystery', 'thriller', 'romance', 'fantasy', 'history']
        if query_len <= 4 and any(kw in query_lower for kw in genre_keywords):
            return 'title'
        
        if query_len <= 4:
            return 'title'
        elif query_len <= 10:
            return 'combined'
        else:
            return 'description'

    def save_embeddings(self, filepath):
        """Save all embeddings to disk, including the new category field"""
        print(f"\n[*] Saving embeddings to: {filepath}")
        
        np.savez_compressed(
            filepath,
            title=self.title_embeddings,
            description=self.description_embeddings,
            summary=self.summary_embeddings,
            review=self.review_embeddings,
            combined=self.combined_embeddings,
            category=self.category_embeddings # [NEW]
        )
        
        print("[OK] Embeddings saved")
    
    def load_embeddings(self, filepath):
        """Load pre-computed embeddings from disk, including the new category field"""
        print(f"\n[*] Loading embeddings from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        data = np.load(filepath)
        
        self.title_embeddings = data['title']
        self.description_embeddings = data['description']
        self.summary_embeddings = data['summary']
        self.review_embeddings = data['review']
        self.combined_embeddings = data['combined']
        self.category_embeddings = data['category'] # [NEW]
        
        print(f"[OK] Loaded 6-field embeddings")
        print(f"    Title: {self.title_embeddings.shape}")
        print(f"    Description: {self.description_embeddings.shape}")
        print(f"    Summary: {self.summary_embeddings.shape}")
        print(f"    Review: {self.review_embeddings.shape}")
        print(f"    Combined: {self.combined_embeddings.shape}")
        print(f"    Category: {self.category_embeddings.shape}")


if __name__ == "__main__":
    # Test
    from src.preprocessing.data_loader import DataLoader
    
    # Tải dữ liệu (giả sử DataLoader đã được định nghĩa và hoạt động đúng)
    loader = DataLoader()
    df = loader.load_master_books()
    
    retrieval = NLPRetrieval()
    retrieval.fit(df) # Sẽ tạo 6 trường embeddings mới
    
    # Test queries
    test_queries = [
        "adventure book with magic for children", 
        "best fantasy books", 
        "science fiction",
        "romantic story about war in 19th century" # Test thêm 1 query phức tạp
    ]
    
    print("\n" + "="*80)
    print("TESTING SYSTEMATIC CATEGORY ALIGNMENT RE-RANKING")
    print("="*80)
    
    for query in test_queries:
        # Gọi search mà không cần strategy='auto' vì hàm search đã tự động Re-rank
        results = retrieval.search(query, top_k=5, strategy='combined')
        
        print(f"\n[QUERY] '{query}'")
        print("--- Top 5 Results (Re-ranked Score) ---")
        
        for i, r in enumerate(results, 1):
            title = r['book_data']['title'][:50]
            category = r['book_data']['categories'][:30]
            print(f"  {i}. {title}")
            print(f"      Score: {r['similarity_score']:.3f} (Raw Content: {r['raw_content_score']:.3f}, Cat Match: {r['category_alignment_score']:.3f})")
            print(f"      Category: {category}")
        print("---------------------------------------")

    retrieval.save_embeddings('data/features/book_embeddings_5field.npz')