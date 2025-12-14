"""
Item-Item Similarity Calculator
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemSimilarityCalculator:
    def __init__(self):
        self.content_similarity_matrix = None
        self.cf_similarity_matrix = None
        self.hybrid_similarity_matrix = None
        self.book_ids = None
        self.book_id_to_idx = None
    
    def build_content_similarity(self, embeddings, book_ids):
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = cosine_similarity(embeddings_normalized)
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix
    
    def build_cf_similarity(self, ratings_df, book_ids, min_common_users=3):
        user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='book_id', values='rating', fill_value=0
        )
        
        missing_books = set(book_ids) - set(user_item_matrix.columns)
        for book_id in missing_books:
            user_item_matrix[book_id] = 0
        
        user_item_matrix = user_item_matrix[book_ids]
        n_books = len(book_ids)
        similarity_matrix = np.zeros((n_books, n_books))
        rating_matrix = user_item_matrix.values
        
        for i in range(n_books):
            for j in range(i+1, n_books):
                mask = (rating_matrix[:, i] > 0) & (rating_matrix[:, j] > 0)
                n_common = mask.sum()
                
                if n_common >= min_common_users:
                    ratings_i = rating_matrix[mask, i]
                    ratings_j = rating_matrix[mask, j]
                    
                    if ratings_i.std() > 0 and ratings_j.std() > 0:
                        corr = np.corrcoef(ratings_i, ratings_j)[0, 1]
                        weight = min(n_common / 50, 1.0)
                        similarity = corr * weight
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
        
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix
    
    def build_hybrid_similarity(self, content_similarity, cf_similarity, 
                               content_weight=0.6, cf_weight=0.4):
        content_norm = (content_similarity - content_similarity.min()) /                        (content_similarity.max() - content_similarity.min() + 1e-10)
        
        cf_min = cf_similarity.min()
        cf_max = cf_similarity.max()
        if cf_max > cf_min:
            cf_norm = (cf_similarity - cf_min) / (cf_max - cf_min)
        else:
            cf_norm = np.zeros_like(cf_similarity)
        
        hybrid_similarity = content_weight * content_norm + cf_weight * cf_norm
        np.fill_diagonal(hybrid_similarity, 0)
        return hybrid_similarity
    
    def fit(self, embeddings, book_ids, ratings_df=None, method='hybrid',
            content_weight=0.6, cf_weight=0.4, min_common_users=3):
        self.book_ids = book_ids
        self.book_id_to_idx = {bid: idx for idx, bid in enumerate(book_ids)}
        
        self.content_similarity_matrix = self.build_content_similarity(embeddings, book_ids)
        
        if method in ['cf', 'hybrid'] and ratings_df is not None:
            self.cf_similarity_matrix = self.build_cf_similarity(
                ratings_df, book_ids, min_common_users
            )
        
        if method == 'hybrid' and self.cf_similarity_matrix is not None:
            self.hybrid_similarity_matrix = self.build_hybrid_similarity(
                self.content_similarity_matrix,
                self.cf_similarity_matrix,
                content_weight, cf_weight
            )
    
    def get_similar_items(self, book_id, top_k=10, method='hybrid'):
        if book_id not in self.book_id_to_idx:
            return []
        
        idx = self.book_id_to_idx[book_id]
        
        if method == 'content':
            sim_matrix = self.content_similarity_matrix
        elif method == 'cf':
            if self.cf_similarity_matrix is not None:
                sim_matrix = self.cf_similarity_matrix
            else:
                sim_matrix = self.content_similarity_matrix
        else:
            if self.hybrid_similarity_matrix is not None:
                sim_matrix = self.hybrid_similarity_matrix
            else:
                sim_matrix = self.content_similarity_matrix
        
        similarities = sim_matrix[idx]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = [
            (self.book_ids[i], float(similarities[i]))
            for i in top_indices if similarities[i] > 0
        ]
        return results
    
    def save(self, filepath):
        import pickle
        data = {
            'content_similarity_matrix': self.content_similarity_matrix,
            'cf_similarity_matrix': self.cf_similarity_matrix,
            'hybrid_similarity_matrix': self.hybrid_similarity_matrix,
            'book_ids': self.book_ids,
            'book_id_to_idx': self.book_id_to_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.content_similarity_matrix = data['content_similarity_matrix']
        self.cf_similarity_matrix = data['cf_similarity_matrix']
        self.hybrid_similarity_matrix = data['hybrid_similarity_matrix']
        self.book_ids = data['book_ids']
        self.book_id_to_idx = data['book_id_to_idx']