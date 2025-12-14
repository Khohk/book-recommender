"""
Cross-Sell Recommendation Engine
"""
class CrossSellEngine:
    def __init__(self, similarity_calculator):
        self.similarity_calc = similarity_calculator
        self.books_df = None
    
    def load_book_metadata(self, books_df):
        self.books_df = books_df.set_index('book_id')
    
    def get_similar_books(self, book_id, top_k=10, method='hybrid', min_score=0.1):
        similar_items = self.similarity_calc.get_similar_items(
            book_id, top_k=top_k, method=method
        )
        
        similar_items = [
            (bid, score) for bid, score in similar_items if score >= min_score
        ]
        
        recommendations = []
        for sim_book_id, score in similar_items:
            if sim_book_id in self.books_df.index:
                book_info = self.books_df.loc[sim_book_id]
                rec = {
                    'book_id': sim_book_id,
                    'title': book_info.get('title', 'Unknown'),
                    'authors': book_info.get('authors', []),
                    'categories': book_info.get('categories', []),
                    'avg_rating': book_info.get('avg_rating', None),
                    'num_ratings': book_info.get('num_ratings', 0),
                    'similarity_score': score,
                    'similarity_method': method
                }
                recommendations.append(rec)
        return recommendations
    
    def get_multi_book_recommendations(self, book_ids, top_k=10, 
                                      method='hybrid', aggregation='max'):
        candidate_scores = {}
        
        for book_id in book_ids:
            similar = self.similarity_calc.get_similar_items(
                book_id, top_k=top_k*2, method=method
            )
            for sim_book_id, score in similar:
                if sim_book_id not in book_ids:
                    if sim_book_id not in candidate_scores:
                        candidate_scores[sim_book_id] = []
                    candidate_scores[sim_book_id].append(score)
        
        aggregated = []
        for book_id, scores in candidate_scores.items():
            if aggregation == 'max':
                final_score = max(scores)
            elif aggregation == 'mean':
                final_score = sum(scores) / len(scores)
            else:
                final_score = min(scores)
            aggregated.append((book_id, final_score))
        
        aggregated.sort(key=lambda x: x[1], reverse=True)
        top_books = aggregated[:top_k]
        
        recommendations = []
        for book_id, score in top_books:
            if book_id in self.books_df.index:
                book_info = self.books_df.loc[book_id]
                rec = {
                    'book_id': book_id,
                    'title': book_info.get('title', 'Unknown'),
                    'authors': book_info.get('authors', []),
                    'categories': book_info.get('categories', []),
                    'avg_rating': book_info.get('avg_rating', None),
                    'similarity_score': score,
                    'similarity_method': method
                }
                recommendations.append(rec)
        return recommendations