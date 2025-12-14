"""
Hybrid Scoring Engine
Combines: SBERT Similarity + Keywords + Entity Matching
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

class HybridScorer:
    """
    Combine multiple NLP signals to score book relevance
    
    Scoring Formula:
    final_score = α × similarity + β × keywords + γ × entities
    """
    
    def __init__(self, nlp_retrieval):
        """
        Args:
            nlp_retrieval: NLPRetrieval instance (already fitted)
        """
        self.retrieval = nlp_retrieval
        self.df = nlp_retrieval.df
        
        # Default weights
        self.weights = config.WEIGHTS_NLP_ONLY
    
    def score(self, query, candidates, weights=None):
        """
        Score candidate books using hybrid approach
        
        Args:
            query: User query string
            candidates: List of dicts from retrieval.search()
            weights: Optional custom weights dict
            
        Returns:
            List of scored candidates (sorted by final_score)
        """
        if weights is None:
            weights = self.weights
        
        print(f"\n[HYBRID] Scoring {len(candidates)} candidates...")
        print(f"   Weights: {weights}")
        
        # Extract candidate book_ids and similarity scores
        candidate_ids = [c['book_id'] for c in candidates]
        sim_scores = {c['book_id']: c['similarity_score'] for c in candidates}
        
        # 1. Extract query features
        query_keywords = set(self.retrieval.extract_keywords(query))
        query_entities = self.retrieval.extract_entities(query)
        
        print(f"   Query keywords: {list(query_keywords)[:5]}")
        print(f"   Query entities: {self._format_entities(query_entities)}")
        
        # 2. Calculate keyword scores
        keyword_scores = self._calculate_keyword_scores(
            query_keywords, candidate_ids
        )
        
        # 3. Calculate entity scores
        entity_scores = self._calculate_entity_scores(
            query_entities, candidate_ids
        )
        
        # 4. Normalize scores to [0, 1]
        # IMPORTANT: Don't normalize similarity (already in [0, 1] from SBERT)
        # Only normalize keywords and entities
        kw_norm = self._normalize_scores(keyword_scores, candidate_ids)
        ent_norm = self._normalize_scores(entity_scores, candidate_ids)
        
        # 5. Combine scores
        final_scores = {}
        for book_id in candidate_ids:
            score = (
                weights.get('similarity', 0.7) * sim_scores[book_id] +  # Use original sim score
                weights.get('keywords', 0.2) * kw_norm[book_id] +
                weights.get('entities', 0.1) * ent_norm[book_id]
            )
            final_scores[book_id] = score
        
        # 6. Update candidates with hybrid scores
        scored_candidates = []
        for candidate in candidates:
            book_id = candidate['book_id']
            book_data = candidate['book_data']
            
            scored_candidates.append({
                'book_id': book_id,
                'title': book_data['title'],
                'author': book_data.get('authors', 'Unknown'),
                'categories': book_data.get('categories', 'N/A'),
                'avg_rating': book_data.get('avg_rating', 'N/A'),
                'description': book_data.get('description', ''),
                'final_score': final_scores[book_id],
                'scores_breakdown': {
                    'similarity': sim_scores[book_id],  # Original similarity score
                    'keywords': kw_norm[book_id],
                    'entities': ent_norm[book_id]
                },
                'matched_keywords': self._get_matched_keywords(
                    query_keywords, book_data
                ),
                'matched_entities': self._get_matched_entities(
                    query_entities, book_data
                )
            })
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"[OK] Hybrid scoring complete")
        print(f"   Top score: {scored_candidates[0]['final_score']:.3f}")
        print(f"   Lowest score: {scored_candidates[-1]['final_score']:.3f}")
        
        return scored_candidates
    
    def _calculate_keyword_scores(self, query_keywords, candidate_ids):
        """
        Calculate keyword overlap scores
        """
        scores = {}
        
        for book_id in candidate_ids:
            # Get book data
            book_row = self.df[self.df['book_id'] == book_id]
            
            if len(book_row) == 0:
                scores[book_id] = 0.0
                continue
            
            book_row = book_row.iloc[0]
            
            # Extract book keywords from combined_text
            book_text = book_row.get('combined_text', '')
            if pd.isna(book_text) or len(str(book_text)) < 10:
                scores[book_id] = 0.0
                continue
            
            book_keywords = set(self.retrieval.extract_keywords(str(book_text)))
            
            # Calculate overlap
            if len(query_keywords) == 0:
                scores[book_id] = 0.0
            else:
                overlap = len(query_keywords & book_keywords)
                scores[book_id] = overlap / len(query_keywords)
        
        return scores
    
    def _calculate_entity_scores(self, query_entities, candidate_ids):
        """
        Calculate entity matching scores
        """
        scores = {}
        
        for book_id in candidate_ids:
            book_row = self.df[self.df['book_id'] == book_id]
            
            if len(book_row) == 0:
                scores[book_id] = 0.0
                continue
            
            book_row = book_row.iloc[0]
            book_text = book_row.get('combined_text', '')
            
            if pd.isna(book_text) or len(str(book_text)) < 10:
                scores[book_id] = 0.0
                continue
            
            book_entities = self.retrieval.extract_entities(str(book_text))
            
            # Calculate entity overlap
            score = 0.0
            total = 0
            
            for ent_type in ['PERSON', 'GPE', 'ORG', 'WORK_OF_ART']:
                q_ents = set(query_entities.get(ent_type, []))
                b_ents = set(book_entities.get(ent_type, []))
                
                if q_ents:
                    matches = len(q_ents & b_ents)
                    score += matches / len(q_ents)
                    total += 1
            
            scores[book_id] = score / total if total > 0 else 0.0
        
        return scores
    
    def _normalize_scores(self, scores, candidate_ids):
        """
        Normalize scores to [0, 1] range
        """
        values = [scores[book_id] for book_id in candidate_ids]
        
        if len(values) == 0:
            return {book_id: 0.0 for book_id in candidate_ids}
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {book_id: 1.0 for book_id in candidate_ids}
        
        normalized = {}
        for book_id in candidate_ids:
            normalized[book_id] = (scores[book_id] - min_val) / (max_val - min_val)
        
        return normalized
    
    def _get_matched_keywords(self, query_keywords, book_data):
        """
        Get keywords that match between query and book
        """
        book_text = book_data.get('combined_text', '')
        if pd.isna(book_text) or len(str(book_text)) < 10:
            return []
        
        book_keywords = set(self.retrieval.extract_keywords(str(book_text)))
        matched = list(query_keywords & book_keywords)
        
        return matched[:5]  # Return top 5
    
    def _get_matched_entities(self, query_entities, book_data):
        """
        Get entities that match between query and book
        """
        book_text = book_data.get('combined_text', '')
        if pd.isna(book_text) or len(str(book_text)) < 10:
            return {}
        
        book_entities = self.retrieval.extract_entities(str(book_text))
        
        matched = {}
        for ent_type in ['PERSON', 'GPE', 'ORG']:
            q_ents = set(query_entities.get(ent_type, []))
            b_ents = set(book_entities.get(ent_type, []))
            overlap = list(q_ents & b_ents)
            if overlap:
                matched[ent_type] = overlap[:3]  # Top 3 per type
        
        return matched
    
    def _format_entities(self, entities):
        """
        Format entities dict for display
        """
        formatted = []
        for ent_type, ent_list in entities.items():
            if ent_list:
                formatted.append(f"{ent_type}={ent_list[:2]}")
        return ", ".join(formatted) if formatted else "None"


if __name__ == "__main__":
    # Test
    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.text_cleaner import TextPreprocessor
    from src.models.nlp_retrieval import NLPRetrieval
    
    # Load data
    loader = DataLoader()
    df = loader.load_master_books()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df = preprocessor.process(df)
    
    # Load retrieval model
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    
    # Get candidates
    query = "adventure book with magic for children"
    candidates = retrieval.search(query, top_k=20)
    
    # Hybrid scoring
    scorer = HybridScorer(retrieval)
    scored = scorer.score(query, candidates)
    
    # Display top 5
    print("\n[TOP 5 HYBRID RESULTS]")
    for i, book in enumerate(scored[:5], 1):
        print(f"\n{i}. {book['title']}")
        print(f"   Final: {book['final_score']:.3f}")
        print(f"   Breakdown: Sim={book['scores_breakdown']['similarity']:.2f}, "
              f"KW={book['scores_breakdown']['keywords']:.2f}, "
              f"Ent={book['scores_breakdown']['entities']:.2f}")
        print(f"   Matched KW: {book['matched_keywords']}")