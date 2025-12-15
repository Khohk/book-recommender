"""
Hybrid Scoring Engine - Correct Architecture
Based on: Building NLP Recommender Systems (MobiDev)

Combines:
1. Semantic Similarity (SBERT) - Core relevance signal
2. Keyword Matching (TF-IDF weighted) - Discriminative terms
3. Entity Matching (NER) - Avoid confusion (e.g., location vs tech)
4. Impact Score (Rating + Popularity) - Quality regularizer

Key Principle: Semantic similarity MUST dominate, others are enhancements
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import config

class HybridScorer:
    """
    Hybrid NLP Recommender with proper signal hierarchy
    
    Scoring Formula:
    final_score = 0.60 × semantic + 0.20 × keywords + 0.10 × entities + 0.10 × impact
    
    Why this weighting:
    - Semantic (60%): Core relevance - "what does user really want?"
    - Keywords (20%): Discriminative terms - "specific requirements"
    - Entities (10%): Avoid confusion - "location vs org vs person"
    - Impact (10%): Quality filter - "is this a good book?"
    """
    
    def __init__(self, nlp_retrieval):
        """
        Args:
            nlp_retrieval: NLPRetrieval instance (already fitted)
        """
        self.retrieval = nlp_retrieval
        self.df = nlp_retrieval.df
        
        # Proper weights hierarchy
        self.weights = {
            'similarity': 0.50,     # Semantic similarity DOMINATES
            'keywords': 0.20,       # Discriminative term matching
            'category': 0.15,       # Category alignment (NEW!)
            'entities': 0.05,       # Entity precision
            'impact': 0.10          # Quality regularizer
        }
        
        # TF-IDF vectorizer for keyword importance
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Category intent mappings (for boosting)
        self.category_intents = {
            'children': ['juvenile', 'children', 'young adult', 'kids'],
            'biography': ['biography', 'memoir', 'autobiography'],
            'fiction': ['fiction', 'novel', 'story'],
            'self-help': ['self-help', 'self help', 'personal development'],
            'productivity': ['business', 'management', 'productivity'],
            'romance': ['romance', 'romantic', 'love story'],
            'mystery': ['mystery', 'detective', 'crime', 'thriller'],
            'science': ['science', 'scientific', 'research'],
            'history': ['history', 'historical']
        }
    
    def fit_tfidf(self, corpus):
        """
        Fit TF-IDF on entire corpus to get term importance
        This is crucial for discriminative keyword scoring
        """
        print("[*] Fitting TF-IDF for keyword importance...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams + bigrams
            stop_words='english',
            min_df=2  # Must appear in at least 2 documents
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        print(f"[OK] TF-IDF fitted with {len(self.tfidf_vectorizer.vocabulary_)} terms")
    
    def score(self, query, candidates, weights=None, strategy='auto'):
        """
        Score candidate books using hybrid approach
        
        Args:
            query: User query string
            candidates: List of dicts from retrieval.search()
            weights: Optional custom weights dict
            strategy: Which field strategy was used ('auto', 'title', 'description', etc.)
            
        Returns:
            List of scored candidates (sorted by final_score)
        """
        if weights is None:
            weights = self.weights
        
        print(f"\n[HYBRID] Scoring {len(candidates)} candidates...")
        print(f"   Weights: {weights}")
        print(f"   Field strategy: {strategy}")
        
        # Fit TF-IDF if not already fitted
        if self.tfidf_vectorizer is None:
            corpus = self.df['combined_text'].fillna('').astype(str).tolist()
            self.fit_tfidf(corpus)
        
        # Extract candidate book_ids and similarity scores
        candidate_ids = [c['book_id'] for c in candidates]
        sim_scores = {c['book_id']: c['similarity_score'] for c in candidates}
        
        # 1. Extract query features
        query_keywords = self._extract_discriminative_keywords(query)
        query_entities = self.retrieval.extract_entities(query)
        
        print(f"   Query keywords (TF-IDF weighted): {list(query_keywords.keys())[:5]}")
        print(f"   Query entities: {self._format_entities(query_entities)}")
        
        # 2. Calculate weighted keyword scores (TF-IDF based)
        keyword_scores = self._calculate_tfidf_keyword_scores(
            query, query_keywords, candidate_ids
        )
        
        # 3. Calculate category alignment scores (NEW!)
        category_scores = self._calculate_category_alignment(
            query, candidate_ids
        )
        
        # 4. Calculate entity precision scores
        entity_scores = self._calculate_entity_precision_scores(
            query_entities, candidate_ids
        )
        
        # 5. Calculate impact scores (quality regularizer)
        impact_scores = self._calculate_impact_scores(candidate_ids)
        
        # 6. Normalize all scores to [0, 1]
        kw_norm = self._normalize_scores(keyword_scores, candidate_ids)
        cat_norm = self._normalize_scores(category_scores, candidate_ids)
        ent_norm = self._normalize_scores(entity_scores, candidate_ids)
        impact_norm = self._normalize_scores(impact_scores, candidate_ids)
        
        # 7. Combine scores with proper hierarchy
        final_scores = {}
        for book_id in candidate_ids:
            score = (
                weights.get('similarity', 0.50) * sim_scores[book_id] +
                weights.get('keywords', 0.20) * kw_norm[book_id] +
                weights.get('category', 0.15) * cat_norm[book_id] +
                weights.get('entities', 0.05) * ent_norm[book_id] +
                weights.get('impact', 0.10) * impact_norm[book_id]
            )
            final_scores[book_id] = score
        
        # 7. Update candidates with hybrid scores
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
                'num_ratings': book_data.get('num_ratings', 0),
                'description': book_data.get('description', ''),
                'final_score': final_scores[book_id],
                'scores_breakdown': {
                    'similarity': sim_scores[book_id],
                    'keywords': kw_norm[book_id],
                    'category': cat_norm[book_id],
                    'entities': ent_norm[book_id],
                    'impact': impact_norm[book_id]
                },
                'matched_keywords': query_keywords,
                'matched_entities': query_entities
            })
        
        # Sort by final score
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"[OK] Hybrid scoring complete")
        print(f"   Top score: {scored_candidates[0]['final_score']:.3f}")
        print(f"   Lowest score: {scored_candidates[-1]['final_score']:.3f}")
        
        return scored_candidates
    
    def _extract_discriminative_keywords(self, query):
        """
        Extract keywords with TF-IDF importance weights
        This filters out generic terms like "book", "fiction"
        """
        # Transform query using fitted TF-IDF
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Get non-zero terms and their weights
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        keywords = {}
        
        for idx in query_vec.nonzero()[1]:
            term = feature_names[idx]
            weight = query_vec[0, idx]
            keywords[term] = weight
        
        # Sort by weight and return top terms
        sorted_kw = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True))
        return sorted_kw
    
    def _calculate_tfidf_keyword_scores(self, query, query_keywords, candidate_ids):
        """
        Calculate weighted keyword overlap using TF-IDF cosine similarity
        This properly handles discriminative vs generic terms
        """
        scores = {}
        
        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])
        
        for book_id in candidate_ids:
            book_row = self.df[self.df['book_id'] == book_id]
            
            if len(book_row) == 0:
                scores[book_id] = 0.0
                continue
            
            book_idx = book_row.index[0]
            book_vec = self.tfidf_matrix[book_idx]
            
            # Cosine similarity between query and book TF-IDF vectors
            similarity = cosine_similarity(query_vec, book_vec)[0][0]
            scores[book_id] = similarity
        
        return scores
    
    def _calculate_category_alignment(self, query, candidate_ids):
        """
        Calculate category alignment score (NEW!)
        Detects query intent and boosts books with matching categories
        
        Example:
        - Query: "children's adventure" → Boost "Juvenile Fiction"
        - Query: "biography of scientists" → Boost "Biography"
        """
        scores = {}
        query_lower = query.lower()
        
        # Detect intents in query
        detected_intents = []
        for intent, keywords in self.category_intents.items():
            if any(kw in query_lower for kw in keywords):
                detected_intents.append(intent)
        
        # If no intent detected, return neutral scores
        if not detected_intents:
            return {book_id: 0.5 for book_id in candidate_ids}
        
        # Score each book based on category alignment
        for book_id in candidate_ids:
            book_row = self.df[self.df['book_id'] == book_id]
            
            if len(book_row) == 0:
                scores[book_id] = 0.0
                continue
            
            book_row = book_row.iloc[0]
            book_category = str(book_row.get('categories', '')).lower()
            
            # Calculate alignment score
            alignment = 0.0
            
            for intent in detected_intents:
                intent_keywords = self.category_intents[intent]
                
                # Check if ANY keyword matches category
                if any(kw in book_category for kw in intent_keywords):
                    alignment += 1.0
            
            # Normalize by number of detected intents
            scores[book_id] = alignment / len(detected_intents) if detected_intents else 0.5
        
        return scores
    
    def _calculate_entity_precision_scores(self, query_entities, candidate_ids):
        """
        Calculate entity matching precision
        Purpose: Avoid confusion (e.g., "Washington" location vs person)
        """
        scores = {}
        
        # If no query entities, return 0 for all (entity matching not applicable)
        if not query_entities or all(len(v) == 0 for v in query_entities.values()):
            return {book_id: 0.0 for book_id in candidate_ids}
        
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
            
            # Calculate precision for each entity type
            type_scores = []
            
            for ent_type in ['PERSON', 'GPE', 'ORG', 'WORK_OF_ART', 'NORP']:
                q_ents = set(query_entities.get(ent_type, []))
                b_ents = set(book_entities.get(ent_type, []))
                
                if q_ents:
                    # Precision: how many query entities appear in book
                    matches = len(q_ents & b_ents)
                    precision = matches / len(q_ents)
                    type_scores.append(precision)
            
            # Average precision across entity types
            scores[book_id] = np.mean(type_scores) if type_scores else 0.0
        
        return scores
    
    def _calculate_impact_scores(self, candidate_ids):
        """
        Calculate impact as quality regularizer (NOT ranking driver)
        Formula: 0.7 × rating + 0.3 × log_popularity
        """
        scores = {}
        
        for book_id in candidate_ids:
            book_row = self.df[self.df['book_id'] == book_id]
            
            if len(book_row) == 0:
                scores[book_id] = 0.0
                continue
            
            book_row = book_row.iloc[0]
            
            # Rating component (0-1)
            avg_rating = book_row.get('avg_rating', 0)
            if pd.isna(avg_rating):
                avg_rating = 0
            norm_rating = float(avg_rating) / 5.0
            
            # Popularity component (log scale, 0-1)
            num_ratings = book_row.get('num_ratings', 0)
            if pd.isna(num_ratings):
                num_ratings = 0
            norm_popularity = np.log1p(float(num_ratings)) / 15.0
            norm_popularity = min(norm_popularity, 1.0)
            
            # Combine: rating is MORE important than popularity
            impact = (0.7 * norm_rating) + (0.3 * norm_popularity)
            scores[book_id] = impact
        
        return scores
    
    def _normalize_scores(self, scores, candidate_ids):
        """
        Min-max normalization to [0, 1]
        """
        values = [scores[book_id] for book_id in candidate_ids]
        
        if len(values) == 0:
            return {book_id: 0.0 for book_id in candidate_ids}
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            # If all scores equal, check if they're all 0
            if max_val == 0:
                return {book_id: 0.0 for book_id in candidate_ids}
            else:
                return {book_id: 1.0 for book_id in candidate_ids}
        
        normalized = {}
        for book_id in candidate_ids:
            normalized[book_id] = (scores[book_id] - min_val) / (max_val - min_val)
        
        return normalized
    
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
        breakdown = book['scores_breakdown']
        print(f"   Breakdown: Sim={breakdown['similarity']:.2f}, "
              f"KW={breakdown['keywords']:.2f}, "
              f"Cat={breakdown['category']:.2f}, "
              f"Ent={breakdown['entities']:.2f}, "
              f"Impact={breakdown['impact']:.2f}")