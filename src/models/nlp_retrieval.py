"""
NLP Retrieval Model
Uses SBERT for semantic search
Implements: Text Similarity, Keyword Extraction, NER
"""

import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import spacy
from src.config import config
import os

class NLPRetrieval:
    """
    Stage 1: Candidate Generation using NLP
    Returns top-N semantically similar books
    """
    
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        
        # Models (lazy loading)
        self._sbert_model = None
        self._keyword_model = None
        self._ner_model = None
        
        # Precomputed features
        self.embeddings = None
        self.book_ids = None
        self.df = None
    
    @property
    def sbert_model(self):
        """Lazy load SBERT model"""
        if self._sbert_model is None:
            print(f"[*] Loading SBERT model: {self.model_name}")
            self._sbert_model = SentenceTransformer(self.model_name)
            print("[OK] SBERT model loaded")
        return self._sbert_model
    
    @property
    def keyword_model(self):
        """Lazy load KeyBERT model"""
        if self._keyword_model is None:
            print("[*] Loading KeyBERT model...")
            self._keyword_model = KeyBERT(self.model_name)
            print("[OK] KeyBERT model loaded")
        return self._keyword_model
    
    @property
    def ner_model(self):
        """Lazy load Spacy NER model"""
        if self._ner_model is None:
            try:
                print(f"[*] Loading Spacy model: {config.SPACY_MODEL}")
                self._ner_model = spacy.load(config.SPACY_MODEL)
                print("[OK] Spacy model loaded")
            except:
                print("[WARNING] Spacy model not available, NER will be disabled")
                self._ner_model = None
        return self._ner_model
    
    def fit(self, df):
        """
        Precompute embeddings for all books
        
        Args:
            df: DataFrame with 'combined_text' column
        """
        self.df = df
        self.book_ids = df['book_id'].tolist()
        
        # Check if embeddings already exist
        if os.path.exists(config.BOOK_EMBEDDINGS):
            print(f"\n[*] Loading precomputed embeddings from: {config.BOOK_EMBEDDINGS}")
            self.embeddings = np.load(config.BOOK_EMBEDDINGS)
            print(f"[OK] Loaded embeddings: {self.embeddings.shape}")
        else:
            # Generate new embeddings
            self._generate_embeddings(df)
    
    def _generate_embeddings(self, df):
        """
        Generate SBERT embeddings for all books
        """
        print(f"\n[*] Generating embeddings for {len(df)} books...")
        print("   (This may take 5-15 minutes depending on dataset size)")
        
        texts = df['combined_text'].tolist()
        
        self.embeddings = self.sbert_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        print(f"[OK] Generated embeddings: {self.embeddings.shape}")
        
        # Save embeddings
        self._save_embeddings()
    
    def _save_embeddings(self):
        """Save embeddings to disk"""
        os.makedirs(config.FEATURES_DIR, exist_ok=True)
        
        print(f"\n[*] Saving embeddings to: {config.BOOK_EMBEDDINGS}")
        np.save(config.BOOK_EMBEDDINGS, self.embeddings)
        
        print(f"[*] Saving book IDs to: {config.BOOK_IDS}")
        with open(config.BOOK_IDS, 'wb') as f:
            pickle.dump(self.book_ids, f)
        
        print("[OK] Embeddings saved successfully")
    
    def search(self, query, top_k=100):
        """
        Semantic search using query text
        
        Args:
            query: Free-text query (Vietnamese or English)
            top_k: Number of candidates to return
            
        Returns:
            List of (book_id, similarity_score) tuples
        """
        print(f"\n[SEARCH] Searching: '{query}'")
        
        # Encode query
        query_embedding = self.sbert_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Create results
        results = []
        for idx in top_indices:
            results.append({
                'book_id': self.book_ids[idx],
                'similarity_score': float(similarities[idx]),
                'book_data': self.df.iloc[idx].to_dict()
            })
        
        print(f"[OK] Found {len(results)} candidates")
        print(f"   Top score: {results[0]['similarity_score']:.3f}")
        print(f"   Lowest score: {results[-1]['similarity_score']:.3f}")
        
        return results
    
    def extract_keywords(self, text, top_n=None):
        """
        Extract keywords from text using KeyBERT
        
        Args:
            text: Input text
            top_n: Number of keywords (default from config)
            
        Returns:
            List of keyword strings
        """
        if top_n is None:
            top_n = config.KEYWORD_TOP_N
        
        try:
            keywords = self.keyword_model.extract_keywords(
                text,
                keyphrase_ngram_range=config.KEYWORD_NGRAM_RANGE,
                stop_words='english',
                top_n=top_n
            )
            return [kw[0] for kw in keywords]
        except:
            return []
    
    def extract_entities(self, text):
        """
        Extract named entities using Spacy
        
        Args:
            text: Input text
            
        Returns:
            Dict of entity_type -> [entity_texts]
        """
        if self.ner_model is None:
            return {ent_type: [] for ent_type in config.ENTITY_TYPES}
        
        try:
            doc = self.ner_model(text[:1000])  # Limit text length
            entities = {ent_type: [] for ent_type in config.ENTITY_TYPES}
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
            
            return entities
        except:
            return {ent_type: [] for ent_type in config.ENTITY_TYPES}


if __name__ == "__main__":
    # Test NLP Retrieval
    from src.preprocessing.data_loader import DataLoader
    from src.preprocessing.text_cleaner import TextPreprocessor
    
    # Load data
    loader = DataLoader()
    df = loader.load_raw_data()
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df = preprocessor.process(df)
    
    # Create retrieval model
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    
    # Test search
    query = "adventure book with magic for children"
    results = retrieval.search(query, top_k=10)
    
    # Display results
    print("\n[*] Top 10 Results:")
    for i, res in enumerate(results, 1):
        print(f"\n{i}. {res['book_data']['title']}")
        print(f"   Score: {res['similarity_score']:.3f}")
        print(f"   Author: {res['book_data'].get('authors', 'Unknown')}")