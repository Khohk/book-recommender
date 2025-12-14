"""
Script 01: Preprocessing and Embedding Generation
Run this ONCE to prepare all data

Steps:
1. Load raw data
2. Create combined text
3. Generate SBERT embeddings
4. Extract keywords (optional)
5. Extract entities (optional)
6. Save all features
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.text_cleaner import TextPreprocessor
from src.models.nlp_retrieval import NLPRetrieval
from src.config import config
import pickle

def main():
    print("="*80)
    print("[*] BOOK RECOMMENDATION SYSTEM - PREPROCESSING PIPELINE")
    print("="*80)
    
    # ========================================
    # STEP 1: LOAD RAW DATA
    # ========================================
    print("\n" + "="*80)
    print("STEP 1: LOADING RAW DATA")
    print("="*80)
    
    loader = DataLoader()
    df = loader.load_master_books()
    loader.get_column_info(df)
    
    # ========================================
    # STEP 2: TEXT PREPROCESSING
    # ========================================
    print("\n" + "="*80)
    print("STEP 2: TEXT PREPROCESSING")
    print("="*80)
    
    preprocessor = TextPreprocessor()
    df = preprocessor.process(df)
    
    # Save processed data
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(config.BOOKS_PROCESSED, index=False)
    print(f"\n💾 Saved processed data to: {config.BOOKS_PROCESSED}")
    
    # ========================================
    # STEP 3: GENERATE EMBEDDINGS
    # ========================================
    print("\n" + "="*80)
    print("STEP 3: GENERATING SBERT EMBEDDINGS")
    print("="*80)
    
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    
    # ========================================
    # STEP 4: EXTRACT KEYWORDS (OPTIONAL)
    # ========================================
    print("\n" + "="*80)
    print("STEP 4: EXTRACTING KEYWORDS (Optional - takes time)")
    print("="*80)
    
    extract_keywords = input("Extract keywords? (y/n): ").lower() == 'y'
    
    if extract_keywords:
        print("\n[*] Extracting keywords for all books...")
        df['keywords'] = df['combined_text'].apply(
            lambda x: retrieval.extract_keywords(x)
        )
        
        # Save keywords
        with open(config.BOOK_KEYWORDS, 'wb') as f:
            pickle.dump(df['keywords'].tolist(), f)
        print(f"[OK] Saved keywords to: {config.BOOK_KEYWORDS}")
    else:
        print("[>>] Skipping keyword extraction")
    
    # ========================================
    # STEP 5: EXTRACT ENTITIES (OPTIONAL)
    # ========================================
    print("\n" + "="*80)
    print("STEP 5: EXTRACTING NAMED ENTITIES (Optional - takes time)")
    print("="*80)
    
    extract_entities = input("Extract entities? (y/n): ").lower() == 'y'
    
    if extract_entities:
        print("\n[*] Extracting entities for all books...")
        df['entities'] = df['combined_text'].apply(
            lambda x: retrieval.extract_entities(x)
        )
        
        # Save entities
        with open(config.BOOK_ENTITIES, 'wb') as f:
            pickle.dump(df['entities'].tolist(), f)
        print(f"[OK] Saved entities to: {config.BOOK_ENTITIES}")
    else:
        print("[>>] Skipping entity extraction")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("[OK] PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. Processed data:  {config.BOOKS_PROCESSED}")
    print(f"  2. Embeddings:      {config.BOOK_EMBEDDINGS}")
    print(f"  3. Book IDs:        {config.BOOK_IDS}")
    
    if extract_keywords:
        print(f"  4. Keywords:        {config.BOOK_KEYWORDS}")
    if extract_entities:
        print(f"  5. Entities:        {config.BOOK_ENTITIES}")
    
    print("\n[*] Dataset Statistics:")
    print(f"   Total books: {len(df)}")
    print(f"   Embedding dim: {retrieval.embeddings.shape[1]}")
    print(f"   Avg text length: {df['text_length'].mean():.0f} chars")
    
    print("\n[>>] Next steps:")
    print("   1. Run: python scripts/02_test_retrieval.py")
    print("   2. Or start webapp: streamlit run webapp/app.py")


if __name__ == "__main__":
    main()