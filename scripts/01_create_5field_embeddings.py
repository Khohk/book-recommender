"""
Script 01: Create 5-Field Embeddings (Including Reviews)
Replaces old single embedding with strategic 5-field approach

Fields:
1. Title + Categories (20%)
2. Description (30%) 
3. Aggregated Summaries (20%)
4. Aggregated Reviews (20%) ← NEW!
5. Combined weighted (10%)
"""

import sys
import os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.config import config

def main():
    print("="*100)
    print("[STEP 1] CREATE 5-FIELD EMBEDDINGS (WITH REVIEWS)")
    print("="*100)
    
    # Load master books
    print(f"\n[*] Loading books from: {config.MASTER_BOOKS}")
    
    try:
        df = pd.read_csv(config.MASTER_BOOKS)
    except:
        # Fallback to processed
        df = pd.read_csv(config.BOOKS_PROCESSED)
    
    print(f"[OK] Loaded {len(df)} books")
    
    # Validate columns
    required_cols = ['title', 'categories', 'description', 
                    'aggregated_summaries', 'aggregated_reviews']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n[ERROR] Missing required columns: {missing}")
        print(f"[INFO] Available columns: {df.columns.tolist()}")
        print("\n[SOLUTION] Check these column names:")
        print("  - 'aggregated_summaries' (not 'summary')")
        print("  - 'aggregated_reviews' (not 'reviews')")
        return
    
    # Check data quality
    print("\n[INFO] Data Quality Check:")
    for col in required_cols:
        non_empty = df[col].notna().sum()
        pct = (non_empty / len(df)) * 100
        print(f"   {col:25s}: {non_empty:4d}/{len(df)} ({pct:.1f}%)")
    
    # Initialize retrieval model
    print("\n[*] Initializing NLP Retrieval with 5-field strategy...")
    retrieval = NLPRetrieval()
    
    # Create embeddings
    print("\n" + "="*100)
    print("CREATING EMBEDDINGS (This will take 5-8 minutes)...")
    print("="*100)
    
    try:
        retrieval.fit(df)
    except Exception as e:
        print(f"\n[ERROR] Failed to create embeddings: {e}")
        return
    
    # Save embeddings
    output_path = 'data/features/book_embeddings_5field.npz'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    retrieval.save_embeddings(output_path)
    
    # Success message
    print("\n" + "="*100)
    print("[SUCCESS] 5-Field Embeddings Created!")
    print("="*100)
    
    print(f"\n📁 Saved to: {output_path}")
    print(f"📊 Total size: {retrieval.title_embeddings.nbytes * 5 / 1024 / 1024:.1f} MB")
    
    print(f"\n[NEXT STEPS]")
    print(f"1. Update config.py:")
    print(f"   BOOK_EMBEDDINGS = '{output_path}'")
    print(f"2. Run test: python scripts/03_test_hybrid.py")
    
    # Quick test
    print("\n" + "="*100)
    print("[QUICK TEST] Comparing Strategies")
    print("="*100)
    
    test_query = [
            "adventure book with magic for children", 
            "best fantasy books", 
            "science fiction",
            "romantic story about war in 19th century" # Test thêm 1 query phức tạp
    ]    
    for strategy in ['title', 'description', 'review', 'combined']:
        print(f"\n[STRATEGY: {strategy.upper()}]")
        results = retrieval.search(test_query, top_k=3, strategy=strategy)
        
        for i, r in enumerate(results, 1):
            title = r['book_data']['title'][:55]
            score = r['similarity_score']
            print(f"  {i}. {title:55s} ({score:.3f})")
    
    # Show best result
    print("\n" + "="*100)
    print("[RECOMMENDATION] Based on test results:")
    print("="*100)
    
    best_results = retrieval.search(test_query, top_k=5, strategy='combined')
    print(f"\nTop 5 using COMBINED strategy:")
    for i, r in enumerate(best_results, 1):
        print(f"  {i}. {r['book_data']['title']}")
        print(f"     Score: {r['similarity_score']:.3f}")
        print(f"     Categories: {r['book_data'].get('categories', 'N/A')}")


if __name__ == "__main__":
    main()