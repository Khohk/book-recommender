"""
Script 02: Test NLP Retrieval
Test semantic search with sample queries
"""

import sys
import os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.config import config

# Sample test queries
TEST_QUERIES = [
    "adventure book with magic for children",
    "sách về lịch sử chiến tranh thế giới",
    "biography of famous scientists",
    "romantic story during world war",
    "self-help book about productivity",
    "mystery thriller with detective",
    "science fiction about space exploration"
]

def display_results(results, query, top_k=10):
    """Display search results in formatted way"""
    
    print("\n" + "="*80)
    print(f"[SEARCH] QUERY: '{query}'")
    print("="*80)
    
    for i, res in enumerate(results[:top_k], 1):
        book = res['book_data']
        score = res['similarity_score']
        
        print(f"\n{'-'*80}")
        print(f"[RANK #{i}]")
        print(f"{'-'*80}")
        print(f"Title:      {book['title']}")
        print(f"Author:     {book.get('authors', 'Unknown')}")
        print(f"Categories: {book.get('categories', 'N/A')}")
        print(f"Rating:     {book.get('rating', 'N/A')}")
        print(f"\nSimilarity Score: {score:.4f}")
        
        # Show description preview
        desc = book.get('description', 'No description')
        if pd.notna(desc) and len(str(desc)) > 0:
            desc_preview = str(desc)[:200] + "..."
            print(f"\nDescription:\n   {desc_preview}")
        
        print()

def main():
    print("="*80)
    print("[TEST] NLP RETRIEVAL SYSTEM")
    print("="*80)
    
    # Load processed data
    print(f"\n[*] Loading processed data from: {config.BOOKS_PROCESSED}")
    df = pd.read_csv(config.BOOKS_PROCESSED)
    print(f"[OK] Loaded {len(df)} books")
    
    # Initialize retrieval model
    print("\n[*] Initializing NLP Retrieval model...")
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    
    # ========================================
    # TEST WITH SAMPLE QUERIES
    # ========================================
    print("\n" + "="*80)
    print("[TEST] RUNNING TESTS WITH SAMPLE QUERIES")
    print("="*80)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST #{i}/{len(TEST_QUERIES)}")
        print(f"{'#'*80}")
        
        # Search
        results = retrieval.search(query, top_k=10)
        
        # Display
        display_results(results, query, top_k=5)  # Show top 5
        
        # Pause between queries
        if i < len(TEST_QUERIES):
            input("\n[PAUSE] Press Enter to continue to next query...")
    
    # ========================================
    # INTERACTIVE MODE
    # ========================================
    print("\n" + "="*80)
    print("[INTERACTIVE] MODE")
    print("="*80)
    print("Enter your own query (Vietnamese or English)")
    print("Type 'quit' to exit\n")
    
    while True:
        user_query = input("[>>] Your query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n[*] Goodbye!")
            break
        
        if not user_query:
            continue
        
        results = retrieval.search(user_query, top_k=10)
        display_results(results, user_query, top_k=10)
        print()
    
    print("\n" + "="*80)
    print("[OK] TEST COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()