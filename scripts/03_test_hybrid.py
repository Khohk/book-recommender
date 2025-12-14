"""
Script 03: Test Hybrid Scoring
Compare Pure SBERT vs Hybrid (SBERT + Keywords + Entities)
"""

import sys
import os
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.models.hybrid_scorer import HybridScorer
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

def display_comparison(query, pure_results, hybrid_results, top_k=5):
    """
    Display side-by-side comparison of Pure SBERT vs Hybrid
    """
    print("\n" + "="*100)
    print(f"QUERY: '{query}'")
    print("="*100)
    
    print("\n" + "-"*100)
    print(f"{'PURE SBERT':<50} | {'HYBRID (SBERT + KW + ENT)':<50}")
    print("-"*100)
    
    for i in range(top_k):
        # Pure SBERT result
        pure = pure_results[i] if i < len(pure_results) else None
        hybrid = hybrid_results[i] if i < len(hybrid_results) else None
        
        # Format pure result
        if pure:
            pure_str = f"#{i+1} {pure['book_data']['title'][:40]}"
            pure_score = f"(Sim: {pure['similarity_score']:.3f})"
        else:
            pure_str = "-"
            pure_score = ""
        
        # Format hybrid result
        if hybrid:
            hybrid_str = f"#{i+1} {hybrid['title'][:40]}"
            hybrid_score = f"(Final: {hybrid['final_score']:.3f})"
        else:
            hybrid_str = "-"
            hybrid_score = ""
        
        print(f"{pure_str:<40} {pure_score:<10} | {hybrid_str:<40} {hybrid_score:<10}")
    
    print("-"*100)

def display_hybrid_details(result):
    """
    Display detailed breakdown of hybrid scoring
    """
    print(f"\nTitle: {result['title']}")
    print(f"Author: {result['author']}")
    print(f"Categories: {result['categories']}")
    
    print(f"\n[SCORES]")
    print(f"  Final Score:  {result['final_score']:.4f}")
    print(f"  - Similarity: {result['scores_breakdown']['similarity']:.4f} (70%)")
    print(f"  - Keywords:   {result['scores_breakdown']['keywords']:.4f} (20%)")
    print(f"  - Entities:   {result['scores_breakdown']['entities']:.4f} (10%)")
    
    if result.get('matched_keywords'):
        print(f"\n[MATCHED KEYWORDS]")
        print(f"  {', '.join(result['matched_keywords'])}")
    
    if result.get('matched_entities'):
        print(f"\n[MATCHED ENTITIES]")
        for ent_type, ents in result['matched_entities'].items():
            print(f"  {ent_type}: {', '.join(ents)}")
    
    # Description preview
    desc = result.get('description', '')
    if desc and len(str(desc)) > 0:
        print(f"\n[DESCRIPTION]")
        print(f"  {str(desc)[:200]}...")

def main():
    print("="*100)
    print("[TEST] HYBRID SCORING SYSTEM")
    print("="*100)
    
    # Load processed data
    print(f"\n[*] Loading processed data from: {config.BOOKS_PROCESSED}")
    df = pd.read_csv(config.BOOKS_PROCESSED)
    print(f"[OK] Loaded {len(df)} books")
    
    # Initialize NLP retrieval
    print("\n[*] Initializing NLP Retrieval model...")
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    
    # Initialize hybrid scorer
    print("\n[*] Initializing Hybrid Scorer...")
    scorer = HybridScorer(retrieval)
    print("[OK] Hybrid scorer ready")
    
    # ========================================
    # MODE SELECTION
    # ========================================
    print("\n" + "="*100)
    print("[MODE SELECTION]")
    print("="*100)
    print("1. Compare Pure SBERT vs Hybrid (side-by-side)")
    print("2. Test Hybrid only (detailed)")
    print("3. Run all test queries")
    
    mode = input("\nSelect mode (1/2/3): ").strip()
    
    if mode == "1":
        # ========================================
        # MODE 1: COMPARISON
        # ========================================
        print("\n" + "="*100)
        print("[MODE 1] COMPARISON: Pure SBERT vs Hybrid")
        print("="*100)
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n\n{'#'*100}")
            print(f"TEST #{i}/{len(TEST_QUERIES)}")
            print(f"{'#'*100}")
            
            # Pure SBERT results
            print("\n[1/2] Getting Pure SBERT results...")
            pure_results = retrieval.search(query, top_k=20)
            
            # Hybrid results
            print("[2/2] Getting Hybrid results...")
            hybrid_results = scorer.score(query, pure_results[:20])
            
            # Display comparison
            display_comparison(query, pure_results, hybrid_results, top_k=5)
            
            if i < len(TEST_QUERIES):
                input("\n[PAUSE] Press Enter to continue...")
    
    elif mode == "2":
        # ========================================
        # MODE 2: HYBRID DETAILED
        # ========================================
        print("\n" + "="*100)
        print("[MODE 2] HYBRID SCORING (Detailed)")
        print("="*100)
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n\n{'#'*100}")
            print(f"TEST #{i}/{len(TEST_QUERIES)}")
            print(f"{'#'*100}")
            print(f"QUERY: '{query}'")
            
            # Get candidates
            print("\n[1/2] Retrieving candidates...")
            candidates = retrieval.search(query, top_k=20)
            
            # Score with hybrid
            print("[2/2] Hybrid scoring...")
            results = scorer.score(query, candidates)
            
            # Display top 3 with details
            print("\n" + "="*100)
            print("[TOP 3 RESULTS]")
            print("="*100)
            
            for rank, result in enumerate(results[:3], 1):
                print(f"\n{'-'*100}")
                print(f"[RANK #{rank}]")
                print(f"{'-'*100}")
                display_hybrid_details(result)
            
            if i < len(TEST_QUERIES):
                input("\n[PAUSE] Press Enter to continue...")
    
    elif mode == "3":
        # ========================================
        # MODE 3: RUN ALL TESTS
        # ========================================
        print("\n" + "="*100)
        print("[MODE 3] RUNNING ALL TESTS")
        print("="*100)
        
        results_summary = []
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] Testing: '{query}'")
            
            # Get candidates
            candidates = retrieval.search(query, top_k=20)
            
            # Score
            results = scorer.score(query, candidates)
            
            # Store summary
            results_summary.append({
                'query': query,
                'top_1_title': results[0]['title'],
                'top_1_score': results[0]['final_score'],
                'top_1_categories': results[0]['categories']
            })
            
            print(f"   Top 1: {results[0]['title']} (Score: {results[0]['final_score']:.3f})")
        
        # Display summary table
        print("\n" + "="*100)
        print("[SUMMARY]")
        print("="*100)
        
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
    
    else:
        print("[ERROR] Invalid mode selection")
        return
    
    # ========================================
    # INTERACTIVE MODE
    # ========================================
    print("\n" + "="*100)
    print("[INTERACTIVE] Custom Query Testing")
    print("="*100)
    print("Enter your own query (or 'quit' to exit)\n")
    
    while True:
        user_query = input("[>>] Your query: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n[*] Goodbye!")
            break
        
        if not user_query:
            continue
        
        # Get results
        print("\n[*] Processing...")
        candidates = retrieval.search(user_query, top_k=20)
        results = scorer.score(user_query, candidates)
        
        # Display top 5
        print("\n" + "="*100)
        print(f"[TOP 5 RESULTS] Query: '{user_query}'")
        print("="*100)
        
        for rank, result in enumerate(results[:5], 1):
            print(f"\n{'-'*100}")
            print(f"[RANK #{rank}]")
            print(f"{'-'*100}")
            display_hybrid_details(result)
        
        print()
    
    print("\n" + "="*100)
    print("[OK] TEST COMPLETED!")
    print("="*100)


if __name__ == "__main__":
    main()