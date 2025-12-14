"""
Script 04: Evaluate Recommendation System
Calculate metrics and generate evaluation report
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.nlp_retrieval import NLPRetrieval
from src.models.hybrid_scorer import HybridScorer
from src.utils.metrics import RecommendationMetrics
from src.config import config

# Test queries with expected categories
TEST_CASES = [
    {
        'query': 'adventure book with magic for children',
        'expected_categories': ['Fiction', 'Fantasy', 'Juvenile Fiction'],
        'expected_keywords': ['adventure', 'magic', 'children']
    },
    {
        'query': 'sách về lịch sử chiến tranh thế giới',
        'expected_categories': ['History', 'Military'],
        'expected_keywords': ['war', 'history', 'world']
    },
    {
        'query': 'biography of famous scientists',
        'expected_categories': ['Biography', 'Science', 'History'],
        'expected_keywords': ['biography', 'scientist']
    },
    {
        'query': 'romantic story during world war',
        'expected_categories': ['Fiction', 'Romance', 'Historical Fiction'],
        'expected_keywords': ['romantic', 'love', 'war']
    },
    {
        'query': 'self-help book about productivity',
        'expected_categories': ['Self-Help', 'Business', 'Psychology'],
        'expected_keywords': ['self-help', 'productivity']
    },
    {
        'query': 'mystery thriller with detective',
        'expected_categories': ['Fiction', 'Mystery', 'Thriller'],
        'expected_keywords': ['mystery', 'detective', 'thriller']
    }
]

def evaluate_relevance(result, expected_categories, expected_keywords):
    """
    Manual relevance evaluation (0-2 scale)
    0 = Not relevant, 1 = Partially relevant, 2 = Highly relevant
    """
    score = 0
    
    # Check category match
    result_cats = str(result.get('categories', '')).lower()
    for exp_cat in expected_categories:
        if exp_cat.lower() in result_cats:
            score += 1
            break
    
    # Check keyword match
    title = str(result.get('title', '')).lower()
    desc = str(result.get('description', '')).lower()
    
    for exp_kw in expected_keywords:
        if exp_kw in title or exp_kw in desc:
            score += 0.5
    
    # Normalize to [0, 2]
    return min(score, 2)

def main():
    print("="*100)
    print("[EVALUATION] RECOMMENDATION SYSTEM")
    print("="*100)
    
    # Load data
    print(f"\n[*] Loading data...")
    df = pd.read_csv(config.BOOKS_PROCESSED)
    print(f"[OK] Loaded {len(df)} books")
    
    # Initialize models
    print("\n[*] Initializing models...")
    retrieval = NLPRetrieval()
    retrieval.fit(df)
    scorer = HybridScorer(retrieval)
    metrics_calc = RecommendationMetrics()
    print("[OK] Models ready")
    
    # ========================================
    # EVALUATE EACH TEST CASE
    # ========================================
    print("\n" + "="*100)
    print("[RUNNING EVALUATIONS]")
    print("="*100)
    
    all_results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        query = test_case['query']
        expected_cats = test_case['expected_categories']
        expected_kws = test_case['expected_keywords']
        
        print(f"\n[{i}/{len(TEST_CASES)}] Query: '{query}'")
        
        # Get recommendations
        candidates = retrieval.search(query, top_k=20)
        hybrid_results = scorer.score(query, candidates)
        
        # Evaluate top-K results
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            top_k_results = hybrid_results[:k]
            
            # Calculate relevance scores
            relevance_scores = [
                evaluate_relevance(r, expected_cats, expected_kws)
                for r in top_k_results
            ]
            
            # Calculate metrics
            precision = sum([1 for s in relevance_scores if s >= 1]) / k
            avg_relevance = np.mean(relevance_scores) / 2  # Normalize to [0, 1]
            
            # Diversity (unique categories)
            categories = [r.get('categories', '') for r in top_k_results]
            unique_cats = len(set([str(c).split(',')[0] for c in categories if c]))
            diversity = unique_cats / k
            
            all_results.append({
                'query': query,
                'k': k,
                'precision': precision,
                'avg_relevance': avg_relevance,
                'diversity': diversity,
                'top_1_title': top_k_results[0]['title'] if top_k_results else 'N/A',
                'top_1_score': top_k_results[0]['final_score'] if top_k_results else 0
            })
        
        # Display quick summary
        print(f"   Top-1: {hybrid_results[0]['title']}")
        print(f"   Score: {hybrid_results[0]['final_score']:.3f}")
        print(f"   Relevance: {evaluate_relevance(hybrid_results[0], expected_cats, expected_kws)}/2")
    
    # ========================================
    # AGGREGATE RESULTS
    # ========================================
    print("\n" + "="*100)
    print("[EVALUATION RESULTS]")
    print("="*100)
    
    results_df = pd.DataFrame(all_results)
    
    # Group by K
    for k in [1, 3, 5, 10]:
        k_results = results_df[results_df['k'] == k]
        
        print(f"\n[METRICS @ K={k}]")
        print(f"  Precision@{k}:    {k_results['precision'].mean():.3f} ± {k_results['precision'].std():.3f}")
        print(f"  Avg Relevance:    {k_results['avg_relevance'].mean():.3f} ± {k_results['avg_relevance'].std():.3f}")
        print(f"  Diversity:        {k_results['diversity'].mean():.3f} ± {k_results['diversity'].std():.3f}")
    
    # ========================================
    # COMPARISON TABLE
    # ========================================
    print("\n" + "="*100)
    print("[DETAILED RESULTS TABLE]")
    print("="*100)
    
    # Show results for K=5
    k5_results = results_df[results_df['k'] == 5].copy()
    k5_results['precision'] = k5_results['precision'].apply(lambda x: f"{x:.2f}")
    k5_results['avg_relevance'] = k5_results['avg_relevance'].apply(lambda x: f"{x:.2f}")
    k5_results['diversity'] = k5_results['diversity'].apply(lambda x: f"{x:.2f}")
    k5_results['top_1_score'] = k5_results['top_1_score'].apply(lambda x: f"{x:.3f}")
    
    print("\n[RESULTS @ K=5]")
    print(k5_results[['query', 'top_1_title', 'precision', 'avg_relevance', 'diversity']].to_string(index=False))
    
    # ========================================
    # QUALITATIVE ANALYSIS
    # ========================================
    print("\n" + "="*100)
    print("[QUALITATIVE ANALYSIS]")
    print("="*100)
    
    # Success cases (precision@5 >= 0.8)
    success_cases = results_df[(results_df['k'] == 5) & (results_df['precision'] >= 0.8)]
    print(f"\n[HIGH PRECISION QUERIES] (Precision@5 >= 0.8)")
    print(f"  Count: {len(success_cases)}/{len(TEST_CASES)}")
    if len(success_cases) > 0:
        for _, case in success_cases.iterrows():
            print(f"  - '{case['query']}' (Precision: {case['precision']:.2f})")
    
    # Failure cases (precision@5 < 0.5)
    failure_cases = results_df[(results_df['k'] == 5) & (results_df['precision'] < 0.5)]
    print(f"\n[LOW PRECISION QUERIES] (Precision@5 < 0.5)")
    print(f"  Count: {len(failure_cases)}/{len(TEST_CASES)}")
    if len(failure_cases) > 0:
        for _, case in failure_cases.iterrows():
            print(f"  - '{case['query']}' (Precision: {case['precision']:.2f})")
            print(f"    Top-1: {case['top_1_title']}")
    
    # ========================================
    # SAVE REPORT
    # ========================================
    output_file = "evaluation_report.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n[SAVE] Detailed results saved to: {output_file}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*100)
    print("[SUMMARY]")
    print("="*100)
    
    avg_precision_5 = results_df[results_df['k'] == 5]['precision'].mean()
    avg_relevance_5 = results_df[results_df['k'] == 5]['avg_relevance'].mean()
    avg_diversity_5 = results_df[results_df['k'] == 5]['diversity'].mean()
    
    print(f"\nOverall Performance @ K=5:")
    print(f"  Precision:     {avg_precision_5:.3f}")
    print(f"  Relevance:     {avg_relevance_5:.3f}")
    print(f"  Diversity:     {avg_diversity_5:.3f}")
    
    # Rating
    if avg_precision_5 >= 0.8:
        rating = "Excellent"
    elif avg_precision_5 >= 0.6:
        rating = "Good"
    elif avg_precision_5 >= 0.4:
        rating = "Fair"
    else:
        rating = "Needs Improvement"
    
    print(f"\nSystem Rating: {rating}")
    
    print("\n" + "="*100)
    print("[EVALUATION COMPLETE]")
    print("="*100)


if __name__ == "__main__":
    main()