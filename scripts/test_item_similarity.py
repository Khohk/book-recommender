"""
Standalone test script
python scripts/09_standalone_test.py
"""
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.models.item_similarity import ItemSimilarityCalculator

def main():
    print("Loading data...")
    embeddings = np.load('data/features/book_embeddings.npy')
    
    with open('data/features/book_ids.pkl', 'rb') as f:
        book_ids = pickle.load(f)
    
    if len(book_ids) != embeddings.shape[0]:
        min_len = min(len(book_ids), embeddings.shape[0])
        book_ids = book_ids[:min_len]
        embeddings = embeddings[:min_len]
    
    ratings_df = pd.read_csv('data/processed/ratings.csv')
    books_df = pd.read_csv('data/processed/books_with_combined_text.csv')

    # Tạo từ điển map từ book_id -> title để tra cứu cho nhanh
    book_titles = dict(zip(books_df['book_id'], books_df['title']))
    
    print("Building similarity matrices...")
    calc = ItemSimilarityCalculator()
    calc.fit(
        embeddings=embeddings,
        book_ids=book_ids,
        ratings_df=ratings_df,
        method='hybrid',
        content_weight=0.6,
        cf_weight=0.4,
        min_common_users=3
    )
    
    print("Testing recommendations...")
    test_book = book_ids[5]
    
    # In ra tên cuốn sách đang được test# 
    test_book_title = book_titles.get(test_book, "Unknown Title")
    print(f"Test Book: {test_book_title} ({test_book})")
    for method in ['content', 'cf', 'hybrid']:
        print(f"\nMethod: {method}")
        similar = calc.get_similar_items(test_book, top_k=5, method=method)
        for i, (bid, score) in enumerate(similar, 1):
            # Dùng từ điển để tra tên sách từ ID (bid)
            rec_title = book_titles.get(bid, "Unknown Title")
            # In ra Tên sách trước, sau đó mới in ID và điểm số
            print(f"  {i}. {rec_title}")
            print(f"     (ID: {bid} | Score: {score:.4f})")
    
    print("\nTest complete!")

if __name__ == '__main__':
    main()