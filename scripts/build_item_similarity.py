"""
Build item similarity matrices
python scripts/07_build_item_similarity.py
"""
import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.models.item_similarity import ItemSimilarityCalculator

def main():
    print("Building similarity matrices...")
    
    embeddings = np.load('data/features/book_embeddings.npy')
    with open('data/features/book_ids.pkl', 'rb') as f:
        book_ids = pickle.load(f)
    
    if len(book_ids) != embeddings.shape[0]:
        min_len = min(len(book_ids), embeddings.shape[0])
        book_ids = book_ids[:min_len]
        embeddings = embeddings[:min_len]
    
    ratings_df = pd.read_csv('data/processed/ratings.csv')
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
    
    calc.save('data/processed/item_similarity.pkl')
    print("Build complete!")

if __name__ == '__main__':
    main()