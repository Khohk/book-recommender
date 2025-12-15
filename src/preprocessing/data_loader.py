"""
Data Loader Module
Loads and validates raw data from CSV
"""

import pandas as pd
import numpy as np
import os
from src.config import config

class DataLoader:
    """
    Load processed master books data
    (Already cleaned and deduplicated)
    """
    
    def __init__(self):
        self.master_books_file = config.MASTER_BOOKS
    
    def load_master_books(self):
        """
        Load master books dataset (1 row per book)
        
        Returns:
            pd.DataFrame: Master books with unique book_id
        """
        print(f"[*] Loading master books from: {self.master_books_file}")
        
        if not os.path.exists(self.master_books_file):
            raise FileNotFoundError(
                f"Master books file not found: {self.master_books_file}\n"
                f"Please run: python scripts/00_prepare_datasets.py first"
            )
        
        # Load CSV
        df = pd.read_csv(self.master_books_file)
        print(f"[OK] Loaded {len(df)} unique books")
        
        # Basic validation
        self._validate_data(df)
        
        # Add impact score if rating columns exist
        df = self._add_impact_score(df)
        
        return df
    
    def _add_impact_score(self, df):
        """
        Add impact_score column based on rating + popularity (NEW)
        
        Impact Score = (normalized_rating × 0.6) + (normalized_popularity × 0.4)
        
        This helps prioritize books that are both highly-rated AND popular
        """
        # Check if required columns exist
        if 'avg_rating' not in df.columns or 'num_ratings' not in df.columns:
            print("[WARNING] Cannot add impact_score: missing avg_rating or num_ratings columns")
            print(f"[INFO] Available columns: {df.columns.tolist()}")
            return df
        
        print("[*] Adding impact score...")
        
        # Handle missing values
        df['avg_rating'] = df['avg_rating'].fillna(0)
        df['num_ratings'] = df['num_ratings'].fillna(0)
        
        # Normalize rating to [0, 1]
        df['norm_rating'] = df['avg_rating'] / 5.0
        
        # Normalize popularity using log scale (to handle huge variance)
        # log1p handles 0 values safely
        max_log_ratings = np.log1p(df['num_ratings'].max())
        df['norm_popularity'] = np.log1p(df['num_ratings']) / max_log_ratings
        
        # Combine: 60% rating quality, 40% popularity
        df['impact_score'] = (0.6 * df['norm_rating']) + (0.4 * df['norm_popularity'])
        
        # Drop temporary columns
        df = df.drop(columns=['norm_rating', 'norm_popularity'])
        
        # Stats
        print(f"[OK] Impact score added")
        print(f"   Mean: {df['impact_score'].mean():.3f}")
        print(f"   Median: {df['impact_score'].median():.3f}")
        print(f"   Top 5 books by impact:")
        top_5 = df.nlargest(5, 'impact_score')[['title', 'avg_rating', 'num_ratings', 'impact_score']]
        for idx, row in top_5.iterrows():
            print(f"      - {row['title'][:40]}: {row['impact_score']:.3f} "
                  f"(rating={row['avg_rating']:.1f}, count={row['num_ratings']})")
        
        return df
    
    def _validate_data(self, df):
        """
        Validate required columns exist
        """
        required_cols = ['book_id', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for duplicates
        duplicates = df['book_id'].duplicated().sum()
        if duplicates > 0:
            print(f"[WARNING] Found {duplicates} duplicate book_ids!")
        else:
            print(f"[OK] No duplicate book_ids found")
        
        print(f"[OK] Data validation passed")
    
    def get_column_info(self, df):
        """
        Display information about columns
        """
        print("\n[INFO] Column Information:")
        print(f"{'Column':<25} {'Non-Null':<10} {'Dtype':<15} {'Sample'}")
        print("-" * 90)
        
        for col in df.columns:
            non_null = df[col].notna().sum()
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0])[:35] if non_null > 0 else "N/A"
            print(f"{col:<25} {non_null:<10} {dtype:<15} {sample}")


if __name__ == "__main__":
    # Test
    loader = DataLoader()
    df = loader.load_master_books()
    loader.get_column_info(df)
    
    # Test impact score
    if 'impact_score' in df.columns:
        print("\n[TEST] Impact Score Statistics:")
        print(df['impact_score'].describe())