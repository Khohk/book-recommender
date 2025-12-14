"""
Data Loader Module
Loads and validates raw data from CSV
"""

import pandas as pd
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
        
        return df
    
    def _rename_columns(self, df):
        """
        Rename columns based on COLUMN_MAPPING
        Original columns → Standardized names
        """
        # Get reverse mapping (standardized → original)
        rename_dict = {v: k for k, v in self.column_mapping.items() if v in df.columns}
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"[OK] Renamed columns: {list(rename_dict.keys())}")
        
        return df
    
    def _validate_data(self,df):
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
        print(f"{'Column':<20} {'Non-Null':<10} {'Dtype':<15} {'Sample'}")
        print("-" * 80)
        
        for col in df.columns:
            non_null = df[col].notna().sum()
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0])[:30] if non_null > 0 else "N/A"
            print(f"{col:<20} {non_null:<10} {dtype:<15} {sample}")


if __name__ == "__main__":
    # Test
    loader = DataLoader()
    df = loader.load_master_books()
    loader.get_column_info(df)