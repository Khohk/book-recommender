"""
Text Cleaning and Preprocessing Module
Combines multiple text columns into one 'combined_text'
"""

import pandas as pd
import numpy as np
from src.config import config

class TextPreprocessor:
    """
    Clean and combine text from multiple columns
    Creates 'combined_text' with TOKEN BUDGET management
    Priority: Reviews (60%) > Description (30%) > Title (10%)
    """
    
    def __init__(self):
        self.text_columns = config.TEXT_COLUMNS
        self.min_length = config.MIN_TEXT_LENGTH
        self.max_length = config.MAX_TEXT_LENGTH
        self.title_repeat = config.TITLE_REPEAT_TIMES
        
        # Token budget (approximate, 1 token ≈ 4 chars)
        self.max_tokens = 512
        self.title_budget_chars = 80      # ~20 tokens (10%)
        self.desc_budget_chars = 600      # ~150 tokens (30%)
        self.reviews_budget_chars = 1200  # ~300 tokens (60%)
        self.meta_budget_chars = 168      # ~42 tokens (10%)
    
    def process(self, df):
        """
        Main processing function
        
        Args:
            df: DataFrame with raw text columns
            
        Returns:
            df: DataFrame with 'combined_text' column added
        """
        print("\n[*] Creating combined text for each book...")
        
        # Create combined text
        df['combined_text'] = df.apply(self._combine_text, axis=1)
        
        # Calculate text statistics
        df['text_length'] = df['combined_text'].str.len()
        
        # Filter out too short/long texts
        original_len = len(df)
        df = df[
            (df['text_length'] >= self.min_length) & 
            (df['text_length'] <= self.max_length)
        ]
        
        if len(df) < original_len:
            print(f"[WARNING] Filtered out {original_len - len(df)} books (text too short/long)")
        
        # Display statistics
        self._print_stats(df)
        
        return df
    
    def _combine_text(self, row):
        """
        Combine multiple text columns into one with priority and token budget control.
        """
        parts = []
        
        # 1. TITLE (highest priority - repeat for weight)
        if pd.notna(row.get('title')):
            title = str(row['title']).strip()
            # Áp dụng giới hạn ký tự cho Title
            title_part = (title + ". ") * self.title_repeat
            parts.append(title_part[:self.title_budget_chars]) 
            
        # 2. METADATA (Categories and Authors)
        # Thêm Metadata trước để đảm bảo chúng luôn có mặt
        if pd.notna(row.get('categories')):
            parts.append(str(row['categories']).strip())
        if pd.notna(row.get('authors')):
            parts.append(f"Author: {str(row['authors']).strip()}")
            
        # 3. DESCRIPTION (Priority 2)
        desc_added = False
        if pd.notna(row.get('description')):
            desc = str(row['description']).strip()
            if len(desc) > 0:
                # Áp dụng giới hạn ký tự cho Description
                parts.append(desc[:self.desc_budget_chars])
                desc_added = True
        
        # 4. REVIEW TEXT (Priority 3 - Bổ sung hoặc Thay thế)
        # Nếu Description đã được thêm, Reviews là bổ sung.
        # Nếu Description không có, Reviews là nguồn thay thế cốt lõi.
        
        # Gộp 'aggregated_reviews' (giả sử là 'review_text' và 'review_summary' đã gộp)
        # Code đã sửa:
        review_content = []

    # 1. Gộp AGGREGATED REVIEWS
        if pd.notna(row.get('aggregated_reviews')):
            reviews = str(row['aggregated_reviews']).strip()
            if len(reviews) > 0:
                review_content.append(reviews)

        # 2. Gộp AGGREGATED SUMMARIES (Kiểm tra độc lập)
        if pd.notna(row.get('aggregated_summaries')):
            summaries = str(row['aggregated_summaries']).strip()
            if len(summaries) > 0:
                review_content.append(summaries)
                
        if review_content:
            review_full = " ".join(review_content)
            if len(review_full) > 0:
                # Áp dụng giới hạn ký tự cho Reviews (Phần lớn ngân sách)
                parts.append(review_full[:self.reviews_budget_chars])
                        
                # Combine all parts
                # Lưu ý: Cần làm phẳng và làm sạch trước khi trả về
        combined = " ".join(parts)
                
        return combined.strip()
    
    def _print_stats(self, df):
        """
        Print text length statistics with token estimates
        """
        print(f"\n[OK] Combined text created for {len(df)} books")
        print(f"\n[INFO] Text Length Statistics:")
        print(f"   Mean:   {df['text_length'].mean():.0f} chars (~{df['text_length'].mean()/4:.0f} tokens)")
        print(f"   Median: {df['text_length'].median():.0f} chars (~{df['text_length'].median()/4:.0f} tokens)")
        print(f"   Min:    {df['text_length'].min():.0f} chars")
        print(f"   Max:    {df['text_length'].max():.0f} chars (~{df['text_length'].max()/4:.0f} tokens)")
        print(f"   Std:    {df['text_length'].std():.0f} chars")
        
        # Token budget check
        avg_tokens = df['text_length'].mean() / 4
        if avg_tokens > 512:
            print(f"\n[WARNING] Average tokens ({avg_tokens:.0f}) exceeds SBERT limit (512)")
        else:
            print(f"\n[OK] Average tokens ({avg_tokens:.0f}) within SBERT limit (512)")
        
        # Show sample
        print(f"\n[SAMPLE] Combined text (first book):")
        sample = df['combined_text'].iloc[0]
        print(f"   Title portion: {sample[:100]}...")
        print(f"   Total length: {len(sample)} chars (~{len(sample)/4:.0f} tokens)")
        
        # Composition analysis
        if len(df) > 0:
            print(f"\n[INFO] Text Composition:")
            has_desc = (df['description'].notna() & (df['description'].str.len() > 0)).sum()
            has_reviews = (df.get('aggregated_reviews', pd.Series()).notna() & 
                          (df.get('aggregated_reviews', pd.Series()).str.len() > 0)).sum()
            print(f"   Books with description: {has_desc}/{len(df)} ({has_desc/len(df)*100:.1f}%)")
            print(f"   Books with reviews: {has_reviews}/{len(df)} ({has_reviews/len(df)*100:.1f}%)")


if __name__ == "__main__":
    # Test
    from src.preprocessing.data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_master_books()
    
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.process(df)
    
    print(f"\n[OK] Final DataFrame shape: {df_processed.shape}")