# test_combine_text.py
import pandas as pd
from src.preprocessing.text_cleaner import TextPreprocessor

# Load master books
df = pd.read_csv("data/processed/master_books.csv")

# Get first book
first_book = df.iloc[0]

print("=" * 80)
print("BEFORE COMBINE")
print("=" * 80)
print(f"Title: {first_book['title']}")
print(f"Description: {first_book['description']}")
print(f"Reviews (first 100): {str(first_book['aggregated_reviews'])[:100]}")

# Create preprocessor
preprocessor = TextPreprocessor()

# Test combine_text on first book
combined = preprocessor._combine_text(first_book)

print("\n" + "=" * 80)
print("AFTER COMBINE")
print("=" * 80)
print(f"Combined text: {combined}")
print(f"Length: {len(combined)}")

# Check components
print("\n" + "=" * 80)
print("CHECK COMPONENTS")
print("=" * 80)
print(f"Has title: {'Its Only Art' in combined}")
print(f"Has description: {False if pd.isna(first_book['description']) else 'Philip Nel' in combined}")
print(f"Has reviews: {'This is only for' in combined or 'If people' in combined}")