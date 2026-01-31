# 📚 Book Recommendation System

Hệ thống gợi ý sách thông minh sử dụng NLP + Hybrid Ranking + Cross-sell

##  Features
- **NLP Retrieval**: SBERT semantic search + NER + KeyBERT
- **Hybrid Ranking**: Content + Keywords + Entities + Impact Score
- **Cross-sell**: Item-Item similarity recommendations
- **Web UI**: Streamlit interactive demo

---

##  Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-team/book-recommender-system.git
cd book-recommender-system
```

### 2. Setup Environment

#### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Spacy model
python -m spacy download en_core_web_sm
```

#### Option B: Using Conda
```bash
conda env create -f environment.yml
conda activate nlp-book-rec
```

### 3. Download Data


## 📂 Data Structure

```
data/
├── raw/
│   ├── reviews.parquet           (36K rows)
│   └── books_details.parquet     (2.7K rows)
├── processed/
│   ├── master_books.csv          (2.7K unique books)
│   ├── ratings.csv               (28K ratings)
│   └── books_with_combined_text.csv
└── features/
    ├── book_embeddings.npy       (~500MB)
    ├── book_ids.pkl
    └── (other features...)
```

---

## 🔧 Development Workflow

### Person 1: NLP Retrieval
```bash
# Your tasks
python scripts/01_preprocess_and_generate_embeddings.py
python scripts/02_test_retrieval.py
python scripts/03_test_hybrid.py
```

### Person 2: Cross-sell
```bash
# Your tasks
python scripts/07_build_item_similarity.py
# Then develop: src/recommendation/cross_sell.py
```

### Person 3: Web UI
```bash
# Your tasks
streamlit run webapp/app.py

# During development (hot reload)
streamlit run webapp/app.py --server.runOnSave true
```

---

##  Testing

### Test NLP Retrieval
```bash
python scripts/02_test_retrieval.py
```

### Test Hybrid Scoring
```bash
python scripts/03_test_hybrid.py
# Select mode 2 for detailed results
```

### Test Web App
```bash
streamlit run webapp/app.py
# Open browser: http://localhost:8501
```

---

##  Project Structure
```
├── src/                  # Core modules
│   ├── preprocessing/    # Data loading & cleaning
│   ├── models/          # ML models (NLP, CF)
│   ├── recommendation/  # Recommendation logic
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
├── webapp/              # Streamlit UI
├── data/                # Data files (gitignored)
└── docs/                # Documentation
```

---

##  Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'src'`
```bash
# Add project root to PYTHONPATH
# Windows:
set PYTHONPATH=%CD%
# Mac/Linux:
export PYTHONPATH=$(pwd)
```

### Issue 2: `FileNotFoundError: data/processed/master_books.csv`
```bash
# Run data preparation first
python scripts/00_prepare_datasets.py
```

### Issue 3: Spacy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue 4: CUDA out of memory (khi generate embeddings)
```python
# In src/config.py, giảm batch size:
EMBEDDING_BATCH_SIZE = 8  # Default: 16
```

---

