# LightFM-Based Movie Recommender System

A comprehensive movie recommendation system that implements and compares three different collaborative filtering approaches using the MovieLens 100K dataset.

## 📋 Overview

This project demonstrates building and evaluating multiple recommendation algorithms:
- **User-Based Collaborative Filtering**: Finds similar users and recommends items they liked
- **Item-Based Collaborative Filtering**: Finds similar items based on user ratings
- **Matrix Factorization**: Uses LightFM's embeddings to learn latent features

All three models are trained on the same dataset and their performance is compared using Precision@10 and Recall@10 metrics.

## 📊 Dataset

- **Source**: MovieLens 100K
- **Users**: 610
- **Movies**: 9,742
- **Ratings**: 100,000 user-movie interactions
- **Train/Test Split**: 90% train, 10% test

### Data Files
- `data/ratings.csv`: User ratings (userId, movieId, rating, timestamp)
- `data/movies.csv`: Movie metadata (movieId, title, genres)

## 🏗️ Project Structure

```
.
├── main.py                          # Main execution script
├── requirements.txt                 # Python dependencies
├── data/
│   ├── ratings.csv                  # User-movie ratings
│   └── movies.csv                   # Movie titles and genres
├── models/
│   ├── __init__.py
│   ├── user_based.py               # User-based collaborative filtering
│   ├── item_based.py               # Item-based collaborative filtering
│   └── matrix_factorization.py     # LightFM matrix factorization
└── utils/
    ├── __init__.py
    └── data_loader.py              # Data loading and preprocessing
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone or download the repository**:
```bash
cd lightfm-recommender
```

2. **Create and activate a virtual environment** (recommended):
```bash
# Windows
python -m venv "Recommender system"
"Recommender system\Scripts\activate"

# macOS/Linux
python3 -m venv "Recommender system"
source "Recommender system/bin/activate"
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the System

Execute the main script:
```bash
python main.py
```

**Expected Output**:
1. Dataset statistics (users, items, interactions)
2. Prompt to enter a user ID (1-610)
3. Top 5 recommendations from each model
4. Precision@10 and Recall@10 metrics for all models
5. Summary comparison table

**Example**:
```
Loading MovieLens 100K dataset...
==================================================
  Dataset: MovieLens 100K
  Users  : 610
  Items  : 9,742
  Train interactions: 90,000
  Test interactions: 10,000
==================================================
Enter original user ID (1-610): 123

>>> Recommendations and Metrics for User 123
...
```

## 📈 Model Comparison

Each model is evaluated on the test set using:

- **Precision@10**: What fraction of the top 10 recommendations were relevant?
  - Formula: (# of relevant items in top 10) / 10
  
- **Recall@10**: What fraction of all relevant items appear in top 10?
  - Formula: (# of relevant items in top 10) / (total relevant items for user)

### Results Summary
The final comparison table shows:
- User-Based CF: Precision and Recall@10
- Item-Based CF: Precision and Recall@10
- Matrix Factorization: Precision and Recall@10

## 🔧 Technical Details

### User-Based Collaborative Filtering
- Computes cosine similarity between user embeddings (from LightFM)
- Finds 10 most similar users
- Recommends items rated highly by similar users
- Uses mean-centered ratings for normalization

### Item-Based Collaborative Filtering
- Computes cosine similarity between item embeddings (from LightFM)
- Recommends items similar to those the user already rated
- Weights recommendations by item similarity and original rating
- Avoids re-recommending already-rated items

### Matrix Factorization (LightFM)
- Uses logistic loss function
- Trains embeddings for both users and items
- Learns latent factors representing user preferences and item characteristics
- Default: 10 epochs of training

## 📦 Dependencies

- **lightfm**: Fast learning-to-rank model for implicit and explicit feedback
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scipy**: Scientific computing (sparse matrices)
- **scikit-learn**: Machine learning utilities (cosine similarity, train-test split)

See `requirements.txt` for specific versions.

## 🎯 Key Features

✅ Multiple recommendation algorithms for comparison  
✅ Train/test evaluation with standard metrics  
✅ Efficient sparse matrix representations  
✅ User-friendly command-line interface  
✅ Detailed performance metrics and summaries  
✅ Modular code structure for easy extension  

## 🔍 How to Extend

### Add a New Model
1. Create a new file in `models/` directory
2. Implement a function `run_new_model()` with signature:
```python
def run_new_model(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, item_id_to_title, movie_id_categories):
    # Your implementation
    return precision_at_k, recall_at_k
```
3. Import and call it in `main.py`

### Modify Evaluation Metrics
Edit the precision@k and recall@k values in each model file (currently set to k=10).

### Use Different Data
Replace `data/ratings.csv` and `data/movies.csv` with your own data, ensuring the same column names.

## 📝 Notes

- User IDs are 1-indexed in the original dataset but converted to 0-indexed internally
- Items with rating 0 are not included in recommendations
- The test set is only used for final evaluation, not during training
- Random state is fixed (42) for reproducibility

## 🐛 Troubleshooting

**"Invalid user ID" error**:
- Enter a user ID between 1 and 610

**"No module named 'lightfm'"**:
- Ensure virtual environment is activated and dependencies are installed: `pip install -r requirements.txt`

**Memory issues with large datasets**:
- The sparse matrix representation keeps memory usage efficient
- Consider using a smaller subset if needed

## 📄 License

This project uses the MovieLens 100K dataset. See the original [MovieLens dataset](https://grouplens.org/datasets/movielens/) for licensing information.

## 👤 Author Notes

This is an educational project demonstrating practical implementation of recommendation systems. It serves as a great starting point for learning collaborative filtering techniques and comparing different algorithmic approaches.

---

**Last Updated**: April 2026
