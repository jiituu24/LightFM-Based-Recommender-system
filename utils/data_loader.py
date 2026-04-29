import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("data/ratings.csv")

    df['user_id'] = df['userId'].astype("category").cat.codes
    df['item_id'] = df['movieId'].astype("category").cat.codes
    movie_id_categories = df['movieId'].astype("category").cat.categories
    user_id_categories = df['userId'].astype("category").cat.categories

    # Compute full dimensions
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1

    # Split into train and test
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    matrix_train = coo_matrix((df_train['rating'], (df_train['user_id'], df_train['item_id'])), shape=(n_users, n_items))
    matrix_test = coo_matrix((df_test['rating'], (df_test['user_id'], df_test['item_id'])), shape=(n_users, n_items))

    # Load movie titles - create mapping from internal item_id to movieId to title
    movies_df = pd.read_csv("data/movies.csv")
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # Create mapping from internal item_id code to movieId to title
    item_id_to_movieid = {i: movie_id_categories[i] for i in range(len(movie_id_categories))}
    item_id_to_title = {i: movie_id_to_title.get(movie_id_categories[i], 'Unknown') for i in range(len(movie_id_categories))}

    return df_train, matrix_train, df_test, matrix_test, item_id_to_title, movie_id_categories