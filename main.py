from utils.data_loader import load_data
from models.user_based import run_user_based
from models.item_based import run_item_based
from models.matrix_factorization import run_matrix_factorization

print("Loading MovieLens 100K dataset...")
df_train, matrix_train, df_test, matrix_test, item_id_to_title, movie_id_categories = load_data()
print("==================================================")
print("  Dataset: MovieLens 100K")
print(f"  Users  : {matrix_train.shape[0]}")
print(f"  Items  : {matrix_train.shape[1]}")
print(f"  Train interactions: {matrix_train.nnz}")
print(f"  Test  interactions: {matrix_test.nnz}")
print("==================================================")

# Ask for user ID
user_id_orig = int(input("Enter original user ID (1-610): "))
user_id = user_id_orig - 1
if user_id < 0 or user_id >= matrix_train.shape[0]:
    print("Invalid user ID. Exiting.")
    exit()

print(f"\n>>> Recommendations and Metrics for User {user_id_orig}")
print("\n>>> Model 1: User-Based Collaborative Filtering")
ub_p, ub_r = run_user_based(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, item_id_to_title, movie_id_categories)

print("\n>>> Model 2: Item-Based Collaborative Filtering")
ib_p, ib_r = run_item_based(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, item_id_to_title, movie_id_categories)

print("\n>>> Model 3: Matrix Factorization")
mf_p, mf_r = run_matrix_factorization(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, item_id_to_title, movie_id_categories)

print("\n======================================================================")
print("  FINAL COMPARISON SUMMARY  (Test Set — k=10)")
print("======================================================================")
print("  Model                               Precision     Recall")
print("  -----------------------------------------------------------------")
print(f"  User-Based CF                          {ub_p:.4f}     {ub_r:.4f}")
print(f"  Item-Based CF                          {ib_p:.4f}     {ib_r:.4f}")
print(f"  Matrix Factorization                   {mf_p:.4f}     {mf_r:.4f}")
print("======================================================================")