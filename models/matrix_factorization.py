def run_matrix_factorization(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, movie_id_to_title, movie_id_categories):
    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k, recall_at_k
    import numpy as np

    model = LightFM(loss='logistic')
    model.fit(matrix_train, epochs=10)

    train_precision = precision_at_k(model, matrix_train, k=10).mean()
    train_recall = recall_at_k(model, matrix_train, k=10).mean()

    test_precision = precision_at_k(model, matrix_test, k=10).mean()
    test_recall = recall_at_k(model, matrix_test, k=10).mean()

    print("==================================================== ")
    print("  Results: Matrix Factorization")
    print("  Metric                     Train       Test")
    print("  ---------------------------------------------")
    print(f"  Precision@10              {train_precision:.4f}     {test_precision:.4f}")
    print(f"  Recall@10                 {train_recall:.4f}     {test_recall:.4f}")
    print("==================================================== ")

    n_items = matrix_train.shape[1]
    scores = model.predict(user_id, np.arange(n_items))

    seen_items = set(df_train[df_train['user_id'] == user_id]['item_id'])

    sorted_items = np.argsort(-scores)
    top_items = [item for item in sorted_items if item not in seen_items][:5]

    recommendations = [f"{movie_id_to_title[item]} ({item})" for item in top_items]

    print(f"Top 5 recommendations for user {user_id_orig} (Matrix Factorization): {recommendations}")

    return test_precision, test_recall