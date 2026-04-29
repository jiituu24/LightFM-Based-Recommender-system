def run_user_based(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, movie_id_to_title, movie_id_categories):
    from lightfm import LightFM
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = LightFM(loss='logistic')
    model.fit(matrix_train, epochs=10)

    user_embeddings = model.user_embeddings
    user_sim = cosine_similarity(user_embeddings)

    train_precision = 0.0
    train_recall = 0.0

    # precompute user means
    user_mean = df_train.groupby('user_id')['rating'].mean().to_dict()
    test_users = df_test['user_id'].unique()
    all_precisions_test = []
    all_recalls_test = []

    for u in test_users:
        similar_users = np.argsort(-user_sim[u])[1:11]
        scores = {}
        sim_sums = {}

        for v in similar_users:
            sim = user_sim[u][v]
            if sim <= 0:
                continue
            user_v_ratings = df_train[df_train['user_id'] == v]

            for _, row in user_v_ratings.iterrows():
                item = row['item_id']
                rating = row['rating']
                mean_v = user_mean.get(v, 0)

                scores[item] = scores.get(item, 0) + sim * (rating - mean_v)
                sim_sums[item] = sim_sums.get(item, 0) + abs(sim)

        seen_items = set(df_train[df_train['user_id'] == u]['item_id'])
        final_scores = {}

        for item in scores:
            if sim_sums[item] > 0:
                final_scores[item] = user_mean.get(u, 0) + scores[item] / sim_sums[item]

        sorted_items = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        top_items = [item for item in sorted_items if item not in seen_items][:10]

        actual_items = set(df_test[df_test['user_id'] == u]['item_id'])
        hits = len(set(top_items) & actual_items)
        precision = hits / 10
        recall = hits / len(actual_items) if len(actual_items) > 0 else 0

        all_precisions_test.append(precision)
        all_recalls_test.append(recall)

    test_precision = np.mean(all_precisions_test)
    test_recall = np.mean(all_recalls_test)

    print("==================================================== ")
    print("  Results: User-Based Collaborative Filtering")
    print("  Metric                     Train       Test")
    print("  ---------------------------------------------")
    print(f"  Precision@10              {train_precision:.4f}     {test_precision:.4f}")
    print(f"  Recall@10                 {train_recall:.4f}     {test_recall:.4f}")
    print("==================================================== ")

    user_embeddings = model.user_embeddings
    user_sim = cosine_similarity(user_embeddings)
    seen_items = set(df_train[df_train['user_id'] == user_id]['item_id'])
    similar_users = np.argsort(-user_sim[user_id])[1:6]

    scores = {}
    sim_sums = {}

    for v in similar_users:
        sim = user_sim[user_id][v]
        if sim <= 0:
            continue
        user_v_ratings = df_train[df_train['user_id'] == v]

        for _, row in user_v_ratings.iterrows():
            item = row['item_id']
            rating = row['rating']
            mean_v = user_mean.get(v, 0)

            scores[item] = scores.get(item, 0) + sim * (rating - mean_v)
            sim_sums[item] = sim_sums.get(item, 0) + abs(sim)

    final_scores = {}

    for item in scores:
        if sim_sums[item] > 0:
            final_scores[item] = user_mean.get(user_id, 0) + scores[item] / sim_sums[item]

    sorted_items = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    top_items = [item for item in sorted_items if item not in seen_items][:5]

    recommendations = [f"{movie_id_to_title[item]} ({item})" for item in top_items]

    print(f"Top 5 recommended items for user {user_id_orig} (User-Based CF): {recommendations}")

    return test_precision, test_recall