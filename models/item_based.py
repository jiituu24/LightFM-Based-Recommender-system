def run_item_based(df_train, matrix_train, df_test, matrix_test, user_id, user_id_orig, movie_id_to_title, movie_id_categories):
    from lightfm import LightFM
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = LightFM(loss='logistic')
    model.fit(matrix_train, epochs=10)

    item_embeddings = model.item_embeddings
    item_sim = cosine_similarity(item_embeddings)

    train_precision = 0.0
    train_recall = 0.0

    # precompute user ratings dict
    user_items_dict = df_train.groupby('user_id')[['item_id','rating']].apply(lambda x: list(zip(x['item_id'], x['rating']))).to_dict()

    test_users = df_test['user_id'].unique()
    all_precisions_test = []
    all_recalls_test = []

    for u in test_users:
        user_items = user_items_dict.get(u, [])
        scores = {}
        sim_sums = {}

        for (i, r_ui) in user_items:
            similar_items = np.argsort(-item_sim[i])[1:11]

            for j in similar_items:
                sim = item_sim[i][j]
                if sim <= 0:
                    continue
                scores[j] = scores.get(j, 0) + sim * r_ui
                sim_sums[j] = sim_sums.get(j, 0) + abs(sim)

        seen_items = set([i for (i, _) in user_items])
        final_scores = {}

        for item in scores:
            if sim_sums[item] > 0:
                final_scores[item] = scores[item] / sim_sums[item]

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
    print("  Results: Item-Based Collaborative Filtering")
    print("  Metric                     Train       Test")
    print("  ---------------------------------------------")
    print(f"  Precision@10              {train_precision:.4f}     {test_precision:.4f}")
    print(f"  Recall@10                 {train_recall:.4f}     {test_recall:.4f}")
    print("==================================================== ")

    user_items = user_items_dict.get(user_id, [])
    seen_items = set([i for (i, _) in user_items])

    scores = {}
    sim_sums = {}

    for (i, r_ui) in user_items:
        similar_items = np.argsort(-item_sim[i])[1:6]

        for j in similar_items:
            sim = item_sim[i][j]
            if sim <= 0:
                continue
            scores[j] = scores.get(j, 0) + sim * r_ui
            sim_sums[j] = sim_sums.get(j, 0) + abs(sim)

    final_scores = {}

    for item in scores:
        if sim_sums[item] > 0:
            final_scores[item] = scores[item] / sim_sums[item]

    sorted_items = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
    top_items = [item for item in sorted_items if item not in seen_items][:5]

    recommendations = [f"{movie_id_to_title[item]} ({item})" for item in top_items]

    print(f"Top 5 similar items for user {user_id_orig} (Item-Based CF): {recommendations}")

    return test_precision, test_recall