import numpy as np


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.int32) / len(is_relevant)
    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.int32) / relevant_items.shape[0]

    return recall_score


def mean_avg_precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_map = 0.0

    n_users = URM_test.shape[0]
    num_eval = 0

    for user_id in range(n_users):
        if user_id % 10000 == 0 :
            print("User {} of {}".format(user_id,n_users))
        relevant_items = URM_test[user_id].indices
        #tracks in the test set per each playlist

        if len(relevant_items) > 0:
            recommended_item = recommender_object.recommend(user_id,at = 10)
            num_eval += 1

            cumulative_precision += precision(recommended_item, relevant_items)
            cumulative_recall += recall(recommended_item, relevant_items)
            cumulative_map += mean_avg_precision(recommended_item, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))
    return cumulative_map


def make_recommendations(recommender, target_playlists, URM_train):
    #recommender.fit(top_k= 50 , shrink=50)
    output_file = open("../data/sample_sumbission.csv", "w")
    output_file.write("playlist_id,track_ids\n")

    for playlist in target_playlists:
        recommendations = recommender.recommend(playlist,at=10)
        output_file.write(str(playlist) + ",")
        recommendations.tofile(output_file, sep=" ", format="%s")
        if playlist == target_playlists[-1]:
            break
        output_file.write('\n')
    output_file.close()
