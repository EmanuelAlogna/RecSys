import numpy as np
from sklearn.preprocessing import minmax_scale

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
    #print('is relevant {}'.format(is_relevant))
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    #print('p_at_k : {}'.format(p_at_k))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    # print('map score {}'.format(map_score))
    # print('\n')
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
        # print('user {}'.format(user_id))
        # print('relevant items: {}'.format(relevant_items))
        if len(relevant_items) > 0:

            recommended_item = recommender_object.recommend2(user_id,at = 10)
            if user_id > 0 and user_id < 100:
                print('\n')
                print('relevant items')
                print(relevant_items)
                print('recommended items')
                print(recommended_item)
                print(np.in1d(relevant_items,recommended_item))

            num_eval += 1
            if user_id == 7 or user_id==25 or user_id==50424:
                print(recommended_item)
            cumulative_precision += precision(recommended_item, relevant_items)
            cumulative_recall += recall(recommended_item, relevant_items)
            #print("Recommended items: {}".format(recommended_item))
            cumulative_map += mean_avg_precision(recommended_item, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))
    return cumulative_map


def make_recommendations(recommender, target_playlists):
    #recommender.fit(top_k= 50 , shrink=50)
    output_file = open("../data/sample_submission.csv", "w")
    output_file.write("playlist_id,track_ids\n")

    for playlist in target_playlists:
        recommendations = recommender.recommend2(playlist,at=10)
        if playlist==7 or playlist==25:
            print(recommendations)
        output_file.write(str(playlist) + ",")
        recommendations.tofile(output_file, sep=" ", format="%s")
        if playlist == target_playlists[-1]:
            break
        output_file.write('\n')
    output_file.close()



def evaluate_algorithm_hybrid(URM_test, recommender1,recommender2,target_playlist,alpha =0.5,):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_map = 0.0
    alpha = alpha
    n_users = URM_test.shape[0]
    num_eval = 0
    URM = recommender1.getURM() + recommender2.getURM()
    for user_id in range(n_users):
        if user_id % 10000 == 0 :
            print("User {} of {}".format(user_id,n_users))
        relevant_items = URM_test[user_id].indices
        #tracks in the test set per each playlist
        # print('user {}'.format(user_id))
        # print('relevant items: {}'.format(relevant_items))
        if len(relevant_items) > 0:
            scores1 = recommender1.get_scores(user_id,exclude_seen=True)
            scores2 = recommender2.get_scores(user_id,exclude_seen=True)


            scores = scores1+scores2
            recommended_item = scores.argsort()[::-1]
            recommended_item = recommended_item[:10]


            num_eval += 1

            cumulative_precision += precision(recommended_item, relevant_items)
            cumulative_recall += recall(recommended_item, relevant_items)
            #print("Recommended items: {}".format(recommended_item))
            cumulative_map += mean_avg_precision(recommended_item, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))
    return cumulative_map


def evaluate_algorithm_hybrid2(URM_test, recommender1,recommender2,recommender3,alpha =0.5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_map = 0.0
    alpha = alpha
    n_users = URM_test.shape[0]
    num_eval = 0
    URM = recommender1.getURM() + recommender2.getURM()
    for user_id in range(n_users):
        if user_id % 10000 == 0 :
            print("User {} of {}".format(user_id,n_users))
        relevant_items = URM_test[user_id].indices
        #tracks in the test set per each playlist
        # print('user {}'.format(user_id))
        # print('relevant items: {}'.format(relevant_items))

        if len(relevant_items) > 0:
            scores1 = recommender1.get_scores(user_id,exclude_seen=True)
            scores2 = recommender2.get_scores(user_id,exclude_seen=True)
            scores3 = recommender3.get_scores(user_id,exclude_seen=True)
            scores = scores1+scores2+scores3

            recommended_item = scores.argsort()[::-1]
            recommended_item = recommended_item[:10]


            num_eval += 1

            cumulative_precision += precision(recommended_item, relevant_items)
            cumulative_recall += recall(recommended_item, relevant_items)
            #print("Recommended items: {}".format(recommended_item))
            cumulative_map += mean_avg_precision(recommended_item, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))
    return cumulative_map





def filter_seen(URM_train, user_id, scores):

    start_pos = URM_train.indptr[user_id]
    end_pos = URM_train.indptr[user_id+1]

    user_profile = URM_train.indices[start_pos:end_pos]

    scores[user_profile] = -np.inf

    return scores


def make_recommendations2(recommender1,recommender2, target_playlists):
    #recommender.fit(top_k= 50 , shrink=50)
    output_file = open("../data/sample_submission.csv", "w")
    output_file.write("playlist_id,track_ids\n")

    for playlist in target_playlists:
        scores1 = recommender1.get_scores(playlist, exclude_seen=True)
        scores2 = recommender2.get_scores(playlist, exclude_seen=True)

        scores = scores1 + scores2
        recommended_item = scores.argsort()[::-1]
        recomendations = recommended_item[:10]


        output_file.write(str(playlist) + ",")
        recomendations.tofile(output_file, sep=" ", format="%s")
        if playlist == target_playlists[-1]:
            break
        output_file.write('\n')
    output_file.close()
