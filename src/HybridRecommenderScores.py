import numpy as np

class HybridRecommenderScores(object):

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        self.URM_train = URM_train
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2


    def fit(self, alpha=0.5):
        self.alpha = alpha


    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id+1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


    def recommend2(self, user_id, at=None, exclude_seen=True):
        item_weights_1 = self.Recommender_1.get_scores(user_id,exclude_seen=False)
        item_weights_2 = self.Recommender_2.get_scores(user_id,exclude_seen=False)

        scores = item_weights_1 +0.3*item_weights_2
        if exclude_seen:
            item_weights = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def get_scores(self, user_id, at=None, exclude_seen=True):
        item_weights_1 = self.Recommender_1.get_scores(user_id, exclude_seen=False)
        item_weights_2 = self.Recommender_2.get_scores(user_id, exclude_seen=False)

        scores = item_weights_1 +  item_weights_2


        return scores