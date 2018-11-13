import numpy as np
from Compute_Similarity_Python import *


class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[1]

    def recommend(self, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


class TopPopularRecommender(object):

    def fit(self, URM_train):
        item_popularity = (URM_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        self.popular_items = np.argsort(item_popularity)

        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, at=10):
        recommended_items = self.popular_items[0:at]

        return recommended_items


class ItemCBFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink, topK=top_k)

        print(top_k)
        print(shrink)
        self.sim_matrix = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen = True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.sim_matrix).toarray().ravel()

        ranking = scores.argsort()[::-1]

        return ranking[:at]
