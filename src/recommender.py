import numpy as np
from src.Compute_Similarity_Python import *
import pyximport
pyximport.install()
from src.Cython.Cosine_Similarity_Cython import Cosine_Similarity


class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[1]

    def recommend(self, at=10):
        recommended_items = np.random.choice(self.numItems, at)

        return recommended_items


class TopPopularRecommender(object):

    def fit(self, URM):
        item_popularity = (URM > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        self.popular_items = np.argsort(item_popularity)

        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, at=10):
        recommended_items = self.popular_items[0:at]

        return recommended_items


class ItemCBFKNNRecommender(object):

    def __init__(self,URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit(self,top_k = 50, shrink = 100, normalize = True, similarity = 'cosine'):
        # similarity_object = Compute_Similarity_Python(self.ICM.T, shrink = shrink, topK= top_k,
        #                                               normalize= normalize, similarity = similarity)
        #
        # self.sim_matrix = similarity_object.compute_similarity()


        similarity_object = Cosine_Similarity(self.ICM.T, top_k, shrink)
        self.sim_matrix = similarity_object.compute_similarity()
        return self.sim_matrix

    def recommend(self, user_id , at = None , exclude_seen =True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.sim_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]
        return ranking[:at]


    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]
        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

class UserBasedCollaborativeRS(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        # RECCOMENDATION USING COSINE OF SIMILARITY IMPLEMENTED WITH PYTHON
        # similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink, topK=top_k,
        #                                               normalize=normalize, similarity=similarity)
        #
        #
        # self.sim_matrix = similarity_object.compute_similarity()

        # RECCOMENDATION USING COSINE OF SIMILARITY IMPLEMENTED WITH CYTHON
        similarity_object = Cosine_Similarity(self.URM.T, top_k, shrink)
        self.sim_matrix = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen= True):

        scores = self.sim_matrix[user_id].dot(self.URM).toarray().ravel()
        ranking = scores.argsort()[::-1]
        return ranking[:at]

class ItemBasedCollaborativeRS(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, top_k=50, shrink=100, normalize=True, similarity='cosine'):
        #RECCOMENDATION USING COSINE OF SIMILARITY IMPLEMENTED WITH PYTHON
        # similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink, topK=top_k,
        #                                               normalize=normalize, similarity=similarity)
        #
        #
        # self.sim_matrix = similarity_object.compute_similarity()

        # RECOMMENDATION USING COSINE OF SIMILARITY IMPLEMENTED WITH CYTHON
        similarity_object = Cosine_Similarity(self.URM, top_k, shrink)
        self.sim_matrix = similarity_object.compute_similarity()
        return self.sim_matrix

    def recommend2(self, user_id, at=None, exclude_seen = True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.sim_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]
        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores