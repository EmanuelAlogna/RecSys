from src.Recommender_utils import *

class HybridRecommender(object):

    def __init__(self, URM_train, sim1, sim2, sparse_weights=False):

        if sim1.shape != sim2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different "
                             "size, S1 is {}, S2 is {}".format(sim1.shape, sim2.shape))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(sim1.copy(), 'csr')
        self.Similarity_2 = check_matrix(sim2.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights

    def fit(self, topK=100, alpha = 0.5):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)

        return self.W

    def recommend2(self, user_id, at=None, exclude_seen = True):
        user_profile = self.URM_train[user_id]
        scores = user_profile.dot(self.W).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]
        user_profile = self.URM_train.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores