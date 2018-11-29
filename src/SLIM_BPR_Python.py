import numpy as np
import time
from src.Recommender_utils import similarityMatrixTopK
from src.Cython.SLIM_BPR_Cython2 import SLIM_BPR_Cython_Epoch
import scipy.sparse as sps

class SLIM_BPR_Recommender(object):

    def __init__(self, URM,learning_rate = 0.1,S_init = None):
        self.URM = URM

        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        self.URM.eliminate_zeros()
        self.learning_rate = learning_rate

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        self.S_init = S_init

        for user_id in range(self.n_users):

            start_pos = self.URM.indptr[user_id]
            end_pos = self.URM.indptr[user_id + 1]

            if len(self.URM.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)



    def sampleTriplet(self):
        # By randomly selecting a user in this way we could end up 
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = self.URM[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id


    def fit(self, learning_rate=0.1, epochs=10,lambda_i = 0.0, lambda_j = 0.0,sgd_mode='sgd'):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode
        cython_epoch = SLIM_BPR_Cython_Epoch(self.URM, self.eligibleUsers,
                 learning_rate = self.learning_rate,topK=200, sgd_mode=self.sgd_mode,S_init = self.S_init,
                                             li_reg = lambda_i ,lj_reg = lambda_j)

        for numEpoch in range(self.epochs):
            print("Epoch {} of {}".format(numEpoch,self.epochs))
            cython_epoch.epochIteration_Cython()

        self.similarity_matrix = cython_epoch.get_S()
        #self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=100,forceSparseOutput=True,inplace=True).T


    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]
        user_ratings = self.URM.data[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]

        relevant_weights = self.similarity_matrix[user_profile]
        scores = relevant_weights.T.dot(user_ratings)

        # user_profile = self.URM[user_id]
        # scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
