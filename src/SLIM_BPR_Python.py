from src.BPR_Sampling import *
import time
import scipy.special
import scipy.sparse.csr


class SLIM_BPR_Python(BPR_Sampling):

    def __init__(self, URM_train):

        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

    def fit(self, epochs=30, learning_rate=0.05, topK=False, lambda_i=0.0025, lambda_j=0.00025):


        self.S = np.zeros((self.n_items, self.n_items)).astype('float32')
        self.topK = topK
        self.learning_rate = learning_rate
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j

        for currentEpoch in range(epochs):
            print(currentEpoch)
            self.epoch_iteration()

        self.W = scipy.sparse.csr_matrix(self.S.T)

    def recommend(self, user_id, at=None, exclude_seen = True):
        user_profile = self.URM_train[user_id]
        scores = user_profile.dot(self.W).toarray().ravel()

        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def epoch_iteration(self):

        u, i, j = self.sampleTriple()

        self.update_weights_loop(u, i, j)

        # self.S[np.arange(0, self.n_items), np.arange(0, self.n_items)] = 0.0

    def update_weights_loop(self, u, i, j):

        x_ui = self.S[i]
        x_uj = self.S[j]

        x_uij = x_ui - x_uj

        sigmoid = scipy.special.expi(-x_uij)

        delta_i = sigmoid - self.lambda_i * self.S[i]
        delta_j = -sigmoid - self.lambda_j * self.S[j]

        self.S[i] += self.learning_rate * delta_i
        self.S[j] += self.learning_rate * delta_j

