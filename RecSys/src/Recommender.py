import pandas as pd
import scipy.sparse as sps
import numpy as np

class RandomRecommender(object):

    def fit(self, URM_train):
        self.numItems = URM_train.shape[1]


    def recommend(self, at=10):
        recommended_items = np.random.choice(self.numItems,at)

        return recommended_items

class TopPopularRecommender(object):

    def fit(self):
        pass

    def recommend(self):
        pass
