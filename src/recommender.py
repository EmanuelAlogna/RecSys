import numpy as np


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

