import numpy as np


class BPR_Sampling(object):

    def __init__(self, URM_train):
        self.URM_train = URM_train
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

    def sampleUser(self):
        while (True):

            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if (numSeenItems > 0 and numSeenItems < self.n_items):
                return user_id


    def sampleItemPair(self, user_id):
        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while (True):

            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id

