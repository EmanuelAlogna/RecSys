import pandas as pd
import scipy.sparse as sps
import numpy as np


def load_train(file_path):
    file = pd.read_csv(file_path)
    URM_row = list(file["playlist_id"])
    URM_column = list(file["track_id"])
    values = np.ones(len(URM_row))

    URM_all = sps.coo_matrix((values, (URM_row, URM_column)))

    return URM_all


def load_track_attributes(file_path):
    file = pd.read_csv(file_path)
    ICM_row = list(file["track_id"])
    ICM_column1 = list(file["album_id"])
    ICM_column2 = list(file["artist_id"])
    ICM_column3 = list(file["duration_sec"])

    ICM_all = sps.lil_matrix((len(ICM_row), 3), dtype=int)  # empty 20635x3 sparse matrix
    for i in range(0, len(ICM_row)):
        ICM_all[i, 0] = ICM_column1[i]
        ICM_all[i, 1] = ICM_column2[i]
        ICM_all[i, 2] = ICM_column3[i]

    return ICM_all


class DataReader(object):

    def __init__(self, file_path, split_train_test=False, train_test_ratio=0.8):
        self.URM_all = load_train("./data/" + file_path)

        if split_train_test:
            num_interactions = len(self.URM_all.data)
            train_mask = np.random.choice([True, False], num_interactions, p=[train_test_ratio, 1 - train_test_ratio])
            self.URM_train = sps.coo_matrix(
                (self.URM_all.data[train_mask], (self.URM_all.row[train_mask], self.URM_all.col[train_mask])))
            self.URM_train = self.URM_train.tocsr()

            test_mask = np.logical_not(train_mask)
            self.URM_test = sps.coo_matrix(
                (self.URM_all.data[test_mask], (self.URM_all.row[test_mask], self.URM_all.col[test_mask])))
            self.URM_test = self.URM_test.tocsr()

        self.URM_all = self.URM_all.tocsr()

    def build_icm(self, file_path):

        ICM_all = load_track_attributes("./data/" + file_path)
        ICM_all = ICM_all.tocsr()
        return ICM_all
