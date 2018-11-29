import pandas as pd
import scipy.sparse as sps
import numpy as np
import time
from scipy.sparse import coo_matrix, hstack

def load_train(file_path):
    file = pd.read_csv(file_path)
    URM_row = list(file["playlist_id"])
    URM_column = list(file["track_id"])
    values = np.ones(len(URM_row))
    URM_all = sps.coo_matrix((values, (URM_row, URM_column)),dtype=int)
    return URM_all


def load_track_attributes(file_path):
    file = pd.read_csv(file_path)
    ICM_row = list(file["track_id"])
    ICM_column1 = list(file["album_id"])
    ICM_column2 = list(file["artist_id"])

    ones = np.ones(len(ICM_row))

    a = sps.coo_matrix((ones, (ICM_row, ICM_column1)))
    b = sps.coo_matrix((ones, (ICM_row, ICM_column2)))
    c = hstack([a, b])
    ICM_all = c.tocsr()

    return ICM_all


def split_dataset(URM_all,train_test_ratio =0.8):
    target_playlists = pd.read_csv('../data/target_playlists.csv');
    URM_test = URM_all.copy();
    URM_train = URM_all.copy();
    sequential_playlists = pd.read_csv('../data/train_sequential.csv')
    sequential_playlists_index = list(target_playlists['playlist_id'])[0:5000];

    # This snippet of code take the 20% of sequential playlist
    # The algorithm set to 0 the 80% of the track index in a playlist.
    # then it creates a mask, comparing all the tracks in a playlist and the track
    # that have to be selected in order to create the test set.
    # The mask becomes the new array of data of the csr matrix
    for i in sequential_playlists_index:
        target_row = i

        #Tracks is a list containing all the tracks contained in a sequential playlist
        tracks = sequential_playlists[sequential_playlists['playlist_id'].eq(i)]
        tracks = list(tracks['track_id'])

        row_start = URM_test.indptr[target_row]
        row_end = URM_test.indptr[target_row + 1]
        row_columns = URM_test.indices[row_start:row_end]
        data = URM_test.data[row_start: row_end]
        size = int(np.ceil(len(data) * train_test_ratio))

        tracks = np.array(tracks)
        tracks[0:size] = 0
        mask = np.in1d(row_columns, tracks)

        URM_test.data[row_start: row_end] = mask

    random_playlist = list(target_playlists['playlist_id'])[5000:];
    for i in random_playlist:
        target_row = i
        row_start = URM_test.indptr[target_row]
        row_end = URM_test.indptr[target_row + 1]
        data = URM_test.data[row_start: row_end]
        test_mask = np.random.choice([True, False], len(data), p=[1-train_test_ratio, train_test_ratio])
        data = data * test_mask
        URM_test.data[row_start: row_end] = data

    all_playlists = pd.read_csv('../data/train.csv')
    all_playlists = list(set(all_playlists['playlist_id']))
    target_playlists = list(target_playlists['playlist_id'])

    mask = np.in1d(all_playlists, target_playlists)
    mask = np.logical_not(mask)
    playlist_not_target = all_playlists * mask
    playlist_not_target = list(set(playlist_not_target))

    for i in playlist_not_target:
        target_row = i
        row_start = URM_test.indptr[target_row]
        row_end = URM_test.indptr[target_row + 1]
        data = URM_test.data[row_start: row_end]
        data = 0
        URM_test.data[row_start: row_end] = data

    # Build the training matrix
    for i in target_playlists:
        target_row = i
        row_start = URM_test.indptr[target_row]
        row_end = URM_test.indptr[target_row + 1]
        data = URM_test.data[row_start: row_end]
        data = np.logical_not(data)
        URM_train.data[row_start:row_end] = data

    URM_train.eliminate_zeros()
    URM_test.eliminate_zeros()


    return URM_train, URM_test





def split_dataset_random(URM_all):
    num_interactions = URM_all.nnz
    file = pd.read_csv('../data/train.csv')
    list_playlist = file['playlist_id'].T
    list_tracks = file['track_id'].T
    ratings = np.ones(num_interactions)
    train_mask = np.random.choice([True,False], num_interactions, p=[0.8, 0.2])
    URM_train = sps.coo_matrix((ratings[train_mask], (list_playlist[train_mask], list_tracks[train_mask])))
    URM_train = URM_train.tocsr()
    train_mask = np.logical_not(train_mask)
    URM_test = sps.coo_matrix((ratings[train_mask], (list_playlist[train_mask], list_tracks[train_mask])))
    URM_test = URM_test.tocsr()
    return URM_train,URM_test


class DataReader(object):

    def __init__(self, file_path, split_train_test=False, train_test_ratio=0.8,
                 build_validation = False,train_validation_ratio=0.9 ):
        self.URM_all = load_train("../data/" + file_path)
        self.URM_all = self.URM_all.tocsr()

        if split_train_test:
            [self.URM_train , self.URM_test] = split_dataset(self.URM_all,train_test_ratio)

        if build_validation:
            [self.URM_train,self.URM_validation] = split_dataset(self.URM_train,train_validation_ratio)


    def build_icm(self, file_path):

        ICM_all = load_track_attributes("../data/" + file_path)
        return ICM_all