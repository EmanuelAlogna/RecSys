import pandas as pd
import scipy.sparse as sps
import numpy as np

def load_csv_into_sparse(file_path):
    file = pd.read_csv(file_path)
    URM_row_name = file.keys()[0]
    URM_row = list(file[URM_row_name])
    URM_column_name = file.keys()[1]
    URM_column = list(file[URM_column_name])
    values = np.ones(len(URM_row))
    URM_all = sps.coo_matrix((values, (URM_row, URM_column)))
    return URM_all

class DataReader(object):

    def __init__(self,file_path,split_train_test=False,train_test_ratio=0.8):

        self.URM_all = load_csv_into_sparse("./data/"+file_path)

        if split_train_test:
            num_interactions = len(self.URM_all.data)
            train_mask = np.random.choice([True,False],num_interactions,p=[train_test_ratio,1-train_test_ratio])
            self.URM_train = sps.coo_matrix((self.URM_all.data[train_mask], (self.URM_all.row[train_mask],self.URM_all.col[train_mask])))
            self.URM_train = self.URM_train.tocsr()

            test_mask = np.logical_not(train_mask)
            self.URM_test = sps.coo_matrix((self.URM_all.data[test_mask],(self.URM_all.row[test_mask],self.URM_all.col[test_mask])))
            self.URM_test = self.URM_test.tocsr()


        self.URM_all = self.URM_all.tocsr()

