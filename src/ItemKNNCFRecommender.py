#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from src.Recommender2 import Recommender
from src.Recommender_utils import check_matrix
from src.SimilarityMatrixRecommender import SimilarityMatrixRecommender


from src.Cython.Cosine_Similarity_Cython import *


class ItemKNNCFRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(ItemKNNCFRecommender, self).__init__()

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        similarity = Cosine_Similarity(self.URM_train, shrink=shrink, topK=topK, normalize=normalize, mode = similarity)


        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

