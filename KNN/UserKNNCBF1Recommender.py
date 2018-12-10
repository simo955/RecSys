#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF

import numpy as np

from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBF1Recommender(SimilarityMatrixRecommender, Recommender):
    """ userKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBF1Recommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, UCM, URM_train, sparse_weights=True):
        super(UserKNNCBF1Recommender, self).__init__()

        self.UCM = UCM.copy()
        
        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based




    def fit(self, topK=400, shrink=20, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.UCM = self.UCM.astype(np.float32)
            self.UCM = okapi_BM_25(self.UCM)

        elif feature_weighting == "TF-IDF":
            self.UCM = self.UCM.astype(np.float32)
            self.UCM = TF_IDF(self.UCM)


        similarity = Compute_Similarity(self.UCM.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

