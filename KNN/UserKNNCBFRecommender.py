#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""


from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBFRecommender(SimilarityMatrixRecommender, Recommender):
    """ UserCBFKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBFRecommender"

    def __init__(self, URM_train, sparse_weights=True):
        super(UserKNNCBFRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train, 'csr')

        self.dataset = None

        self.sparse_weights = sparse_weights

        self.compute_item_score = self.compute_score_user_based



    def fit(self, UCM_matrix, topK=400, shrink=20, similarity='cosine',normalize=True, **similarity_args):
       
        self.topK = topK
        self.shrink = shrink

        similarity = Compute_Similarity(UCM_matrix.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)

        if self.sparse_weights:
            self.W_sparse = similarity.compute_similarity()
        else:
            self.W = similarity.compute_similarity()
            self.W = self.W.toarray()

