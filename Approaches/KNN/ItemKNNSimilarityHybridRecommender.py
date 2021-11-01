#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18


"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from sklearn.preprocessing import normalize
import numpy as np



#Tunato miglior risultato ottenuto con alpha =0.8. Alpha moltiplica la similarity di CF
#Tuning miglior risultato ottenuto con aplha=0.8 MAP': 0.08526621576298066


class ItemKNNSimilarityHybridRecommender(SimilarityMatrixRecommender, Recommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender"
    

    def __init__(self, URM_train, Recommender_1, Recommender_2, norm="l2", sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender, self).__init__()
        
        self.norm=norm
        
        #Get Similarity matrix (W_sparse) from Recommender1 and normalize its with norm2
        Similarity_1  = normalize(Recommender_1.W_sparse, norm=self.norm, axis=1, copy=True, return_norm=False)
        #Get Similarity matrix (W_sparse) from Recommender2 and normalize its value for its max
        Similarity_2  = normalize(Recommender_2.W_sparse, norm=self.norm, axis=1, copy=True, return_norm=False)
            
        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                Similarity_1.shape, Similarity_2.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')

        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=100, alpha = 0.8):

        self.topK = topK
        self.alpha = alpha

        W = self.Similarity_1*self.alpha + self.Similarity_2*(1-self.alpha)

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)

