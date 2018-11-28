#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender

#Tunato miglior risultato ottenuto con alpha =0.8. Alpha moltiplica la similarity di CF
#Tuning miglior risultato ottenuto con aplha=0.8 MAP': 0.08526621576298066
#Bisogna passargli recommender già fittati

class ItemKNNSimilarityHybridRecommender3(SimilarityMatrixRecommender, Recommender):
    """ ItemKNNSimilarityHybridRecommender3
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityHybridRecommender3"


    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3, sparse_weights=True):
        super(ItemKNNSimilarityHybridRecommender3, self).__init__()
        
        #Get Similarity matrix (W_sparse) from Recommender1 and normalize its value for its max
        Similarity_1= Recommender_1.W_sparse/ Recommender_1.W_sparse.max()

        #Get Similarity matrix (W_sparse) from Recommender2 and normalize its value for its max
        Similarity_2= Recommender_2.W_sparse/ Recommender_2.W_sparse.max()
        
        #Get Similarity matrix (W_sparse) from Recommender3 and normalize its value for its max
        Similarity_3= Recommender_3.W_sparse/ Recommender_3.W_sparse.max()


     
        if Similarity_1.shape != Similarity_2.shape or Similarity_1.shape != Similarity_3.shape or Similarity_2.shape != Similarity_3.shape :
            raise ValueError("ItemKNNSimilarityHybridRecommender3: similarities have different size, S1 is {}, S2 is {}, S3 is {}".format(
                Similarity_1.shape, Similarity_2.shape, Similarity_3.shape
            ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')



        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights


    def fit(self, topK=100, alpha = 0.4, beta= 0.4, gamma=0.2):

        self.topK = topK
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma



        W = self.Similarity_1*self.alpha + self.Similarity_2*(self.beta)+self.Similarity_3*( self.gamma)

        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)

