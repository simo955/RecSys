#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Recommender import Recommender
from sklearn.preprocessing import normalize




class SuperHybridRecommender(Recommender):
    """ ItemKNNScoresHybridRecommender"""

    RECOMMENDER_NAME = "SuperHybridRecommender"


    def __init__(self, URM_train, RecList, Type = "Similarity"):
        super(SuperHybridRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.n_recs= len(RecList)
        
        if self.n_recs<2:
            print("ERROR numero di recommenders non sufficente per ibridare")
            return

        self.RecList= RecList
        
        if Type=="Similarity":
            self.similarity_list=[]
            
            #get similarities normalized
            for n in range(self.n_recs):
                Similarity = normalize(RecList[n].W_sparse, axis=1, copy=True, return_norm=False)
                self.similarity_list.append(Similarity)
            
            #check shape of the similarities
            similarity_shape = self.similarity_list[0].shape
            for n in self.similarity_list:
                if similarity_shape != self.similarity_list[n].shape:
                    raise ValueError("ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(similarity_shape, self.similarity_list[n].shape))
            
            
            self.sparse_weights = sparse_weights
            self.fit= fitSimilarity
            
            
            
    def fitSimilarity(self, topK=100, alpha = 0.8, beta= None, gamma=None, delta=None, epsilon=None, six=None):

        self.topK = topK
        
        Similarity_1  = self.similarity_list[0]
        Similarity_2  = self.similarity_list[1]
        
        if (self.n_recs == 2):
            W = Similarity_1*alpha + Similarity_2*(1-alpha)
        
        elif (self.n_recs == 3 and beta is not None and gamma is not None):
            Similarity_3  = self.similarity_list[2]
            W = Similarity_1*alpha + Similarity_2*(beta)+ Similarity_3*(gamma)
            
        elif (self.n_recs == 4 and beta is not None and gamma is not None and delta is not None):
            Similarity_3  = self.similarity_list[2]
            Similarity_4  = self.similarity_list[3]
            W = Similarity_1*alpha + Similarity_2*(beta) + Similarity_3*(gamma)+ Similarity_4*(delta)
            
        elif (self.n_recs == 5 and beta is not None and gamma is not None and delta is not None and epsilon is not None):
            Similarity_3  = self.similarity_list[2]
            Similarity_4  = self.similarity_list[3]
            Similarity_5  = self.similarity_list[4]
            W = Similarity_1*alpha + Similarity_2*(beta) + Similarity_3*(gamma)+ Similarity_4*(delta)+Similarity_5*(epsilon)
            
        elif (self.n_recs == 6 and beta is not None and gamma is not None and delta is not None and epsilon is not None and six is not None):
            Similarity_3  = self.similarity_list[2]
            Similarity_4  = self.similarity_list[3]
            Similarity_5  = self.similarity_list[4]
            Similarity_6  = self.similarity_list[5]            
            W = Similarity_1*alpha + Similarity_2*(beta) + Similarity_3*(gamma)+ Similarity_4*(delta) + Similarity_5*(epsilon)+ Similarity_6*(six)


        if self.sparse_weights:
            self.W_sparse = similarityMatrixTopK(W, forceSparseOutput=True, k=self.topK)
        else:
            self.W = similarityMatrixTopK(W, forceSparseOutput=False, k=self.topK)
  
    def fit(self, alpha = 0.5):

        self.alpha = alpha      


    def compute_score_hybrid(self, user_id_array):

        #versione standard normalizzazione
        #print("sd")
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_1 = item_weights_1/item_weights_1.max()
        
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_2 = item_weights_2/item_weights_2.max()

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights
        
        
        