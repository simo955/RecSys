#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Recommender import Recommender
from sklearn.preprocessing import normalize




class SuperHybridRecommender(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "SuperHybridRecommender"


    def __init__(self, URM_train, RecList, Type = "Similarity"):
        super(SuperHybridRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        n_recs= len(RecList)
        
        if n_recs<2:
            print("ERROR numero di recommenders non sufficente per ibridare")
            return

        self.RecList= RecList
        
        if Type=="Similarity":
            similarity_list=[]
            for n in n_recs:
                Similarity  = normalize(RecList[n].W_sparse, axis=1, copy=True, return_norm=False)
                similarity_list.append(Similarity)
            


                
            
            
            
        else:
            self.compute_item_score = self.compute_score_hybrid
            
  
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
        
        
        