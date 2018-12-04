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


class ItemKNNScoresHybridRecommender(Recommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"


    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
        
        self.compute_item_score = self.compute_score_hybrid
        
  
    def fit(self, alpha = 0.5):

        self.alpha = alpha      


    def compute_score_hybrid(self, user_id_array):
        '''
        versione standard normalizzazione
        print("sd")
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_1 = item_weights_1/item_weights_1.max()
        
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_2 = item_weights_2/item_weights_2.max()

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights
        
        '''
        
        '''
        normalizzazione l1
        
        print("l1")
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_1 = normalize(item_weights_1, norm="l1", axis=1, copy=True, return_norm=False)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_2 = normalize(item_weights_2, norm="l1", axis=1, copy=True, return_norm=False)
        
        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights
        '''
        
        
        '''
        normalizzazione l2
        '''
        print("l2")
        item_weights_1 = self.Recommender_1.compute_item_score(user_id_array)
        item_weights_1 = normalize(item_weights_1, norm="l2", axis=1, copy=True, return_norm=False)
        item_weights_2 = self.Recommender_2.compute_item_score(user_id_array)
        item_weights_2 = normalize(item_weights_2, norm="l2", axis=1, copy=True, return_norm=False)
        
        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights
        
        '''
        '''
        