#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17


"""


import numpy as np
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Recommender import Recommender
from sklearn.preprocessing import normalize




class Hybrid_List(Recommender):
    """ Hybrid_List
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "Hybrid_List"


    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(Hybrid_List, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2
    
    def fit(self,user_id_list1, user_id_list2=None):
        self.user_id_list1 = user_id_list1
        
        if user_id_list2 is not None:
            self.user_id_list2 = user_id_list2
        
               
        
    def recommend(self, user_id_array, cutoff = 10, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):   
        if user_id_array in  self.user_id_list1:
 
            return self.Recommender_1.recommend(user_id_array)
        else :
            return self.Recommender_2.recommend(user_id_array)
            
        
        
        