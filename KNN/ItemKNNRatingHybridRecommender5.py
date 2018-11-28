#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import numpy as np
from scipy import sparse



class ItemKNNRatingHybridRecommender5(SimilarityMatrixRecommender, Recommender):
    """ ItemKNNRatingHybridRecommender5
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNRatingHybridRecommender5"


    def __init__(self, URM_train, Recommender_1, Recommender_2, Recommender_3,Recommender_4, Recommender_5,userID, sparse_weights=True):
        super(ItemKNNRatingHybridRecommender5, self).__init__()
        
        #Get Similarity matrix (W_sparse) from Recommender1 and normalize its value for its max
        ratingsMatrix1=np.asmatrix(Recommender_1.recommendBatch(userID))
        ratingsMatrix1= ratingsMatrix1/ratingsMatrix1.max()

        #Get Similarity matrix (W_sparse) from Recommender2 and normalize its value for its max
        ratingsMatrix2=np.asmatrix(Recommender_2.recommendBatch(userID))
        ratingsMatrix2= ratingsMatrix2/ratingsMatrix2.max()
        
        #Get Similarity matrix (W_sparse) from Recommender3 and normalize its value for its max
        ratingsMatrix3=np.asmatrix(Recommender_3.recommendBatch(userID))
        ratingsMatrix3= ratingsMatrix3/ratingsMatrix3.max()
        
        #Get Similarity matrix (W_sparse) from Recommender4 and normalize its value for its max
        ratingsMatrix4=np.asmatrix(Recommender_4.recommendBatch(userID))
        ratingsMatrix4= ratingsMatrix4/ratingsMatrix4.max()
        
        #Get Similarity matrix (W_sparse) from Recommender5 and normalize its value for its max
        ratingsMatrix5=np.asmatrix(Recommender_5.recommendBatch(userID))
        ratingsMatrix5= ratingsMatrix5/ratingsMatrix5.max()
        

        # CSR is faster during evaluation
        self.RatingsMatrix1 = sparse.csr_matrix(ratingsMatrix1) 
        self.RatingsMatrix2 = sparse.csr_matrix(ratingsMatrix2) 
        self.RatingsMatrix3 = sparse.csr_matrix(ratingsMatrix3) 
        self.RatingsMatrix4 = sparse.csr_matrix(ratingsMatrix4)
        self.RatingsMatrix5 = sparse.csr_matrix(ratingsMatrix5) 







        self.URM_train = check_matrix(URM_train.copy(), 'csr')

        self.sparse_weights = sparse_weights
        
        
       #creata cosi che sia facile tunare modificando i valori di alpha
    def fit(self, alpha = 0.5,beta =0.2,gamma =0.2,delta=0.1,epsilon=0.1):
        self.ratingsMatrix = (self.RatingsMatrix1*alpha + self.RatingsMatrix2*(beta)+ self.RatingsMatrix3*(gamma)+ self.RatingsMatrix4*(delta)+self.RatingsMatrix5*(beta)).toarray()
        







    def recommend(self, user_id_array, cutoff = 10, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):
        

         # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1       
        
        user_profile = self.ratingsMatrix[user_id_array]
        scores_batch = user_profile




        # if self.normalize:
        #     # normalization will keep the scores in the same range
        #     # of value of the ratings in dataset
        #     user_profile = self.URM_train[user_id]
        #
        #     rated = user_profile.copy()
        #     rated.data = np.ones_like(rated.data)
        #     if self.sparse_weights:
        #         den = rated.dot(self.W_sparse).toarray().ravel()
        #     else:
        #         den = rated.dot(self.W).ravel()
        #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
        #     scores /= den


        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)

        # scores_batch = np.arange(0,3260).reshape((1, -1))
        # scores_batch = np.repeat(scores_batch, 1000, axis = 0)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:,0:cutoff]


 
        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]



        ranking_list = ranking.tolist()
        
        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]
        
        print("PERSONAL RECOMMENDER2")


        return ranking_list
        



    
