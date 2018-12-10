from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.ItemKNNSimilarityHybridRecommender3 import ItemKNNSimilarityHybridRecommender3

from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
from KNN.ItemKNNScoresHybridRecommender3 import ItemKNNScoresHybridRecommender3

print("CIAO")

import traceback, os
import scipy.sparse
import pandas as pd
import numpy as np
URM_all = scipy.sparse.load_npz('./Matrix/URM_all_matrix.npz')
ICM_all = scipy.sparse.load_npz('./Matrix/ICM_all_matrix.npz')
URM_train = scipy.sparse.load_npz('./Matrix/URM_train_matrix.npz')
URM_test = scipy.sparse.load_npz('./Matrix/URM_test_matrix.npz')

# Load playlistsIDS
data_playlists = pd.read_csv('all/train.csv', low_memory=False)
userID = data_playlists.playlist_id.unique()

'''
CBFrecommender = ItemKNNCBFRecommender(ICM_all,URM_train)
CBFrecommender.fit()
CFrecommender = ItemKNNCFRecommender(URM_train)
CFrecommender.fit()
UCFrecommender = UserKNNCFRecommender(URM_train)
UCFrecommender.fit()
'''

recommender = SLIMElasticNetRecommender(URM_train)
alpha= [1, 0.01, 0.001,0.0001]
topK = [10, 100, 200, 500, 800]
l1_ratio = [1.0, 0.1, 0.5, 0.05, 1e-2, 1e-4]
for a in alpha:
    for k in topK:
        for l in l1_ratio:
            print("ITERATION alpha={},topK={}, l1={}".format(a, k, l))
            recommender.fit(alpha=a,topK= k, l1_ratio=l)
            result = recommender.evaluateRecommendations(URM_test)
            print("Recommender MAP is= {}".format(result["MAP"]))