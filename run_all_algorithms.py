from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from MatrixFactorization.PyTorch.MF_MSE_PyTorch import MF_MSE_PyTorch

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
from KNN.ItemKNNScoresHybridRecommender3 import ItemKNNScoresHybridRecommender3
from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.ItemKNNSimilarityHybridRecommender3 import ItemKNNSimilarityHybridRecommender3
from KNN.ItemKNNSimilarityHybridRecommender4 import ItemKNNSimilarityHybridRecommender4



#from GraphBased.RP3betaRecommender import RP3betaRecommender
#from GraphBased.P3alphaRecommender import P3alphaRecommenderURM_all = scipy.sparse.load_npz('./Matrix/URM_all_matrix.npz')


import traceback, os
import scipy.sparse
import pandas as pd
import numpy as np

URM_all = scipy.sparse.load_npz('./Matrix/URM_all_matrix.npz')
ICM_all = scipy.sparse.load_npz('./Matrix/ICM_all_matrix.npz')
URM_train = scipy.sparse.load_npz('./Matrix/URM_train_matrix.npz')
URM_test = scipy.sparse.load_npz('./Matrix/URM_test_matrix.npz')

#Load playlistsIDS
data_playlists= pd.read_csv('all/train.csv',low_memory = False)
userID=data_playlists.playlist_id.unique()

SLIMrecommender = SLIM_BPR_Cython(URM_train)
SLIMrecommender.fit()

result = SLIMrecommender.evaluateRecommendations(URM_test)
print("Recommender MAP is= {}".format(result["MAP"]))