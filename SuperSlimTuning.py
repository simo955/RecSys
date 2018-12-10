from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, \
    MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
from KNN.ItemKNNScoresHybridRecommender3 import ItemKNNScoresHybridRecommender3

from KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from KNN.ItemKNNSimilarityHybridRecommender3 import ItemKNNSimilarityHybridRecommender3
from KNN.ItemKNNSimilarityHybridRecommender4 import ItemKNNSimilarityHybridRecommender4

# from GraphBased.RP3betaRecommender import RP3betaRecommender
# from GraphBased.P3alphaRecommender import P3alphaRecommender

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

CFrecommender = ItemKNNCFRecommender(URM_train)
CFrecommender.fit()
SLIMrecommender = SLIM_BPR_Cython(URM_train)
SLIMrecommender.fit()
CBFrecommender = ItemKNNCBFRecommender(ICM_all, URM_train)
CBFrecommender.fit()
SLIMErecommender = SLIMElasticNetRecommender(URM_train)
SLIMErecommender.fit()
UCFRecommender = UserKNNCFRecommender(URM_train)
UCFRecommender.fit()

print("JUST TO KNOW")
print(SLIMErecommender.evaluateRecommendations(URM_test))

SuperSimilarity = ItemKNNSimilarityHybridRecommender(URM_train, SLIMrecommender, SLIMErecommender)


alpha = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
topk1 = [100, 400, 800]
topk2 = [100, 200,400,600 ,800]

alphaSim= [0.6]
import random

for sim in alphaSim:
    for t in topk1:
        SuperSimilarity.fit(alpha=sim, topK=t)

        H3ScoresSuper = ItemKNNSimilarityHybridRecommender3(URM_train, SuperSimilarity, CFrecommender,
                                                            CBFrecommender)
        print("ALPHAA" * 30)
        for a in alpha:
            print("BETAA" * 30)
            for b in alpha:
                for c in alpha:
                    print("ITERATION sim={},alpha={},beta={}, gamma={}".format(sim, a, b, c))
                    for t1 in topk2:
                        print("TOPK ={}".format(t1))
                        H3ScoresSuper.fit(topK=t1, alpha=a, beta=b, gamma=c)
                        result = H3ScoresSuper.evaluateRecommendations(URM_test)
                        print("Recommender MAP is= {}".format(result["MAP"]))

