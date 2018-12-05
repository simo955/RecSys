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

SuperSimilarity = ItemKNNSimilarityHybridRecommender(URM_train, SLIMrecommender, SLIMErecommender)

normList = ["l2", "l1", "max"]
topk = [100, 400, 800]

import random

for y in range(700):
    print("Iteration {}".format(y))
    sim = random.uniform(0.3, 0.7)
    a = random.uniform(0.1, 0.7)
    b = random.uniform(0.1, 0.8)
    c = random.uniform(0.1, 0.5)
    d = random.uniform(0.1, 0.4)

    print("ITERATION sim={},alpha={},beta={}, gamma={}, delta={}".format(sim, a, b, c, d))

    SuperSimilarity.fit(alpha=sim)
    for n in normList:
        print("NORM ={}".format(n))

        H3Scores1 = ItemKNNSimilarityHybridRecommender3(URM_train, SuperSimilarity, CFrecommender, CBFrecommender,
                                                        norm=n)
        H4Scores1 = ItemKNNSimilarityHybridRecommender4(URM_train, SLIMrecommender, SLIMErecommender, CFrecommender,
                                                        CBFrecommender, norm=n)

        for t in topk:
            print("TOPK ={}".format(t))

            H3Scores1.fit(topK=t, alpha=a, beta=b, gamma=c)
            result = H3Scores1.evaluateRecommendations(URM_test)
            print("Recommender MAP is= {}".format(result["MAP"]))

            H4Scores1.fit(topK=t, alpha=a, beta=b, gamma=c, delta=d)
            result = H4Scores1.evaluateRecommendations(URM_test)
            print("Recommender MAP is= {}".format(result["MAP"]))

