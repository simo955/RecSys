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
from KNN.ItemKNNScoresHybridRecommender4 import ItemKNNScoresHybridRecommender4
from KNN.ItemKNNScoresHybridRecommender5 import ItemKNNScoresHybridRecommender5


import traceback, os
import scipy.sparse
import pandas as pd
import numpy as np


#Load Matrix
URM_all = scipy.sparse.load_npz('./Matrix/URM_all_matrix.npz')
ICM_all = scipy.sparse.load_npz('./Matrix/ICM_all_matrix.npz')
URM_train = scipy.sparse.load_npz('./Matrix/URM_train_matrix.npz')
URM_test = scipy.sparse.load_npz('./Matrix/URM_test_matrix.npz')

#Load playlistsIDS
data_playlists= pd.read_csv('all/train.csv',low_memory = False)
userID=data_playlists.playlist_id.unique()

#FITTING
CFrecommender=ItemKNNCFRecommender(URM_train)
SLIMrecommender=SLIM_BPR_Cython(URM_train)
CBFrecommender=ItemKNNCBFRecommender(ICM_all,URM_train)
UCFrecommender=UserKNNCFRecommender(URM_train)
SLIMELASTICrecommender=SLIMElasticNetRecommender(URM_train)

UCFrecommender.fit()
CFrecommender.fit()
SLIMrecommender.fit()
SLIMELASTICrecommender.fit()
CBFrecommender.fit()

SuperScores=ItemKNNScoresHybridRecommender(URM_train,SLIMrecommender,SLIMELASTICrecommender)
SuperScores.fit(alpha=0.5)

H4Scores1=ItemKNNScoresHybridRecommender4(URM_train,SuperScores,CFrecommender,UCFrecommender, CBFrecommender)
H4Scores2=ItemKNNScoresHybridRecommender4(URM_train,SuperScores,UCFrecommender,CFrecommender, CBFrecommender)
H4Scores3=ItemKNNScoresHybridRecommender4(URM_train,SuperScores,CFrecommender,UCFrecommender,CBFrecommender)

H4XScores1=ItemKNNScoresHybridRecommender4(URM_train,SLIMELASTICrecommender,CFrecommender,UCFrecommender,CBFrecommender)
H4XScores2=ItemKNNScoresHybridRecommender4(URM_train,SLIMELASTICrecommender,SLIMrecommender,UCFrecommender, CBFrecommender)

H5Scores1=ItemKNNScoresHybridRecommender5(URM_train,SLIMELASTICrecommender,SLIMrecommender,CFrecommender,UCFrecommender, CBFrecommender)

alphaList = [(0.5, 0.5, 0.3, 0.2, 0.1), (0.5, 0.4, 0.2,0.2,0.1), (0.5, 0.5, 0.3,0.2,0.2), (0.5, 0.3, 0.1,0.1,0.1), (0.4, 0.3, 0.2,0.1,0.2), (0.4, 0.2, 0.2,0.3,0.3), (0.6, 0.2, 0.2,0.2,0.2), (0.5, 0.2, 0.2,0.1,0.1), (0.3, 0.5, 0.2,0.3,0.1), (0.3, 0.2, 0.2,0.1,0.2), (0.3, 0.3, 0.1,0.1,0.1)]
'''
for a, b, c, d, e in alphaList:
    print("X"*10)
    print("ITERATION alpha={},beta={}, gamma={}, delta={}, epsilon={}".format(a, b, c, d, e))

    H4Scores1.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4Scores2.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4Scores3.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores3.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4XScores1.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4XScores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4XScores2.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4XScores2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    print("5 RECOMMENDER")
    H5Scores1.fit(alpha=a, beta=b, gamma=c, delta=d, epsilon=e)
    result = H5Scores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))
'''
#tuning per 4 scores , vediamo se performa meglio di 3 scores
#si prova anche la classe con 5 scores
#i recommender con X nel nome funzionano senza superslim, vediamo se vanno meglio
print ("FINE BLOCCO 1"*5)




import random
for y in range(600):
    print("Iteration".format(y))

    a = random.uniform(0.1, 0.9)
    b = random.uniform(0.1, 0.7)
    c = random.uniform(0.1, 0.5)
    d = random.uniform(0.1, 0.4)
    e = random.uniform(0.1, 0.4)

    print("ITERATION alpha={},beta={}, gamma={}, delta={}, epsilon={}".format(a, b, c, d, e))

    H4Scores1.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4Scores2.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4Scores3.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4Scores3.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4XScores1.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4XScores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    H4XScores2.fit(alpha=a, beta=b, gamma=c, delta=d)
    result = H4XScores2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    print("5 RECOMMENDER")
    H5Scores1.fit(alpha=a, beta=b, gamma=c, delta=d, epsilon=e)
    result = H5Scores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

#le combo possibili di valori da dargli sono veramente molteplici, vediamo se cosi la becchiamo. Si punta a superare 0.0983