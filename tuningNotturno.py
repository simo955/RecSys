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

from KNN.ItemKNNRatingHybridRecommender import ItemKNNRatingHybridRecommender
from KNN.ItemKNNRatingHybridRecommender3 import ItemKNNRatingHybridRecommender3
from KNN.ItemKNNRatingHybridRecommender4 import ItemKNNRatingHybridRecommender4
from KNN.ItemKNNRatingHybridRecommender5 import ItemKNNRatingHybridRecommender5

from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
from KNN.ItemKNNScoresHybridRecommender3 import ItemKNNScoresHybridRecommender3

import traceback, os
import scipy.sparse
import pandas as pd
import numpy as np


#Load Matrix
URM_all = scipy.sparse.load_npz('URM_all_matrix.npz')
ICM_all = scipy.sparse.load_npz('ICM_all_matrix.npz')
URM_train = scipy.sparse.load_npz('URM_train_matrix.npz')
URM_test = scipy.sparse.load_npz('URM_test_matrix.npz')

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

alpha=[0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9]
SuperScores=ItemKNNScoresHybridRecommender(URM_train,SLIMrecommender,SLIMELASTICrecommender)
for alp in alpha:
    print("XXXXXXXXXXXXXXXXXXXX alp={}".format(alp))
    SuperScores.fit(alpha=alp)
    result = SuperScores.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

SuperScores.fit(alpha=0.5)

HScores1=ItemKNNScoresHybridRecommender3(URM_train,SuperScores,CFrecommender,UCFrecommender)
HScores2=ItemKNNScoresHybridRecommender3(URM_train,SuperScores,UCFrecommender,CBFrecommender)
HScores3=ItemKNNScoresHybridRecommender3(URM_train,SuperScores,CFrecommender,CBFrecommender)

HRating1=ItemKNNRatingHybridRecommender3(URM_train,SuperScores,CFrecommender,UCFrecommender,userID)
HRating2=ItemKNNRatingHybridRecommender3(URM_train,SuperScores,UCFrecommender,CBFrecommender,userID)
HRating3=ItemKNNRatingHybridRecommender3(URM_train,SuperScores,CFrecommender,CBFrecommender,userID)


alphaList = [(0.5, 0.5, 0.2), (0.5, 0.4, 0.1), (0.5, 0.5, 0.3), (0.5, 0.3, 0.1), (0.4, 0.3, 0.2), (0.4, 0.2, 0.2), (0.6, 0.2, 0.2), (0.5, 0.2, 0.2), (0.3, 0.5, 0.2), (0.3, 0.2, 0.2), (0.3, 0.3, 0.1)]

for a, b, c in alphaList:
    print("ITERATION alpha={},beta={}, gamma={}".format(a, b, c))

    print("SCORES")
    HScores1.fit(alpha=a, beta=b, gamma=c)
    result = HScores1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))
    HScores2.fit(alpha=a, beta=b, gamma=c)
    result = HScores2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))
    HScores3.fit(alpha=a, beta=b, gamma=c)
    result = HScores3.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

    print("RATINGS")
    HRating1.fit(alpha=a, beta=b, gamma=c)
    result = HRating1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))
    HRating2.fit(alpha=a, beta=b, gamma=c)
    result = HRating2.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))
    HRating3.fit(alpha=a, beta=b, gamma=c)
    result = HRating3.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result["MAP"]))

#tuning per capire come tunare super slim usando Scores, e per capire se è meglio la classe rating o quella scores