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


#DEFINITION
#similarity
HrecommenderThree1=ItemKNNSimilarityHybridRecommender3(URM_train,SLIMrecommender, CFrecommender,CBFrecommender)
HrecommenderThree2=ItemKNNSimilarityHybridRecommender3(URM_train,SLIMrecommender, CFrecommender, SLIMELASTICrecommender)
HrecommenderThree3=ItemKNNSimilarityHybridRecommender3(URM_train,SLIMrecommender, SLIMELASTICrecommender,CBFrecommender)
HrecommenderThree4=ItemKNNSimilarityHybridRecommender3(URM_train,CFrecommender, SLIMELASTICrecommender,CBFrecommender)

#ratings
H3recommender1 = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, CBFrecommender, userID)
H3recommender1X = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, CBFrecommender, userID, "XX")
H3recommender2 = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, UCFrecommender, userID)
H3recommender2X = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, UCFrecommender, userID, "XX")
H3recommender3 = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, SLIMELASTICrecommender, userID)
H3recommender3X = ItemKNNRatingHybridRecommender3(URM_train,CFrecommender, SLIMrecommender, SLIMELASTICrecommender, userID, "XX")
H3recommender4 = ItemKNNRatingHybridRecommender3(URM_train,UCFrecommender, SLIMrecommender, CBFrecommender, userID)
H3recommender4X = ItemKNNRatingHybridRecommender3(URM_train,UCFrecommender, SLIMrecommender, CBFrecommender, userID, "XX")



H4recommender1 = ItemKNNRatingHybridRecommender4(URM_train,SLIMrecommender,CFrecommender,UCFrecommender, CBFrecommender, userID)
H4recommender2 = ItemKNNRatingHybridRecommender4(URM_train,SLIMrecommender,CFrecommender, SLIMELASTICrecommender,CBFrecommender, userID)
H4recommender3 = ItemKNNRatingHybridRecommender4(URM_train,SLIMrecommender,CFrecommender, UCFrecommender,SLIMELASTICrecommender, userID)
H4recommender4 = ItemKNNRatingHybridRecommender4(URM_train,SLIMrecommender,UCFrecommender,SLIMELASTICrecommender,CBFrecommender, userID)


H5recommender1 = ItemKNNRatingHybridRecommender5(URM_train,SLIMrecommender,CFrecommender,UCFrecommender,SLIMELASTICrecommender,CBFrecommender, userID)


#SCORES
import random

top = [40, 200, 500]

for y in range(800):
    print("Iteration".format(y))

    a = random.uniform(0.1, 0.9)
    b = random.uniform(0.1, 0.9)
    c = random.uniform(0.1, 0.9)
    d = random.uniform(0.1, 0.5)
    e = random.uniform(0.1, 0.5)

    # similarity
    for x in top:
        print(x)
        HrecommenderThree1 = fit(topK=x, alpha=a, beta=b, gamma=c)
        HrecommenderThree2 = fit(topK=x, alpha=a, beta=b, gamma=c)
        HrecommenderThree3 = fit(topK=x, alpha=a, beta=b, gamma=c)
        HrecommenderThree4 = fit(topK=x, alpha=a, beta=b, gamma=c)

        print("SIMILARITY")
        result1 = HrecommenderThree1.evaluateRecommendations(URM_test)
        print("Recommender MAP is= {}".format(result1["MAP"]))
        result2 = HrecommenderThree2.evaluateRecommendations(URM_test)
        print("Recommender2 MAP is= {}".format(result2["MAP"]))
        result3 = HrecommenderThree3.evaluateRecommendations(URM_test)
        print("Recommender3 MAP is= {}".format(result3["MAP"]))
        result4 = HrecommenderThree4.evaluateRecommendations(URM_test)
        print("Recommender4 MAP is= {}".format(result4["MAP"]))

    # ratings
    H3recommender1 = fit(alpha=a, beta=b, gamma=c)
    H3recommender1X = fit(alpha=a, beta=b, gamma=c)
    H3recommender2 = fit(alpha=a, beta=b, gamma=c)
    H3recommender2X = fit(alpha=a, beta=b, gamma=c)
    H3recommender3 = fit(alpha=a, beta=b, gamma=c)
    H3recommender3X = fit(alpha=a, beta=b, gamma=c)
    H3recommender4 = fit(alpha=a, beta=b, gamma=c)
    H3recommender4X = fit(alpha=a, beta=b, gamma=c)

    # ratings
    H4recommender1 = fit(alpha=a, beta=b, gamma=c, delta=d)
    H4recommender2 = fit(alpha=a, beta=b, gamma=c, delta=d)
    H4recommender3 = fit(alpha=a, beta=b, gamma=c, delta=d)
    H4recommender4 = fit(alpha=a, beta=b, gamma=c, delta=d)

    # ratings
    H5recommender1 = fit(alpha=a, beta=b, gamma=c, delta=d, epsilon=e)

    print("RATINGS3")
    result5 = H3recommender1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result5["MAP"]))
    result6 = H3recommender1X.evaluateRecommendations(URM_test)
    print("Recommender2 MAP is= {}".format(result6["MAP"]))
    result6 = H3recommender2.evaluateRecommendations(URM_test)
    print("Recommender3 MAP is= {}".format(result7["MAP"]))
    result8 = H3recommender2X.evaluateRecommendations(URM_test)
    print("Recommender4 MAP is= {}".format(result8["MAP"]))
    result9 = H3recommender3.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result9["MAP"]))
    result10 = H3recommender3X.evaluateRecommendations(URM_test)
    print("Recommender2 MAP is= {}".format(result10["MAP"]))
    result11 = H3recommender4.evaluateRecommendations(URM_test)
    print("Recommender3 MAP is= {}".format(result11["MAP"]))
    result12 = H3recommender4X.evaluateRecommendations(URM_test)
    print("Recommender4 MAP is= {}".format(result12["MAP"]))

    print("RATINGS4")
    result13 = H4recommender1.evaluateRecommendations(URM_test)
    print("Recommender MAP is= {}".format(result13["MAP"]))
    result14 = H4recommender2.evaluateRecommendations(URM_test)
    print("Recommender2 MAP is= {}".format(result14["MAP"]))
    result15 = H4recommender3.evaluateRecommendations(URM_test)
    print("Recommender3 MAP is= {}".format(result15["MAP"]))
    result16 = H4recommender4.evaluateRecommendations(URM_test)
    print("Recommender4 MAP is= {}".format(result16["MAP"]))

    print("RATINGS5")
    result17 = H5recommender1.evaluateRecommendations(URM_test)
    print("Recommender4 MAP is= {}".format(result17["MAP"]))
