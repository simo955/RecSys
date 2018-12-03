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
from KNN.ItemKNNSimilarityHybridRecommender4 import ItemKNNSimilarityHybridRecommender4

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

SuperSlimSimilarity1=ItemKNNSimilarityHybridRecommender(URM_train,SLIMrecommender,SLIMELASTICrecommender,"l2")

alpha=[0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9]
topK=[100,500,1000]
alphaList = [(0.5, 0.5, 0.2), (0.5, 0.4, 0.1), (0.5, 0.5, 0.3), (0.5, 0.3, 0.1), (0.4, 0.3, 0.2), (0.4, 0.2, 0.2), (0.6, 0.2, 0.2), (0.5, 0.2, 0.2), (0.3, 0.5, 0.2), (0.3, 0.2, 0.2), (0.3, 0.3, 0.1)]

for alp in alpha:
    for y in topK:
        print("XXXXXXXXXXXXXXXXXXXX alp={},y={}".format(alp,y))
        SuperSlimSimilarity1.fit(alpha=alp, topK=y)
        HSimilarity11 = ItemKNNSimilarityHybridRecommender3(URM_train, SLIMELASTICrecommender, CFrecommender,CBFrecommender, "l2")
        HSimilarity21 = ItemKNNSimilarityHybridRecommender3(URM_train, SLIMrecommender, CFrecommender, CBFrecommender, "l2")
        HSimilarity31 = ItemKNNSimilarityHybridRecommender3(URM_train, SuperSlimSimilarity1, CFrecommender, CBFrecommender,"l2")
        for a, b, c in alphaList:
            print("ITERATION alpha={},beta={}, gamma={}".format(a, b, c))
            for x in topK:
                print("TOPK={}".format(x))
                print ("L2")
                HSimilarity11.fit(topK=x, alpha=a, beta=b, gamma=c)
                result = HSimilarity11.evaluateRecommendations(URM_test)
                print("Recommender MAP is= {}".format(result["MAP"]))
                HSimilarity21.fit(topK=x, alpha=a, beta=b, gamma=c)
                result = HSimilarity21.evaluateRecommendations(URM_test)
                print("Recommender MAP is= {}".format(result["MAP"]))
                HSimilarity31.fit(topK=x, alpha=a, beta=b, gamma=c)
                result = HSimilarity31.evaluateRecommendations(URM_test)
                print("Recommender MAP is= {}".format(result["MAP"]))

print("Tuning voluto per confrontare le normalizzazioni diverse nella matrice similarity. I valori potrebbero comunque essere alti perchè uso gli algoritmi ben tuniati")