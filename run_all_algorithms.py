from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from KNN.ItemKNNRatingHybridRecommender import ItemKNNRatingHybridRecommender
from KNN.ItemKNNRatingHybridRecommender3 import ItemKNNRatingHybridRecommender3
from KNN.ItemKNNRatingHybridRecommender4 import ItemKNNRatingHybridRecommender4

from KNN.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender
from KNN.ItemKNNScoresHybridRecommender3 import ItemKNNScoresHybridRecommender3


import traceback, os
import scipy.sparse
URM_all = scipy.sparse.load_npz('./Matrix/URM_all_matrix.npz')
ICM_all = scipy.sparse.load_npz('./Matrix/ICM_all_matrix.npz')
URM_train = scipy.sparse.load_npz('./Matrix/URM_train_matrix.npz')
URM_test = scipy.sparse.load_npz('./Matrix/URM_test_matrix.npz')
#Load playlistsIDS
data_playlists= pd.read_csv('all/train.csv',low_memory = False)
userID=data_playlists.playlist_id.unique()

CFrecommender = ItemKNNCFRecommender(URM_train)
CFrecommender.fit()

CBFrecommender = ItemKNNCBFRecommender(ICM_all,URM_train)
CBFrecommender.fit()


H3recc=ItemKNNScoresHybridRecommender(URM_train, CFrecommender,CBFrecommender)
H3recc.fit(alpha=0.8)
print(H3recc.evaluateRecommendations(URM_test))