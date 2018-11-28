
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender


import traceback, os


if __name__ == '__main__':





    import scipy.sparse
    URM_all = scipy.sparse.load_npz('URM_all_matrix.npz')
    ICM_all = scipy.sparse.load_npz('ICM_all_matrix.npz')
    URM_train = scipy.sparse.load_npz('URM_train_matrix.npz')
    URM_test = scipy.sparse.load_npz('URM_test_matrix.npz')

    recommender_list = [
        #Random,
        #TopPop,
        #P3alphaRecommender,
        #RP3betaRecommender,
        #ItemKNNCFRecommender,
        #UserKNNCFRecommender,
        #MatrixFactorization_BPR_Cython,
        #MatrixFactorization_FunkSVD_Cython,
        #PureSVDRecommender,
        #SLIM_BPR_Cython,
        SLIMElasticNetRecommender
        ]


    from Base.Evaluation.Evaluator import SequentialEvaluator

    evaluator = SequentialEvaluator(URM_test, [20], exclude_seen=True)


    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_all_algorithms.txt", "a")


    for recommender_class in recommender_list:

        try:

            print("Algorithm: {}".format(recommender_class))



            recommender = recommender_class(URM_train)
            recommender.fit()
            result = recommender.evaluateRecommendations(URM_test)
            print("Recommender MAP is= {}".format(result["MAP"]))
            # results_run, results_run_string = evaluator.evaluateRecommender(recommender)

            # print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
            # logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
            # logFile.flush()

        except Exception as e:
            traceback.print_exc()
            logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
            logFile.flush()
