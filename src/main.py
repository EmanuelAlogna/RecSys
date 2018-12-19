from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time
import matplotlib.pyplot as plt
from src.SLIM_BPR_Cython import SLIM_BPR_Cython
#from src.MF_Python import *
#from src.SLIM_BPR_Python2 import *
from src.Evaluator import *
from src.AbstractClassSearch import *
from src.ItemKNNCFRecommender import *
from src.BayesianSearch import *
import os
import scipy.sparse as sps
from src.HybridRecommender import *
from sklearn.preprocessing import normalize

dr = DataReader("train.csv", split_train_test=True, build_validation = False)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

print('number of non zero elements')
print(URM_train.nnz)



#URM_validation = dr.URM_validation
#URM_validation = dr.URM_validation



ICM_all,ICM_all2 = dr.build_icm("tracks.csv")
file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])


file = pd.read_csv("../data/tracks.csv")
duration = list(file["duration_sec"])


UF = np.dot(URM_all,ICM_all)
UF.data = np.ones(UF.nnz)

from sklearn.preprocessing import maxabs_scale

# x = np.random.randn(20645)
# print(x)
# x = maxabs_scale(x)
# print(x)


# print('URM')
# print(URM_train)
# print('\n')
# print('ICM')
# print(ICM_all)
# print('\n')
# print('UF')
#print(UF)
# print('\n')
#




#######################################################################
############################## EVALUATION #############################
#######################################################################
"""

# importing evaluator objects
evaluator_validation = SequentialEvaluator(URM_validation, cutoff_list=[5])
evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5, 10])

evaluator_validation = EvaluatorWrapper(evaluator_validation)
evaluator_test = EvaluatorWrapper(evaluator_test)

# create bayesian search object
BPR = SLIM_BPR_Cython

parameterSearch = BayesianSearch(BPR,
                                 evaluator_validation=evaluator_validation,
                                 evaluator_test=evaluator_test)
metric_to_optimize = "MAP"

# define parameters range
hyperparamethers_range_dictionary = {}
hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 15, 20, 25, 30]
hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad"]
#hyperparamethers_range_dictionary["learning_rate"] = [0.1, 0.01, 0.001, 0.2]
hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                         DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':True, 'positive_threshold':0},
                         DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                         DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True,
                                                         "evaluator_object":evaluator_validation,
                                                         "lower_validatons_allowed":10, "validation_metric":metric_to_optimize},
                           DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

output_root_path = "result_experiments4/"


# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)


n_cases = 3


# parameterSearch is a object from the BayesianSearch class
best_parameters = parameterSearch.search(recommenderDictionary,
                                         n_cases = n_cases,
                                         output_root_path = output_root_path,
                                         metric=metric_to_optimize)


"""


###                  ###
### PARAMETER TUNING ###
###                  ###






"""
A = [0.05, 0.1, 0.2]

#batch_tick = [1, 10, 100, 1000, 10000, 100000, 1000000]
#recommender_class = ItemBasedCollaborativeRS(URM_all)


gamma = [0.7, 0.75, 0.80, 0.85, 0.90, 0.95]

top_k = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

top_k2 = [900, 1000, 1100, 1200, 1300, 1400, 1500]
#shrink_tick = [0, 10, 50, 100, 200, 300, 500, 1000]
#epoch_tick = [2,3,10,15,20]
#MAP_per_k = []
for g in gamma:
    BPR.fit(epochs=5, batch_size=1, topK=1000, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.1, lambda_j=0.1, gamma = g)
    print("K:" + str(g))
    evaluate_algorithm(URM_test, BPR)

#    MAP_per_k.append(result)

#plt.plot(x_tick, MAP_per_k)
#plt.ylabel('MAP')
#plt.xlabel('K')
#plt.show()
"""
"""




"""
start_time = time.time()


'''
print("MAP OF 0.09363 ON KAGGLE")

ItemCF = ItemBasedCollaborativeRS(URM_train)
print("Fit collaborative filtering")
sim2 = ItemCF.fit(top_k=300, shrink=10)
evaluate_algorithm(URM_test,ItemCF)


sim2 = normalize(sim2, norm='l2')

BPR = SLIM_BPR_Cython(URM_train, sim2, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
                      final_model_sparse_weights=False, )
CF_normalized = BPR.fit(epochs=4, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
CF_normalized = normalize(CF_normalized, norm='l2')
#evaluate_algorithm(URM_test, BPR)


ContentB = ItemCBFKNNRecommender(URM_train, ICM_all2)
print("Fit content based")

sim1 = ContentB.fit(20, 10)

sim1 = normalize(sim1, norm='l2')
BPR2 = SLIM_BPR_Cython(URM_train, sim1, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
                      final_model_sparse_weights=False)

CB_normalized = BPR2.fit(epochs=22, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
CB_normalized = normalize(CB_normalized, norm='l2')
#evaluate_algorithm(URM_test, BPR2)


hybrid2 = HybridRecommender(URM_train, CB_normalized, CF_normalized)

sim = hybrid2.fit(340, 0.3)
evaluate_algorithm(URM_test, hybrid2)
'''

###### MATRIX FACTORIZATION #####
from src.MatrixFactorization_Cython import MatrixFactorization_Cython






#
# # importing evaluator objects
# evaluator_validation = SequentialEvaluator(URM_validation, cutoff_list=[5])
# evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5, 10])
#
# evaluator_validation = EvaluatorWrapper(evaluator_validation)
# evaluator_test = EvaluatorWrapper(evaluator_test)
#
# # create bayesian search object
# #BPR = SLIM_BPR_Cython
# MF = MatrixFactorization_Cython
#
# parameterSearch = BayesianSearch(MF,
#                                  evaluator_validation=evaluator_validation,
#                                  evaluator_test=evaluator_test)
# metric_to_optimize = "MAP"
#
# # define parameters range
# hyperparamethers_range_dictionary = {}
# hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
# # hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
# hyperparamethers_range_dictionary["num_factors"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
# hyperparamethers_range_dictionary["batch_size"] = [1]
# hyperparamethers_range_dictionary["positive_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
# hyperparamethers_range_dictionary["negative_reg"] = [0.0, 1e-3, 1e-6, 1e-9]
# hyperparamethers_range_dictionary["learning_rate"] = [1e-2, 1e-3, 1e-4, 1e-5]
#
# recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
#                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold': 0},
#                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
#                          DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n": 20, "stop_on_validation": True,
#                                                            "evaluator_object": evaluator_validation,
#                                                            "lower_validatons_allowed": 20,
#                                                            "validation_metric": metric_to_optimize},
#                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
#
#
# output_root_path = "result_experiments4/"
#
#
# # If directory does not exist, create
# if not os.path.exists(output_root_path):
#     os.makedirs(output_root_path)
#
#
# n_cases = 20
#
#
# # parameterSearch is a object from the BayesianSearch class
# best_parameters = parameterSearch.search(recommenderDictionary,
#                                          n_cases = n_cases,
#                                          output_root_path = output_root_path,
#                                          metric=metric_to_optimize)
#
#
#
#
from src.S_SLIM_BPR_Cython import S_SLIM_BPR_Cython
# from src.MABPR_Cython import *
# #print(normalize(np.dot(ICM_all , ICM_all.T),norm='l2'))
#
# MABPR = MABPR_Cython(URM_train, ICM_all, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
#                       final_model_sparse_weights=False,symmetric=False)
# MABPR.fit(epochs=20, batch_size=1, topK=200, sgd_mode='sgd', learning_rate=0.01, lambda_i=0.001, lambda_j=0.0001)
# evaluate_algorithm(URM_test, MABPR)



from src.MABPR_Cython import MABPR_Cython
from src.HybridRecommenderScores import HybridRecommenderScores
#
# print("MAP OF 0.09363 ON KAGGLE")
# ItemCF = ItemBasedCollaborativeRS(URM_train)
# print("Fit collaborative filtering")
# sim2 = ItemCF.fit(top_k=300, shrink=10)
# sim2 = normalize(sim2, norm='l2')
# BPR = SLIM_BPR_Cython(URM_train, sim2, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
#                       final_model_sparse_weights=False)
# CF_normalized = BPR.fit(epochs=4, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
# CF_normalized = normalize(CF_normalized, norm='l2')
# evaluate_algorithm(URM_test, BPR)
#
# MABPR = MABPR_Cython(URM_train, ICM_all,URM_test, UF = UF,recompile_cython=False, positive_threshold=1, train_with_sparse_weights=True,
#                       final_model_sparse_weights=False,symmetric=False)
# MABPR.fit(epochs=1, batch_size=1, topK=200, sgd_mode='sgd', negative_items=False,learning_rate=0.01, lambda_i=0.01, lambda_j=0.0001)
#
#
# Hybrid = HybridRecommenderScores(URM_train,BPR,MABPR)
# evaluate_algorithm(URM_test,Hybrid)
#
#
# ContentB = ItemCBFKNNRecommender(URM_train, ICM_all)
# print("Fit content based")
# sim1 = ContentB.fit(20, 10)
# sim1 = normalize(sim1, norm='l2')
# BPR2 = SLIM_BPR_Cython(URM_train, sim1, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
#                       final_model_sparse_weights=False)
# CB_normalized = BPR2.fit(epochs=22, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
# CB_normalized = normalize(CB_normalized, norm='l2')
# evaluate_algorithm(URM_test, BPR2)
# Hybrid2 = HybridRecommenderScores(URM_train,BPR2,MABPR)
#
#
# Hybrid3 = HybridRecommenderScores(URM_train, Hybrid, Hybrid2)
# evaluate_algorithm(URM_test,Hybrid3)


# CF_normalized = BPR.fit(epochs=4, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
# CF_normalized = normalize(CF_normalized, norm='l2')
# #evaluate_algorithm(URM_test, BPR)
# ContentB = ItemCBFKNNRecommender(URM_train, ICM_all2)
# print("Fit content based")
# sim1 = ContentB.fit(20, 10)
# sim1 = normalize(sim1, norm='l2')
# BPR2 = SLIM_BPR_Cython(URM_train, sim1, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
#                       final_model_sparse_weights=False)
# CB_normalized = BPR2.fit(epochs=22, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
# CB_normalized = normalize(CB_normalized, norm='l2')
# #evaluate_algorithm(URM_test, BPR2)
# hybrid2 = HybridRecommender(URM_train, CB_normalized, CF_normalized)
# sim = hybrid2.fit(340, 0.3)
# evaluate_algorithm(URM_test, hybrid2)




from src.HybridRecommenderScores import *
ItemCF = ItemBasedCollaborativeRS(URM_train)
print("Fit collaborative filtering")
sim2 = ItemCF.fit(top_k=300, shrink=10)
sim2 = normalize(sim2, norm='l2')





BPR = SLIM_BPR_Cython(URM_train, sim2, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
                      final_model_sparse_weights=False, scan=False)
BPR.fit(epochs=1, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
evaluate_algorithm(URM_test,BPR)


print('MABPR IMPROVEMENT')


MABPR = MABPR_Cython(URM_train, ICM_all,URM_test, UF = UF,recompile_cython=False, positive_threshold=1, train_with_sparse_weights=True,
                      final_model_sparse_weights=False,symmetric=False)
MABPR.fit(epochs=1, batch_size=1, topK=200, sgd_mode='sgd', negative_items=False,learning_rate=0.01, lambda_i=0.01, lambda_j=0.0001)
#evaluate_algorithm(URM_test,MABPR)






Hybrid = HybridRecommenderScores(URM_train,BPR,MABPR)
evaluate_algorithm(URM_test,Hybrid)

MABPR2 = MABPR_Cython(URM_train, ICM_all,URM_test, UF = UF,recompile_cython=False, positive_threshold=1, train_with_sparse_weights=True,
                      final_model_sparse_weights=False,symmetric=False)
MABPR2.fit(epochs=1, batch_size=1, topK=200, sgd_mode='sgd', negative_items=True,learning_rate=0.01, lambda_i=0.01, lambda_j=0.0001)

Hybrid2 = HybridRecommenderScores(URM_train,Hybrid,MABPR2)
evaluate_algorithm(URM_test,Hybrid2)

#evaluate_algorithm(URM_test,MABPR)





#make_recommendations(Hybrid2,target_playlist)


#make_recommendations2(BPR,MABPR,target_playlist)
# MABPR2 = MABPR_Cython(URM_train, ICM_all,URM_test, UF = UF,recompile_cython=False, positive_threshold=1, train_with_sparse_weights=True,
#                       final_model_sparse_weights=False,symmetric=False)
# MABPR2.fit(epochs=4, batch_size=1, topK=200, sgd_mode='sgd', negative_items=True,learning_rate=0.01, lambda_i=0.0001, lambda_j=0.0001)
#
#
# evaluate_algorithm_hybrid2(URM_test, BPR,MABPR,MABPR2)
#



print("Total time: {}".format(time.time() - start_time))
