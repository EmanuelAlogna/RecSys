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



dr = DataReader("train.csv", split_train_test=True, build_validation = False)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

print(URM_train.shape)
#URM_validation = dr.URM_validation


ICM_all = dr.build_icm("tracks.csv")
file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])




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





#make_recommendations(BPR, target_playlist)

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
# print("Start computation sim matrix")
#
#
ItemCF = ItemBasedCollaborativeRS(URM_train)
sim = ItemCF.fit(top_k=300, shrink=10)
# W1 = sim
# MAP = 0
#
# while MAP < 0.0737:
#     dr = DataReader("train.csv", split_train_test=True, build_validation=False)
#     URM_all = dr.URM_all
#     URM_train = dr.URM_train
#     URM_test = dr.URM_test
#
#     ItemCF = ItemBasedCollaborativeRS(URM_train)
#     sim = ItemCF.fit(top_k=300, shrink=10)
#     print(type(sim))
#       MAP = evaluate_algorithm(URM_test, ItemCF)
#       print(MAP)
#
#evaluate_algorithm(URM_test, ItemCF)

print("Sim matrix built")


print("Start fitting")

BPR = SLIM_BPR_Cython(URM_train, sim, URM_test, recompile_cython=False, positive_threshold=1, train_with_sparse_weights=False,
                      final_model_sparse_weights=False)

BPR.fit(epochs=15, batch_size=10000, topK=700, sgd_mode='adagrad', learning_rate=0.01, lambda_i=0.01, lambda_j=0.01)
#
# W2 = BPR.W
# alpha =[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# for x in alpha:
#     W3 = x*W1 + (1-x)*W2
#     BPR.W = W3
#     evaluate_algorithm(URM_test, BPR)


print("Fitting ended")
print("Start evaluation")
evaluate_algorithm(URM_test, BPR)

#make_recommendations(BPR, target_playlist)


print("Evaluation ended")

print("Total time: {}".format(time.time() - start_time))