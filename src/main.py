from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time
import matplotlib.pyplot as plt
#from src.SLIM_BPR_Recommender import *
from src.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps
from src.SLIM_BPR_Python import SLIM_BPR_Recommender
from src.SLIM_BPR_ElasticNet import SLIMElasticNetRecommender
dr = DataReader("train.csv", split_train_test=True,
                train_test_ratio=0.8,build_validation=True,train_validation_ratio=0.9)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test
URM_val = dr.URM_validation
# print(URM_all[7])
# print('\n')
# print(URM_train[7])
# print('\n')
# print(URM_test[7])
# print('\n')
# print(URM_val[7])

# file = pd.read_csv('../data/target_playlists.csv')
# ordered_playlists = np.array(file['playlist_id'][0:5000])
# file2 = pd.read_csv('../data/train.csv')
#
# def initialize_S(playlists):
#     S = dict()
#     for i in ordered_playlists:
#         S['{}'.format(i)] =list(file2[file2['playlist_id'] == i].get('track_id'))
#     return S

# S = initialize_S(ordered_playlists)
# print(S)
# print(type(S))
# X = sps.csr_matrix(S)
# print(X)
# def sample_ordered_playlist(playlist_id,S):
#     tracks = S['{}'.format(playlist_id)]
#     track_in_last_position = True
#     while track_in_last_position:
#         pos_item = np.random.choice(tracks)
#         index = np.where(tracks == pos_item)
#         index = int(index[0])
#         if index != (len(tracks)-1):
#             track_in_last_position = False
#
#     neg_item  = np.random.choice(tracks[index+1:])
#     return pos_item,neg_item
#
#
#
#
# def initialize_W(epochs = 20 ,w = None):
#     n_items = URM_all.shape[1]
#     W = w
#     start_time = time.time()
#     for epoch in np.arange(epochs):
#         if epoch % 2 == 0:
#             print("Epoch {} of {}".format(epoch, epochs))
#         epoch_iteration(ordered_playlists,W,learning_rate=0.1)
#     print("Total time: {}".format(time.time()-start_time))
#     return W
#
# def epoch_iteration(ordered_playlist,W,learning_rate = 0.1):
#     learning_rate = 0.1
#     for playlist_id in ordered_playlists:
#         pos_item,neg_item = sample_ordered_playlist(playlist_id,S)
#
#         userSeenItems = URM_train[playlist_id,:].indices
#
#         x_i = W[pos_item, userSeenItems].sum()
#         x_j = W[neg_item, userSeenItems].sum()
#
#         x_ij = x_i - x_j
#
#         gradient = 1 / (1 + np.exp(x_ij))
#
#         # Update
#         W[pos_item, userSeenItems] += learning_rate * gradient
#         W[pos_item, pos_item] = 0
#
#         W[neg_item, userSeenItems] -= learning_rate * gradient
#         W[neg_item, neg_item] = 0
#
#


# W = initialize_W(epochs=1)
# print(sps.csr_matrix(W)[0,460])
# print(sps.csr_matrix(W)[460,0])






#ICM_all = dr.build_icm("tracks.csv");
file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])




#make_recommendations(ItemCF,target_playlist,URM_train)

start_time = time.time()
# #
#W = initialize_W(epochs=10)
# BPR = SLIM_BPR_Cython(URM_train, recompile_cython=False,positive_threshold=1,
#                        train_with_sparse_weights=False,final_model_sparse_weights=False,
#                       W_init=None
#                       )
# BPR.fit(epochs=5, batch_size=1, sgd_mode='adagrad',topK=1000, learning_rate=0.1,lambda_i=0.001,lambda_j=0.001)
# evaluate_algorithm(URM_test, BPR)

# BPR = SLIM_BPR_Cython(URM_all,positive_threshold=1,train_with_sparse_weights=False,
#                       final_model_sparse_weights=True,symmetric=True)
#
# BPR.fit(epochs=6, batch_size=1, sgd_mode='adagrad',topK=500, learning_rate=0.1,lambda_i=0.001,lambda_j=0.001)
# evaluate_algorithm(URM_test,BPR)
# make_recommendations(BPR,target_playlists=target_playlist)


# x_tick = [10,11,12]
# # #shrink_tick = [0,10,50,100,200,500]
# MAP_per_k = []
# for k in x_tick:
#     BPR.fit(epochs=k, batch_size=1, sgd_mode='adagrad',topK=200, learning_rate=0.1,lambda_i=0.001,lambda_j=0.001)
#     print("Number of epochs:" + str(k) )
#     result = evaluate_algorithm(URM_val, BPR)
#     MAP_per_k.append(result)
#
# plt.plot(x_tick, MAP_per_k)
# plt.ylabel('MAP')
# plt.xlabel('epochs')
# plt.show()




# ItemCF = ItemBasedCollaborativeRS(URM_train)
# ItemCF.fit(top_k=800,shrink=0)
# evaluate_algorithm(URM_test,ItemCF)
#make_recommendations(ItemCF,target_playlist)

from src.ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from src.Evaluation.Evaluator import SequentialEvaluator

evaluator_validation_earlystopping = SequentialEvaluator(URM_val, cutoff_list=[5])
evaluator_test = SequentialEvaluator(URM_test, cutoff_list=[5, 10])

evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
evaluator_test = EvaluatorWrapper(evaluator_test)

from src.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.ParameterTuning.BayesianSearch import BayesianSearch
from src.SLIM_BPR_Cython import SLIM_BPR_Cython

recommender_class = SLIM_BPR_Cython

parameterSearch = BayesianSearch(recommender_class,
                                 evaluator_validation=evaluator_validation,
                                 evaluator_test=evaluator_test)

from src.ParameterTuning.AbstractClassSearch import DictionaryKeys

metric_to_optimize = 'MAP'

hyperparamethers_range_dictionary = {}
hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
#hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
#hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
hyperparamethers_range_dictionary["learning_rate"] = [1e-1, 1e-2, 1e-3,1e-4,1e-5]
hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-3, 1e-5, 1e-7]
hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-3, 1e-5, 1e-7]

recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                        DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':True, 'positive_threshold':0},
                        DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                        DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True,
                                                            "evaluator_object":evaluator_validation_earlystopping,
                                                            "lower_validatons_allowed":10, "validation_metric":metric_to_optimize},
                        DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
output_root_path = "result_experiments/"

import os

# If directory does not exist, create
if not os.path.exists(output_root_path):
    os.makedirs(output_root_path)

output_root_path += recommender_class.RECOMMENDER_NAME

n_cases = 35
metric_to_optimize = "MAP"

best_parameters = parameterSearch.search(recommenderDictionary,
                                         n_cases = n_cases,
                                         output_root_path = output_root_path,
                                         metric=metric_to_optimize)

print("Total time: {}".format(time.time() - start_time))