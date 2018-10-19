from src.DataReader import *
from src.Recommender import *
from src.metrics import *
import pandas as pd
import time

dr = DataReader("train.csv",split_train_test=True)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

rr = RandomRecommender()
rr.fit(URM_train)

file = pd.read_csv("./data/train.csv")
list_playlist = list(set(file['playlist_id']))

file2 = pd.read_csv("./data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])
start_time = time.time()
#evaluate_algorithm(URM_test,rr,list_playlist)
evaluate_algorithm(URM_test,rr,target_playlist)
print("Total time: {}".format(time.time()-start_time))
make_recommendations(rr,target_playlist,URM_train)
# # #recommended_item = rr.recommend(10)
# #
# # # rel_item = URM_test[0].indices
# # # print(URM_test[0])
# # # print(URM_test[1])
# # # prec = precision(recommended_item,rel_item)
# # # rec = recall(recommended_item,rel_item)
# #
# # file = pd.read_csv("./data/target_playlists.csv",'r')
# # #print(file['playlist_id'])
# # p_id = list(file['playlist_id'])
# #
# # acc_prec = 0
# # acc_rec = 0
# # # for i  in p_id:
# # #      recommended_item = rr.recommend(10)
# # #      rel_item = URM_all[i].indices
# # #      acc_prec += precision(recommended_item,rel_item)
# # #      acc_rec += recall(recommended_item,rel_item)
# # #
# # # avg_prec = acc_prec / len(p_id)
# # # avg_rec = acc_rec/ len(p_id)
# #
# # # print("Random Recommender")
# # # print("Average precision: {}".format(avg_prec))
# # # print("Average recall: {}".format(avg_rec))
# #
# #
# # playlist = list(set(dr.URM_all.tocoo().row))
# # print(playlist[0:10])
# #
# # #evaluate_algorithm(dr.URM_test,rr,playlist)
# #
# # help(np.flip)

# from src.DataReader import *
#
# dr = DataReader("train.csv",split_train_test=True ,train_test_ratio = 0.7)
# print(dr.URM_all)
# # dr.URM_train
# # dr.URM_test

# def make_recomm(recommender, target_playlist):
#
#     recommender.fit(urm_train)
#
#     for i in target_playlist:
#         r = recommend(i,10);
#
#     #create csr_matrix
#     #
#
# rr = randomRecommener()
#
# make_recomm(rr)