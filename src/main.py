from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time
import src.Compute_Similarity_Python as csp

dr = DataReader("train.csv", split_train_test=True)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

rr = RandomRecommender()

topR = TopPopularRecommender()

ICM_all = dr.build_icm("tracks.csv");

file = pd.read_csv("../data/train.csv")
list_playlist = list(set(file['playlist_id']))

file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])
#print(URM_all[0:10])
#start_time = time.time()


#make_recommendations(topR, target_playlist, URM_train)

#evaluate_algorithm(URM_test,topR)
#print("Total time: {}".format(time.time() - start_time))
#
# CSP = csp.Compute_Similarity_Python(ICM_all.T)
# sim = CSP.compute_similarity()
# print(sim)
# CSP = csp.Compute_Similarity_Python(ICM_all.T,topK=50)
# sim = CSP.compute_similarity()
# print(sim)
#


start_time = time.time()
cbr = ItemCBFKNNRecommender(URM_train,ICM_all)
cbr.fit(top_k= 50 , shrink=0)
#rec = cbr.recommend(0,at=10)
evaluate_algorithm(URM_test,cbr)
print("Tempo totale: {}".format(time.time()-start_time))