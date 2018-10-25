from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time

dr = DataReader("train.csv", split_train_test=True)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

rr = RandomRecommender()

topR = TopPopularRecommender()

ICM_all = dr.build_icm("tracks.csv");
print(ICM_all[0:10])

file = pd.read_csv("../data/train.csv")
list_playlist = list(set(file['playlist_id']))

file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])
#print(URM_all[0:10])
#start_time = time.time()


#make_recommendations(topR, target_playlist, URM_train)

#evaluate_algorithm(URM_test,topR)
#print("Total time: {}".format(time.time() - start_time))


