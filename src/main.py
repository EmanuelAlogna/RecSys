from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time
import matplotlib.pyplot as plt


dr = DataReader("train.csv", split_train_test=True)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

ICM_all = dr.build_icm("tracks.csv");
file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])

start_time = time.time()
ItemCF = ItemBasedCollaborativeRS(URM_train)
ItemCF.fit(top_k=50,shrink=50)
evaluate_algorithm(URM_test,ItemCF)

#make_recommendations(ItemCF,target_playlist,URM_train)

###                  ###
### PARAMETER TUNING ###
###                  ###

# x_tick = [10,50,100,200,500]
# MAP_per_k = []
# for k in x_tick:
#     cbr.fit(top_k= k , shrink= 10)
#     result = evaluate_algorithm(URM_test, cbr)
#     MAP_per_k.append(result)

# plt.plot(x_tick, MAP_per_k)
# plt.ylabel('MAP')
# plt.xlabel('Shrink')
# plt.show()

print("Total time: {}".format(time.time() - start_time))