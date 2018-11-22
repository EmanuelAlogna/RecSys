from src.dataReader import *
from src.recommender import *
from src.metrics import *
import pandas as pd
import time
import matplotlib.pyplot as plt
#from src.SLIM_BPR_Recommender import *
from src.SLIM_BPR_Cython import SLIM_BPR_Cython
import scipy.sparse as sps

dr = DataReader("train.csv", split_train_test=True)
URM_all = dr.URM_all
URM_train = dr.URM_train
URM_test = dr.URM_test

ICM_all = dr.build_icm("tracks.csv");
file2 = pd.read_csv("../data/target_playlists.csv")
target_playlist = list(file2['playlist_id'])


# ItemCF = ItemBasedCollaborativeRS(URM_train)
# ItemCF.fit(top_k=50,shrink=50)
#evaluate_algorithm(URM_test,ItemCF)

#make_recommendations(ItemCF,target_playlist,URM_train)

###                  ###
### PARAMETER TUNING ###
###                  ###

#x_tick = [10,50,100,200,500]
#shrink_tick = [0,10,50,100,200,500]
#MAP_per_k = []
#for k in x_tick:
#    for j in shrink_tick:
#        ItemCF.fit(top_k = k , shrink = j)
#        print("K:" + str(k) + "| " + "SHRINK" + str(j))
#        result = evaluate_algorithm(URM_test, ItemCF)
#        MAP_per_k.append(result)

# plt.plot(x_tick, MAP_per_k)
# plt.ylabel('MAP')
# plt.xlabel('Shrink')
# plt.show()

# Sampling = BPR_Sampling(URM_train)
# A = Sampling.sampleTriple()

start_time = time.time()
#
BPR = SLIM_BPR_Cython(URM_train, recompile_cython=False,positive_threshold=1,train_with_sparse_weights=False,final_model_sparse_weights=True)
BPR.fit(epochs=5, batch_size=1, sgd_mode='adagrad',topK=1000, learning_rate=0.1,lambda_i=0.001,lambda_j=0.001)
evaluate_algorithm(URM_test, BPR)

# x_tick = [4,5,6,7]
# # #shrink_tick = [0,10,50,100,200,500]
# MAP_per_k = []
# for k in x_tick:
#     BPR.fit(epochs=k, batch_size=1, sgd_mode='adagrad',topK=200, learning_rate=0.1,lambda_i=0.001,lambda_j=0.001)
#     print("K:" + str(k) )
#     result = evaluate_algorithm(URM_test, BPR)
#     MAP_per_k.append(result)

plt.plot(x_tick, MAP_per_k)
plt.ylabel('MAP')
plt.xlabel('topK')
plt.show()

print("Total time: {}".format(time.time() - start_time))