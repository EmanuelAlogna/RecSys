from src.Recommender_utils import similarityMatrixTopK, removeTopPop
from src.Compute_Similarity_Python import *
import pyximport
pyximport.install()
from src.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from src.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from src.Recommender2 import Recommender
from src.metrics import *
from sklearn.preprocessing import normalize,maxabs_scale


class MABPR_Cython(SimilarityMatrixRecommender, Recommender, Incremental_Training_Early_Stopping):

    def __init__(self, URM_train, ICM,URM_test,UF= None, positive_threshold=4, URM_validation = None,
                 recompile_cython = False, final_model_sparse_weights = True, train_with_sparse_weights = False,
                 symmetric = True):


        super(MABPR_Cython, self).__init__()


        self.URM_train = URM_train.copy()
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.positive_threshold = positive_threshold
        self.ICM = ICM
        self.n_features = ICM.shape[1]
        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights
        self.UF = UF
        self.URM_test = URM_test
        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None


        if self.train_with_sparse_weights:
            self.sparse_weights = True

        assert self.URM_train.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, " \
                                       "positive threshold is too high"


        self.symmetric = symmetric

        if not self.train_with_sparse_weights:

            n_items = URM_train.shape[1]
            requiredGB = 8 * n_items**2 / 1e+06

            if symmetric:
                requiredGB /=2

            print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(n_items, requiredGB))


        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")


    def fit(self, epochs=25,negative_items = False, logFile=None,
            batch_size = 1000, lambda_i = 0.0, lambda_j = 0.0, learning_rate = 0.0075, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta_1=0.9, beta_2=0.999,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "MAP",
            evaluator_object = None, validation_every_n = 1):


        # Import compiled module
        from src.Cython.MABPR_Cython_Epoch import MABPR_Cython_Epoch

        self.sgd_mode = sgd_mode
        self.epochs = epochs



        #sim = np.zeros((self.n_items, self.n_items))

        self.cythonEpoch = MABPR_Cython_Epoch(self.URM_train,
                                              UF = self.UF,
                                              ICM = self.ICM,
                                              n_features = self.n_features,
                                              neg_items = negative_items,
                                                 train_with_sparse_weights = self.train_with_sparse_weights,
                                                 final_model_sparse_weights = self.sparse_weights,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 li_reg = lambda_i,
                                                 lj_reg = lambda_j,
                                                 batch_size=1,
                                                 symmetric = self.symmetric,
                                                 sgd_mode = sgd_mode,
                                                 gamma=gamma,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2)




        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        # if evaluator_object is None and stop_on_validation:
        #     evaluator_object = SequentialEvaluator(self.URM_validation, [5])


        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate


        # self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
        #                             validation_metric, lower_validatons_allowed, evaluator_object,
        #                             algorithm_name = self.RECOMMENDER_NAME)

        self.train()

        self.get_S_incremental_and_set_W()
        #self.W_sparse = normalize(self.W_sparse,norm='l2')
        sys.stdout.flush()
        return self.W_sparse


    def recommend2(self, user_id, at=None, exclude_seen=True,filter_top_pop= True):
        # compute the scores using the dot product

        # if filter_top_pop:
        #     removeTopPop(self.URM_train,percentageToRemove=0.2)
        #
        # user_profile = self.URM_train[user_id]
        # scores = user_profile.dot(self.W_sparse).toarray().ravel()
        if self.sparse_weights:
            user_profile = self.W_sparse[user_id]

            scores = user_profile.dot(self.ICM.T).toarray().ravel()

        else:
            user_profile = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            user_ratings = self.URM_train.data[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)


        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]



        return ranking[:at]



    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id+1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def _run_epoch(self):
       self.cythonEpoch.epochIteration_Cython(self.ICM)


    def get_URM_train(self):
        return self.URM_train.copy()

    def train(self):
        current_epoch = 0
        while current_epoch < self.epochs:
            print("Epoch {} of {}".format(current_epoch, self.epochs))
            self._run_epoch()

            # if current_epoch % 2 == 0 and current_epoch != 0 :
            #     self.W_sparse = normalize(self.cythonEpoch.get_S(),norm='l2')
            #     #print(self.W_sparse)
            #     print(self.W_sparse)
            #     evaluate_algorithm(self.URM_test,self)
            """
            if current_epoch > 13:
                self.get_S_incremental_and_set_W()

                evaluate_algorithm(self.URM_test, self)
            """

            current_epoch += 1


    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
        else:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S_incremental, k = self.topK)
            else:
                self.W = self.S_incremental


    def get_scores(self,user_id,exclude_seen = True):
        user_profile = self.W_sparse[user_id]

        scores = user_profile.dot(self.ICM.T).toarray().ravel()
        scores = maxabs_scale(scores)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        return scores


    def getURM(self):
        return np.dot(self.W_sparse,self.ICM.T)