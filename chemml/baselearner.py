import numpy as np


class BaseLearner:
    def __init__(self, train_X, train_Y, train_id, test_X, test_Y,
                 test_id, kernel_config, optimizer=None):
        assert (self.__class__ != BaseLearner)
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_id = train_id
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_id = test_id
        self.kernel_config = kernel_config
        self.kernel = kernel_config.kernel
        self.optimizer = optimizer
        self.model = None

    @staticmethod
    def get_most_similar_graphs(K, n=5):
        return np.argsort(-K)[:, :min(n, K.shape[1])]