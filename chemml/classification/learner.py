import pandas as pd
import numpy as np
import sys
from chemml.base_learner import KernelClassificationBaseLearner
from .gpc.gpc import GPC
from .svm.svm import SVC


class ClassificationLearner(KernelClassificationBaseLearner):
    def train(self, train_X=None, train_y=None, train_id=None):
        if train_X is None:
            train_X = self.train_X
        if train_y is None:
            train_y = self.train_Y
        if train_id is None:
            train_id = self.train_id
        if self.model.__class__ == GPC:
            self.model.fit(train_X, train_y)
            self.model.X_id_ = train_id
            print('hyperparameter: ', self.model.kernel_.hyperparameters)
        elif self.model.__class__ == SVC:
            self.model.fit(train_X, train_y)
            self.model.X_id_ = train_id
            print('hyperparameter: ', self.model.kernel.hyperparameters)
        else:
            raise RuntimeError(f'Unknown classifier {self.model}')

    def evaluate_test(self):
        x = self.test_X
        y = self.test_Y
        y_pred = self.model.predict(x)
        return self.evaluate_df_(x, y, y_pred, self.test_id,
                                 n_most_similar=None)

    def evaluate_train(self):
        x = self.train_X
        y = self.train_Y
        y_pred = self.model.predict(x)
        return self.evaluate_df_(x, y, y_pred, self.train_id)

    def evaluate_loocv(self):
        x = self.train_X
        y = self.train_Y
        y_pred = self.model.predict_loocv(x, y)
        return self.evaluate_df_(x, y, y_pred, self.train_id, n_most_similar=5)
