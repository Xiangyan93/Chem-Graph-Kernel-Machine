#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
from sklearn.svm import SVC


class SVMClassifier(SVC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._SVC = SVC(*args, **kwargs)

    @property
    def kernel_(self):
        return self._SVC.kernel

    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y!=None)[0]
        elif y.dtype == float:
            idx = ~np.isnan(y)
        else:
            idx = np.arange(len(y))
        return np.asarray(X)[idx], y[idx]  #.astype(int)

    def fit(self, X, y, sample_weight=None):
        self.SVCs = []
        if y.ndim == 1:
            X_, y_ = self._remove_nan_X_y(X, y)
            super().fit(X_, y_, sample_weight)
        else:
            for i in range(y.shape[1]):
                SVC = copy.deepcopy(self._SVC)
                X_, y_ = self._remove_nan_X_y(X, y[:, i])
                SVC.fit(X_, y_, sample_weight)
                self.SVCs.append(SVC)

    def predict(self, X):
        if self.SVCs:
            y_mean = []
            for SVC in self.SVCs:
                y_mean.append(SVC.predict(X))
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict(X)

    def predict_proba(self, X):
        if self.SVCs:
            y_mean = []
            for SVC in self.SVCs:
                y_mean.append(SVC.predict_proba(X)[:, 1])
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict_proba(X)[:, 1]