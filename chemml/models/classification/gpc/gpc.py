#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.gaussian_process._gpc import GaussianProcessClassifier as GPC
import numpy as np
import copy


class GaussianProcessClassifier(GPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._GPC = GPC(*args, **kwargs)

    @property
    def kernel_(self):
        return self._GPC.kernel

    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y!=None)[0]
        else:
            idx = ~np.isnan(y)
        return np.asarray(X)[idx], y[idx].astype(int)

    def fit(self, X, y):
        self.GPCs = []
        if y.ndim == 1:
            X_, y_ = self._remove_nan_X_y(X, y)
            super().fit(X_, y_)
        else:
            for i in range(y.shape[1]):
                GPC = copy.deepcopy(self._GPC)
                X_, y_ = self._remove_nan_X_y(X, y[:, i])
                GPC.fit(X_, y_)
                self.GPCs.append(GPC)

    def predict(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                y_mean.append(GPC.predict(X))
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict(X)

    def predict_proba(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                y_mean.append(GPC.predict_proba(X)[:, 1])
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict_proba(X)[:, 1]