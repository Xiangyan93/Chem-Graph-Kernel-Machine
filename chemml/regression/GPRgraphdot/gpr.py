#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import math
import numpy as np
from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor


class GPR(GaussianProcessRegressor):
    def predict(self, X, return_std=False, return_cov=False, memory_save=True):
        if return_cov or not memory_save:
            return super().predict(X, return_std=return_std,
                                 return_cov=return_cov)
        else:
            N = X.shape[0]
            y_mean = np.array([])
            y_std = np.array([])
            for i in range(math.ceil(N / 1000)):
                X_ = X[i * 1000:(i + 1) * 1000]
                if return_std:
                    y_mean_, y_std_ = super().predict(
                        X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = super().predict(
                        X_, return_std=return_std, return_cov=return_cov)
                y_mean = np.r_[y_mean, y_mean_]
            if return_std:
                return y_mean, y_std
            else:
                return y_mean

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        model = cls(kernel)
        model.__dict__.update(**store_dict)
        return model

    """sklearn GPR parameters"""
    @property
    def kernel_(self):
        return self.kernel

    @property
    def X_train_(self):
        return self._X

    @X_train_.setter
    def X_train_(self, value):
        self._X = value
