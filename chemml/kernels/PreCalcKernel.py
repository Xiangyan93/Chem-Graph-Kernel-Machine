#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple

import numpy as np

from .BaseKernelConfig import BaseKernelConfig
from .HybridKernel import *


class PreCalcKernel:
    def __init__(self, X: np.ndarray, K: np.ndarray, theta: np.ndarray):
        idx = np.argsort(X)
        self.X = X[idx]
        self.K = K[idx][:, idx]
        self.theta_ = theta
        self.hyperparameters_ = np.exp(self.theta_)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X.ravel())
        Y_idx = np.searchsorted(self.X, Y.ravel()) if Y is not None else X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], np.zeros((len(X_idx), len(Y_idx), 1))
        else:
            return self.K[X_idx][:, Y_idx]

    def diag(self, X, eval_gradient=False):
        X_idx = np.searchsorted(self.X, X).ravel()
        if eval_gradient:
            return np.diag(self.K)[X_idx], np.zeros((len(X_idx), 1))
        else:
            return np.diag(self.K)[X_idx]

    @property
    def hyperparameters(self):
        return self.hyperparameters_

    @property
    def theta(self):
        return np.log(self.hyperparameters_)

    @theta.setter
    def theta(self, value):
        self.hyperparameters_ = np.exp(value)

    @property
    def n_dims(self):
        return len(self.theta)

    @property
    def bounds(self):
        theta = self.theta.reshape(-1, 1)
        return np.c_[theta, theta]

    @property
    def requires_vector_input(self):
        return False

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            X=self.X,
            K=self.K,
            theta=self.theta_
        )


class PreCalcKernelConfig(BaseKernelConfig):
    def __init__(self, kernel_dict: Dict,
                 N_RBF: int = 0,
                 sigma_RBF: List[float] = None,
                 sigma_RBF_bounds: List[Tuple[float, float]] = None):
        super().__init__(N_RBF, sigma_RBF, sigma_RBF_bounds)
        self.type = 'preCalc'
        if N_RBF == 0:
            self.kernel = self.get_preCalc_kernel(kernel_dict)
        else:
            kernels = [self.get_preCalc_kernel(kernel_dict)]
            kernels += self._get_rbf_kernel()
            composition = [(0,)] + \
                          [tuple(np.arange(1, N_RBF + 1))]
            self.kernel = HybridKernel(
                kernel_list=kernels,
                composition=composition,
                hybrid_rule='product',
            )

    @staticmethod
    def get_preCalc_kernel(kernel_dict: Dict):
        X = kernel_dict['X']
        K = kernel_dict['K']
        theta = kernel_dict['theta']
        return PreCalcKernel(X, K, theta)
