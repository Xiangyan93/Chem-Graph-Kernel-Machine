#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import copy
import numpy as np


class HybridKernel:
    def __init__(self, kernel_list: List, composition: List[Tuple[int]],
                 hybrid_rule: Literal['product'] = 'product'):
        self.kernel_list = kernel_list
        self.composition = composition
        self.hybrid_rule = hybrid_rule

    @property
    def nkernel(self) -> int:
        return len(self.kernel_list)

    def get_X_list(self, X: np.ndarray) -> List[np.ndarray]:
        def f(c):
            return X[:, c]

        X = self._format_X(X)
        X_list = list(map(f, self.composition))
        return X_list

    @staticmethod
    def _format_X(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X.reshape(1, X.size)  # .tolist()
        else:
            return X

    def __call__(self, X: np.ndarray, Y: np.ndarray = None,
                 eval_gradient: bool = False):
        X_list = self.get_X_list(X)
        Y_list = self.get_X_list(Y) if Y is not None else None
        if eval_gradient:
            covariance_matrix = 1
            gradient_matrix_list = list(map(int, np.ones(self.nkernel).tolist()))
            for i, kernel in enumerate(self.kernel_list):
                Xi = X_list[i]
                Yi = Y_list[i] if Y is not None else None
                output = kernel(Xi, Y=Yi, eval_gradient=True)
                if self.hybrid_rule == 'product':
                    covariance_matrix *= output[0]
                    for j in range(self.nkernel):
                        if j == i:
                            gradient_matrix_list[j] = gradient_matrix_list[j] * output[1]
                        else:
                            shape = output[0].shape + (1,)
                            gradient_matrix_list[j] = gradient_matrix_list[j] * output[0].reshape(shape)
            gradient_matrix = gradient_matrix_list[0]
            for i, gm in enumerate(gradient_matrix_list):
                if i != 0:
                    gradient_matrix = np.c_[
                        gradient_matrix, gradient_matrix_list[i]]
            return covariance_matrix, gradient_matrix
        else:
            covariance_matrix = 1
            for i, kernel in enumerate(self.kernel_list):
                Xi = X_list[i]
                Yi = Y_list[i] if Y is not None else None
                output = kernel(Xi, Y=Yi, eval_gradient=False)
                if self.hybrid_rule == 'product':
                    covariance_matrix *= output
            return covariance_matrix

    def diag(self, X) -> List[float]:
        X_list = self.get_X_list(X)
        diag_list = [self.kernel_list[i].diag(X_list[i]) for i in
                     range(len(self.kernel_list))]
        if self.hybrid_rule == 'product':
            return np.product(diag_list, axis=0)
        else:
            raise Exception('Unknown combined rule %s' % self.hybrid_rule)

    def is_stationary(self):
        return False

    @property
    def requires_vector_input(self):
        return False

    @property
    def n_dims_list(self) -> List[int]:
        """Numbers of hyperparameters."""
        return [len(kernel.theta) for kernel in self.kernel_list]

    @property
    def n_dims(self) -> int:
        return sum(self.n_dims_list)

    @property
    def hyperparameters(self):
        return np.exp(self.theta)

    @property
    def theta(self):
        return np.concatenate([kernel.theta for kernel in self.kernel_list])

    @theta.setter
    def theta(self, value):
        if len(value) != len(self.theta):
            raise Exception('The length of n_dims and theta must the same')
        s = 0
        e = 0
        for i, kernel in enumerate(self.kernel_list):
            e += self.n_dims_list[i]
            kernel.theta = value[s:e]
            s += self.n_dims_list[i]

    @property
    def bounds(self):
        for i, kernel in enumerate(self.kernel_list):
            if i == 0:
                bounds = self.kernel_list[0].bounds
            else:
                bounds = np.r_[bounds, kernel.bounds]
        return bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            kernel_list=self.kernel_list,
            composition=self.composition,
            combined_rule=self.hybrid_rule,
        )

    def load(self, result_dir):
        for i, kernel in enumerate(self.kernel_list):
            if hasattr(kernel, 'PreCalculate'):
                kernel.load(result_dir)
