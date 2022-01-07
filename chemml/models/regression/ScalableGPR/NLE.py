#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from graphdot.linalg.cholesky import CholSolver


class NaiveLocalExpertGP:
    """Transductive Naive Local Experts of Gaussian process regression.

    """
    def __init__(self, kernel, alpha=1e-8, n_local=500, normalize_y=False, kernel_options={}):
        self.kernel = kernel
        self.alpha = alpha
        self.n_local = n_local
        self.normalize_y = normalize_y
        self.kernel_options = kernel_options

    @property
    def X(self):
        '''The input values of the training set.'''
        try:
            return self._X
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provide using fit().'
            )

    @X.setter
    def X(self, X):
        self._X = np.asarray(X)

    @property
    def y(self):
        '''The output/target values of the training set.'''
        try:
            return self._y
        except AttributeError:
            raise AttributeError(
                'Training data does not exist. Please provide using fit().'
            )

    @y.setter
    def y(self, _y):
        if self.normalize_y is True:
            self.y_mean, self.y_std = np.mean(_y), np.std(_y)
            self._y = (np.asarray(_y) - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0, 1
            self._y = np.asarray(_y)

    def _gramian(self, X, Y=None, kernel=None, diag=False):
        kernel = kernel or self.kernel
        if Y is None:
            if diag is True:
                return kernel.diag(X, **self.kernel_options) + self.alpha
            else:
                K = kernel(X, **self.kernel_options)
                K.flat[::len(K) + 1] += self.alpha
                return K
        else:
            if diag is True:
                raise ValueError(
                    'Diagonal Gramian does not exist between two sets.'
                )
            else:
                return kernel(X, Y, **self.kernel_options)

    def _invert(self, K):
        try:
            return CholSolver(K), np.prod(np.linalg.slogdet(K))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                'Kernel matrix singular, it is likely corrupted with NaNs and Infs '
                'because a pseudoinverse could not be computed.'
            )

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, Z, return_std=False):
        y_mean = []
        y_std = []
        for z in tqdm(Z, total=len(Z)):
            Z_ = z.reshape(1, -1)
            Ks = self._gramian(Z_, self.X)
            local_idx = np.argsort(-Ks)[:, :min(self.n_local, Ks.shape[1])][0]
            Ks_local = Ks[:, local_idx]
            X_local = self.X[local_idx]
            y_local = self.y[local_idx]
            K_local = self._gramian(X_local)
            Kinv_local, _ = self._invert(K_local)
            Ky_local = Kinv_local @ y_local
            y_mean.append((Ks_local @ Ky_local) * self.y_std + self.y_mean)
            if return_std is True:
                Kss = self._gramian(Z, diag=True)
                std = np.sqrt(
                    np.maximum(0, Kss - (Ks_local @ (Kinv_local @ Ks_local.T)).diagonal())
                )
                y_std.append(std)
        if return_std:
            return np.concatenate(y_mean), np.concatenate(y_std)
        else:
            return np.concatenate(y_mean)
