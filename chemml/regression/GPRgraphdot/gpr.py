#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import math
import numpy as np
from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor


class GPR(GaussianProcessRegressor):
    def fit(self, *args, **kwargs):
        self.X_id_ = kwargs.pop('id', None)
        return super().fit(*args, **kwargs)

    def predict_(self, Z, return_std=False, return_cov=False):
        """Predict using the trained GPR model.

        Parameters
        ----------
        Z: list of objects or feature vectors.
            Input values of the unknown data.
        return_std: boolean
            If True, the standard-deviations of the predictions at the query
            points are returned along with the mean.
        return_cov: boolean
            If True, the covariance of the predictions at the query points are
            returned along with the mean.

        Returns
        -------
        ymean: 1D array
            Mean of the predictive distribution at query points.
        std: 1D array
            Standard deviation of the predictive distribution at query points.
        cov: 2D matrix
            Covariance of the predictive distribution at query points.
        """
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(Z, diag=True)
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std)
        elif return_cov is True:
            Kss = self._gramian(Z)
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov)
        else:
            return ymean

    def predict(self, X, return_std=False, return_cov=False, memory_save=True,
                n_memory_save=1000):
        if return_cov or not memory_save:
            return super().predict(X, return_std=return_std,
                                   return_cov=return_cov)
        else:
            N = X.shape[0]
            y_mean = np.array([])
            y_std = np.array([])
            for i in range(math.ceil(N / n_memory_save)):
                X_ = X[i * n_memory_save:(i + 1) * n_memory_save]
                if return_std:
                    [y_mean_, y_std_] = self.predict_(
                        X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = self.predict_(
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
