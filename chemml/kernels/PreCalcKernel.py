#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from chemml.kernels.BaseKernelConfig import BaseKernelConfig
from chemml.kernels.MultipleKernel import *


class PreCalcKernel:
    def __init__(self, X, K, theta):
        self.X = X
        self.K = K
        self.theta_ = theta
        self.exptheta = np.exp(self.theta_)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X).ravel()
        Y_idx = np.searchsorted(self.X, Y).ravel() if Y is not None else X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], \
                   np.zeros((len(X_idx), len(Y_idx), 1))
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
        return ()

    @property
    def theta(self):
        return np.log(self.exptheta)

    @theta.setter
    def theta(self, value):
        self.exptheta = np.exp(value)

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

'''
def _Kc(self, super, x, y, eval_gradient=False):
    x, x_weight = self.x2graph(x), self.x2weight(x)
    y, y_weight = self.x2graph(y), self.x2weight(y)
    if eval_gradient:
        if not x and not y:
            return 1.0, np.zeros(len(super.theta))
        elif not x or not y:
            return 0., np.zeros(len(super.theta))
        else:
            Kxy, dKxy = super.__call__(x, Y=y, eval_gradient=True)
            Kxx, dKxx = super.__call__(x, eval_gradient=True)
            Kyy, dKyy = super.__call__(y, eval_gradient=True)
            Fxy = np.einsum("i,j,ij", x_weight, y_weight, Kxy)
            dFxy = np.einsum("i,j,ijk->k", x_weight, y_weight, dKxy)
            Fxx = np.einsum("i,j,ij", x_weight, x_weight, Kxx)
            dFxx = np.einsum("i,j,ijk->k", x_weight, x_weight, dKxx)
            Fyy = np.einsum("i,j,ij", y_weight, y_weight, Kyy)
            dFyy = np.einsum("i,j,ijk->k", y_weight, y_weight, dKyy)

            def get_reaction_smarts(g, g_weight):
                reactants = []
                products = []
                for i, weight in enumerate(g_weight):
                    if weight > 0:
                        reactants.append(g.smiles)
                    elif weight < 0:
                        products.append(g.smiles)
                return '.'.join(reactants) + '>>' '.'.join(products)

            if Fxx <= 0.:
                raise Exception('trivial reaction: ', get_reaction_smarts(x, x_weight))
            if Fyy == 0.:
                raise Exception('trivial reaction: ', get_reaction_smarts(y, y_weight))
            sqrtFxxFyy = np.sqrt(Fxx * Fyy)
            return Fxy / sqrtFxxFyy, \
                   (dFxy - 0.5 * dFxx / Fxx - 0.5 * dFyy / Fyy) / sqrtFxxFyy
    else:
        if not x and not y:
            return 1.0
        elif not x or not y:
            return 0.
        else:
            Kxy = super.__call__(x, Y=y)
            Kxx = super.__call__(x)
            Kyy = super.__call__(y)
            Fxy = np.einsum("i,j,ij", x_weight, y_weight, Kxy)
            Fxx = np.einsum("i,j,ij", x_weight, x_weight, Kxx)
            Fyy = np.einsum("i,j,ij", y_weight, y_weight, Kyy)
            sqrtFxxFyy = np.sqrt(Fxx * Fyy)
            return Fxy / sqrtFxxFyy


def _call(self, X, Y=None, eval_gradient=False, *args, **kwargs):
    if Y is None:
        Xidx, Yidx = np.triu_indices(len(X), k=1)
        Xidx, Yidx = Xidx.astype(np.uint32), Yidx.astype(np.uint32)
        Y = X
        symmetric = True
    else:
        Xidx, Yidx = np.indices((len(X), len(Y)), dtype=np.uint32)
        Xidx = Xidx.ravel()
        Yidx = Yidx.ravel()
        symmetric = False

    K = np.zeros((len(X), len(Y)))
    if eval_gradient:
        K_gradient = np.zeros((len(X), len(Y), self.theta.shape[0]))
        for i in Xidx:
            for j in Yidx:
                K[i][j], K_gradient[i][j] = self.Kc(X[i], Y[j],
                                                    eval_gradient=True)
    else:
        for n in range(len(Xidx)):
            i = Xidx[n]
            j = Yidx[n]
            K[i][j] = self.Kc(X[i], Y[j])
    if symmetric:
        K = K + K.T
        K[np.diag_indices_from(K)] += 1.0
        if eval_gradient:
            K_gradient = K_gradient + K_gradient.transpose([1, 0, 2])

    if eval_gradient:
        return K, K_gradient
    else:
        return K


class ConvolutionPreCalcKernel(PreCalcKernel):
    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        return _call(self, X, Y=None, eval_gradient=eval_gradient,
                     *args, **kwargs)

    def Kc(self, x, y, eval_gradient=False):
        return _Kc(self, super(), x, y, eval_gradient=eval_gradient)

    @staticmethod
    def x2graph(x):
        return x[::2]

    @staticmethod
    def x2weight(x):
        return x[1::2]
'''


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
            self.kernel = MultipleKernel(
                kernel_list=kernels,
                composition=composition,
                combined_rule='product',
            )

    def get_preCalc_kernel(self, kernel_dict: Dict):
        # kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
        X = kernel_dict['group_id']
        K = kernel_dict['K']
        theta = kernel_dict['theta']
        return PreCalcKernel(X, K, theta)
