#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import os
import json
import pickle

import numpy as np
from hyperopt import hp
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF
)


class BaseKernelConfig:
    def __init__(self, N_RBF: int = 0,
                 sigma_RBF: List[float] = None,
                 sigma_RBF_bounds: List[Tuple[float, float]] = None):
        self.N_RBF = N_RBF
        self.sigma_RBF = sigma_RBF
        self.sigma_RBF_bounds = sigma_RBF_bounds
        self.kernel = self._get_rbf_kernel()[0]

    def get_kernel_dict(self, X: np.ndarray, X_labels: np.ndarray) -> Dict:
        K = self.kernel(X)
        return {
            'X': X_labels,
            'K': K,
            'theta': self.kernel.theta
        }

    def _update_kernel(self):
        self.kernel = self._get_rbf_kernel()[0]

    def _get_rbf_kernel(self) -> List:
        if self.N_RBF != 0:
            if len(self.sigma_RBF) != 1 and len(self.sigma_RBF) != self.N_RBF:
                raise RuntimeError('features_mol and hyperparameters must be the'
                                   ' same length')
            add_kernel = RBF(length_scale=self.sigma_RBF,
                             length_scale_bounds=self.sigma_RBF_bounds)
        # ConstantKernel(1.0, (1e-3, 1e3)) * \
            return [add_kernel]
        else:
            return [None]

    # functions for Bayesian optimization of hyperparameters.
    def get_space(self):
        SPACE = dict()
        if self.sigma_RBF is not None:
            for i in range(len(self.sigma_RBF)):
                hp_key = 'RBF:%d:' % i
                hp_ = self._get_hp(hp_key, [self.sigma_RBF[i],
                                           self.sigma_RBF_bounds[i]])
                if hp_ is not None:
                    SPACE[hp_key] = hp_
        return SPACE

    def update_from_space(self, hyperdict: Dict[str, Union[int, float]]):
        for key, value in hyperdict.items():
            n, term, microterm = key.split(':')
            # RBF kernels
            if n == 'RBF':
                n_rbf = int(term)
                self.sigma_RBF[n_rbf] = value
        self._update_kernel()

    @staticmethod
    def _get_hp(key, value):
        if value[1] == 'fixed':
            return None
        elif value[0] in ['Additive', 'Tensorproduct']:
            return hp.choice(key, value[1])
        elif len(value) == 2:
            return hp.uniform(key, low=value[1][0], high=value[1][1])
        elif len(value) == 3:
            return hp.quniform(key, low=value[1][0], high=value[1][1],
                               q=value[2])
        else:
            raise RuntimeError('.')

    # save functions.
    def save_hyperparameters(self, path: str):
        if self.sigma_RBF is not None:
            rbf = {
                'sigma_RBF': self.sigma_RBF,
                'sigma_RBF_bounds': self.sigma_RBF_bounds
            }
            open(os.path.join(path, 'sigma_RBF.json'), 'w').write(
                json.dumps(rbf, indent=1, sort_keys=False))

    def save_kernel_matrix(self, path: str, X: np.ndarray, X_labels: List[str]):
        """Save kernel.pkl file that used for preCalc kernels."""
        kernel_dict = self.get_kernel_dict(X, X_labels)
        kernel_pkl = os.path.join(path, 'kernel.pkl')
        pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)
