#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF
)
import numpy as np
import pandas as pd


class KernelConfig:
    def __init__(self, N_RBF: int = 0, sigma_RBF: np.ndarray = 1.0):
        assert (self.__class__ != KernelConfig)
        self.N_RBF = N_RBF
        self.sigma_RBF = sigma_RBF

    def get_rbf_kernel(self) -> List:
        if self.N_RBF != 0:
            if self.sigma_RBF.__class__ == float:
                self.sigma_RBF *= np.ones(self.N_RBF)
            if self.N_RBF != len(self.sigma_RBF):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                         RBF(length_scale=self.sigma_RBF)
            return [add_kernel]
        else:
            return []
