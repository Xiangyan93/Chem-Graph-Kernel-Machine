#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF
)
import numpy as np
import pandas as pd


class BaseKernelConfig:
    def __init__(self, N_RBF: int = 0,
                 sigma_RBF: List[float] = [1.0],
                 sigma_RBF_bound: List[Tuple[float, float]] = ['fixed']):
        assert (self.__class__ != BaseKernelConfig)
        self.N_RBF = N_RBF
        self.sigma_RBF = sigma_RBF
        self.sigma_RBF_bound = sigma_RBF_bound

    def _get_rbf_kernel(self) -> List:
        if self.N_RBF != 0:
            if len(self.sigma_RBF) != 1 and len(self.sigma_RBF) != self.N_RBF:
                raise RuntimeError('molfeatures and hyperparameters must be the'
                                   ' same length')
            add_kernel = RBF(length_scale=self.sigma_RBF,
                             length_scale_bounds=self.sigma_RBF_bound)
        # ConstantKernel(1.0, (1e-3, 1e3)) * \
            return [add_kernel]
        else:
            return []
