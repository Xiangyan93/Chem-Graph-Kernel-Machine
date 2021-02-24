#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF
)
import numpy as np
import pandas as pd


class KernelConfig:
    def __init__(self, add_features, add_hyperparameters, params):
        assert (self.__class__ != KernelConfig)
        self.add_features = add_features
        self.add_hyperparameters = add_hyperparameters
        self.params = params

    def get_rbf_kernel(self):
        if None not in [self.add_features, self.add_hyperparameters]:
            if len(self.add_features) != len(self.add_hyperparameters):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                         RBF(length_scale=self.add_hyperparameters)
            return [add_kernel]
        else:
            return []
