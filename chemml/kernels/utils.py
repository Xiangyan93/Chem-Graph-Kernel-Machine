# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pickle
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from ..args import KernelArgs
from ..data import Dataset


def get_features_hyperparameters(args: KernelArgs, N_RBF: int) -> \
        Tuple[Optional[List[float]], Optional[List[Tuple[float, float]]]]:
    if N_RBF == 0:
        sigma_RBF, sigma_RBF_bounds = None, None
    elif args.features_hyperparameters_file is not None:
        rbf = json.load(open(args.features_hyperparameters_file))
        sigma_RBF = rbf['sigma_RBF']
        sigma_RBF_bounds = rbf['sigma_RBF_bounds']
    else:
        sigma_RBF = args.features_hyperparameters
        sigma_RBF_bounds = args.features_hyperparameters_bounds
        if len(sigma_RBF) != 1 and len(sigma_RBF) != N_RBF:
            raise RuntimeError(f'The number of features({N_RBF}) not equal to the number of hyperparameters'
                               f'({len(sigma_RBF)})')
        elif sigma_RBF_bounds != 'fixed' \
                and len(sigma_RBF) == 1 \
                and N_RBF != 1 \
                and not args.single_features_hyperparameter:
            sigma_RBF *= N_RBF
            sigma_RBF_bounds *= N_RBF
    return sigma_RBF, sigma_RBF_bounds


def get_kernel_config(args: KernelArgs, dataset: Dataset,
                      kernel_dict: Dict = None):
    if args.graph_kernel_type is None:
        N_RBF = dataset.N_features_mol + dataset.N_features_add
        assert N_RBF != 0
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            args, N_RBF
        )
        params = {
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,  # np.concatenate(sigma_RBF),
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from chemml.kernels.BaseKernelConfig import BaseKernelConfig
        return BaseKernelConfig(**params)
    elif args.graph_kernel_type == 'graph':
        graph_hyperparameters = [
            json.load(open(j)) for j in args.graph_hyperparameters
        ]
        assert dataset.N_MGK + dataset.N_conv_MGK == len(graph_hyperparameters)

        N_RBF = dataset.N_features_mol + dataset.N_features_add
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            args, N_RBF
        )

        params = {
            'N_MGK': dataset.N_MGK,
            'N_conv_MGK': dataset.N_conv_MGK,
            'graph_hyperparameters': graph_hyperparameters,
            'unique': False,
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,  # np.concatenate(sigma_RBF),
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from chemml.kernels.GraphKernel import GraphKernelConfig
        return GraphKernelConfig(**params)
    else:
        N_RBF = 0 if dataset.data[0].features_add is None \
            else dataset.data[0].features_add.shape[1]
        sigma_RBF, sigma_RBF_bounds = get_features_hyperparameters(
            args, N_RBF
        )

        if kernel_dict is None:
            kernel_pkl = os.path.join(args.save_dir, 'kernel.pkl')
            kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
        params = {
            'kernel_dict': kernel_dict,
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        from chemml.kernels.PreCalcKernel import PreCalcKernelConfig
        return PreCalcKernelConfig(**params)


def graph2preCalc(dataset: Dataset, kernel_config):
    X = dataset.X_mol
    kernel = kernel_config.kernel
    K = kernel(X)
    kernel_dict = {
        'group_id': dataset.X_gid.ravel(),
        'K': K,
        'theta': kernel.theta
    }
