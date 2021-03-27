import os
import json
from typing import Dict, Iterator, List, Optional, Union, Literal
import numpy as np
from chemml.args import KernelArgs
from chemml.data import Dataset
from chemml.kernels.GraphKernel import GraphKernelConfig
from chemml.kernels.PreCalcKernel import PreCalcKernelConfig


def get_kernel_info(args):
    N_MGK = 0
    N_conv_MGK = 0
    if args.pure_columns is not None:
        N_MGK += len(args.pure_columns)
    if args.mixture_columns is not None:
        if args.mixture_type == 'single_graph':
            N_MGK += len(args.mixture_columns)
        else:
            N_conv_MGK += len(args.mixture_columns)
    return N_MGK, N_conv_MGK


def get_kernel_config(args: KernelArgs, dataset: Dataset):
    N_MGK, N_conv_MGK = get_kernel_info(args)
    if args.kernel_type == 'graph':
        graph_hyperparameters = [
            json.load(open(j)) for j in args.graph_hyperparameters
        ]
        assert N_MGK + N_conv_MGK == len(graph_hyperparameters)

        N_RBF_molfeatures = 0 if dataset.data[0]._X_molfeatures is None \
            else dataset.data[0]._X_molfeatures.shape[1]
        N_RBF_addfeatures = 0 if dataset.data[0].addfeatures is None \
            else dataset.data[0].addfeatures.shape[1]
        N_RBF = N_RBF_molfeatures + N_RBF_addfeatures

        if args.features_hyperparameters_file is not None:
            rbf = json.load(open(args.features_hyperparameters_file))
            sigma_RBF = rbf['sigma_RBF']
            sigma_RBF_bounds = rbf['sigma_RBF_bounds']
        else:
            sigma_RBF = args.features_hyperparameters
            sigma_RBF_bounds = [(
                args.features_hyperparameters_min[i],
                args.features_hyperparameters_max[i])
                for i in range(len(args.features_hyperparameters))]
        params = {
            'N_MGK': N_MGK,
            'N_conv_MGK': N_conv_MGK,
            'graph_hyperparameters': graph_hyperparameters,
            'unique': False,
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,# np.concatenate(sigma_RBF),
            'sigma_RBF_bound': sigma_RBF_bounds, # * N_RBF,
        }
        return GraphKernelConfig(**params)
    else:
        N_RBF = 0 if dataset.data[0].addfeatures is None \
            else dataset.data[0].addfeatures.shape[1]

        if args.features_hyperparameters_file is not None:
            rbf = json.load(open(args.features_hyperparameters_file))
            sigma_RBF = rbf['sigma_RBF']
            sigma_RBF_bounds = rbf['sigma_RBF_bounds']
        else:
            sigma_RBF = args.features_hyperparameters
            if args.features_hyperparameters is None:
                sigma_RBF_bounds = None
            else:
                sigma_RBF_bounds = [(
                    args.features_hyperparameters_min[i],
                    args.features_hyperparameters_max[i])
                    for i in range(len(args.features_hyperparameters))]
        params = {
            'f_kernel': os.path.join(args.save_dir, 'kernel.pkl'),
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,
            'sigma_RBF_bounds': sigma_RBF_bounds,  # * N_RBF,
        }
        return PreCalcKernelConfig(**params)
