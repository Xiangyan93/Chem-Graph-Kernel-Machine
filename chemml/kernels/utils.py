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


def set_kernel(args: KernelArgs, dataset: Dataset):
    N_MGK, N_conv_MGK = get_kernel_info(args)
    if args.kernel_type == 'graph':
        graph_hyperparameters = [
            json.loads(open(j, 'r').readline())
            for j in args.graph_hyperparameters
        ]
        assert N_MGK + N_conv_MGK == len(graph_hyperparameters)
        N_RBF_molfeatures = 0 if dataset.data[0]._X_molfeatures is None \
            else dataset.data[0]._X_molfeatures.shape[1]
        N_RBF_addfeatures = 0 if dataset.data[0].addfeatures is None \
            else dataset.data[0].addfeatures.shape[1]
        N_RBF = N_RBF_molfeatures + N_RBF_addfeatures
        sigma_RBF = []
        if args.molfeatures_hyperparameters.__class__ == float:
            sigma_RBF.append(np.ones(N_RBF_molfeatures) * \
                             args.molfeatures_hyperparameters)
        else:
            sigma_RBF.append(args.molfeatures_hyperparameters)
        if args.addfeatures_hyperparameters.__class__ == float:
            sigma_RBF.append(np.ones(N_RBF_addfeatures) * \
                             args.addfeatures_hyperparameters)
        else:
            sigma_RBF.append(args.addfeatures_hyperparameters)
        params = {
            'N_MGK': N_MGK,
            'N_conv_MGK': N_conv_MGK,
            'graph_hyperparameters': graph_hyperparameters,
            'unique': False,
            'N_RBF': N_RBF,
            'sigma_RBF': np.concatenate(sigma_RBF),
        }
        return GraphKernelConfig(**params).kernel
    else:
        N_RBF = 0 if dataset.data[0].addfeatures is None \
            else dataset.data[0].addfeatures.shape[1]
        sigma_RBF = np.ones(N_RBF) * args.molfeatures_hyperparameters \
            if args.addfeatures_hyperparameters.__class__ == float \
            else args.addfeatures_hyperparameters
        params = {
            'f_kernel': os.path.join(args.save_dir, 'kernel.pkl'),
            'N_RBF': N_RBF,
            'sigma_RBF': sigma_RBF,
        }
        return PreCalcKernelConfig(**params).kernel
