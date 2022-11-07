#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import pickle
from chemml.args import KernelBlockArgs
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config


def main(args: KernelBlockArgs) -> None:
    assert args.graph_kernel_type == 'graph'
    assert args.feature_columns is None
    assert args.n_jobs == 1
    # load data set.
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = 'graph'
    # set kernel_config
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      features_kernel_type=args.features_kernel_type,
                                      features_hyperparameters=args.features_hyperparameters,
                                      features_hyperparameters_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters)
    kernel = kernel_config.kernel
    print('**\tCalculating kernel matrix\t**')
    X = dataset.X_repr[args.X_idx[0]:args.X_idx[1]]
    if args.block_id[0] == args.block_id[1]:
        Y = X
        K = kernel(dataset.X[args.X_idx[0]:args.X_idx[1]])
    else:
        Y = dataset.X_repr[args.Y_idx[0]:args.Y_idx[1]]
        K = kernel(dataset.X[args.X_idx[0]:args.X_idx[1]], dataset.X[args.Y_idx[0]:args.Y_idx[1]])
    kernel_dict = {
        'X': X,
        'Y': Y,
        'K': K,
        'theta': kernel.theta
    }
    print('**\tEnd Calculating kernel matrix\t**')
    if not os.path.exists(os.path.join(args.save_dir, 'KernelBlock')):
        os.mkdir(os.path.join(args.save_dir, 'KernelBlock'))
    kernel_pkl = os.path.join(args.save_dir, 'KernelBlock',
                              'kernel_%d_%d.pkl' % (args.block_id[0], args.block_id[1]))
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main(args=KernelBlockArgs().parse_args())
