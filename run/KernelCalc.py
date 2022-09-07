#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import pickle
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from chemml.args import KernelArgs


def main(args: KernelArgs) -> None:
    assert args.graph_kernel_type == 'graph'
    assert args.feature_columns is None
    assert args.n_jobs == 1
    # load data set.
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = 'graph'
    # set kernel_config
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      rbf_length_scale=args.features_hyperparameters,
                                      rbf_length_scale_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters)
    print('**\tCalculating kernel matrix\t**')
    kernel_dict = kernel_config.get_kernel_dict(dataset.X, dataset.X_repr.ravel())
    print('**\tEnd Calculating kernel matrix\t**')
    kernel_pkl = os.path.join(args.save_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main(args=KernelArgs().parse_args())
