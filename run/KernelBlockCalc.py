#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import pickle
from chemml.args import KernelBlockArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config


def main(args: KernelBlockArgs) -> None:
    assert args.graph_kernel_type == 'graph'
    assert args.feature_columns is None
    assert args.n_jobs == 1
    # load data set.
    dataset = Dataset.load(path=args.save_dir, args=args)
    assert dataset.graph_kernel_type == 'graph'
    # set kernel_config
    kernel = get_kernel_config(args, dataset).kernel
    print('**\tCalculating kernel matrix\t**')
    kernel_dict = {
        'X': dataset.X_repr[args.X_idx],
        'Y': dataset.X_repr[args.Y_idx],
        'K': kernel(dataset.X[args.X_idx], dataset.X[args.Y_idx]),
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
