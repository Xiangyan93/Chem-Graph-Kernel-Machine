#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.args import KernelArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config


def main(args: KernelArgs) -> None:
    assert args.kernel_type == 'graph'
    assert args.feature_columns is None
    assert args.n_jobs == 1
    dataset = Dataset.load(args.save_dir)
    dataset.update_args(args)
    dataset.normalize_features()
    assert dataset.kernel_type == 'graph'
    X = dataset.X
    # set kernel_config
    kernel = get_kernel_config(args, dataset).kernel
    print('**\tCalculating kernel matrix\t**')
    K = kernel(X)
    # print(dataset)
    kernel_dict = {
        'group_id': dataset.X_gid.ravel(),
        'K': K,
        'theta': kernel.theta
    }
    print('**\tEnd Calculating kernel matrix\t**')
    kernel_pkl = os.path.join(args.save_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main(args=KernelArgs().parse_args())
