#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import pickle
import numpy as np
from chemml.args import KernelBlockArgs


def main(args: KernelBlockArgs) -> None:
    K = np.zeros((args.block_id[0], args.block_id[1]), dtype=object)
    X = []
    theta = []
    for i in range(args.block_id[0]):
        for j in range(args.block_id[1]):
            if i <= j:
                kernel_pkl = os.path.join(args.save_dir, 'KernelBlock',
                                          'kernel_%d_%d.pkl' % (i, j))
                kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
                K[i][j] = kernel_dict['K']
                if i == 0:
                    X.append(kernel_dict['Y'])
            else:
                kernel_pkl = os.path.join(args.save_dir, 'KernelBlock',
                                          'kernel_%d_%d.pkl' % (j, i))
                kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
                K[i][j] = kernel_dict['K'].T
            theta.append(kernel_dict['theta'])
    X = np.concatenate(X).ravel()
    K = np.bmat(K.tolist()).A
    assert (K.shape == (X.shape[0], X.shape[0]))
    for i, theta_ in enumerate(theta):
        assert (False not in (theta_ == theta[0]))
    kernel_dict = {
        'X': X,
        'K': K,
        'theta': theta[0]
    }
    kernel_pkl = os.path.join(args.save_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main(args=KernelBlockArgs().parse_args())
