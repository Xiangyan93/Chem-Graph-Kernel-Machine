#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *


def main():
    parser = argparse.ArgumentParser(
        description='Concatenate the kernel matrix from blocks.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--block_config', type=str, help='Block parameters\n'
                                         'format: block_length:max(x_id),'
                                         'max(y_id)\n'
                                         'example: 10000:2,2\n'
    )
    args = parser.parse_args()

    block_length, block_x_id, block_y_id = set_block_config(args.block_config)
    assert (block_x_id == block_y_id)
    K = np.zeros((block_x_id, block_y_id), dtype=object)
    group_id = []
    theta = []
    for i in range(block_x_id):
        for j in range(block_y_id):
            if i <= j:
                kernel_pkl = os.path.join(args.result_dir, 'kernel_%d_%d.pkl' %
                                          (i, j))
                kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
                K[i][j] = kernel_dict['K']
                if i == 0:
                    group_id.append(kernel_dict['group_id_Y'])
            else:
                kernel_pkl = os.path.join(args.result_dir, 'kernel_%d_%d.pkl' %
                                          (j, i))
                kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
                K[i][j] = kernel_dict['K'].T
            theta.append(kernel_dict['theta'])
    group_id = np.concatenate(group_id)
    K = np.bmat(K.tolist()).A
    assert (K.shape == (group_id.shape[0], group_id.shape[0]))
    for i, theta_ in enumerate(theta):
        assert (False not in (theta_ == theta[0]))
    kernel_dict = {
        'group_id': group_id,
        'K': K,
        'theta': theta[0]
    }
    kernel_pkl = os.path.join(args.result_dir, 'kernel.pkl')
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
