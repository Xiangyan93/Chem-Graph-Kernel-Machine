#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute the kernel matrix of a block.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str,
        help='Columns in input data. Only one multi graph can be assigned.\n'
             'format: single_graph:multi_graph:targets\n'
             'examples: inchi::tt\n'
    )
    parser.add_argument(
        '--block_config', type=str, help='Block parameters\n'
                                         'format: block_length:x_id,y_id\n'
                                         'examples: 10000:0,0\n'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=1,
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()

    # set_graph_property
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    # set block config
    block_length, block_x_id, block_y_id = set_block_config(args.block_config)
    x0, x1 = block_x_id * block_length, (block_x_id + 1) * block_length
    y0, y1 = block_y_id * block_length, (block_y_id + 1) * block_length
    # get df
    df = get_df(args.input,
                os.path.join(args.result_dir,
                             '%s.pkl' % ','.join(properties)),
                single_graph, multi_graph, reaction_graph)
    n = 0
    K = []
    group_id_X = []
    group_id_Y = []
    theta = []
    for graph_column in single_graph:
        print('***\tReading graph kernel matrix block: %s\t***' % graph_column)
        X, group_id = get_Xgroupid_from_df(df, [graph_column], [])
        X, Y = X[x0:x1], X[y0:y1]
        dict = pickle.load(open(os.path.join(
            args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                             (graph_column, block_x_id, block_y_id)), 'rb'))
        idx_x = np.searchsorted(dict['graph_X'], X).ravel()
        idx_y = np.searchsorted(dict['graph_Y'], Y).ravel()
        K.append(dict['K_graph'][idx_x][:, idx_y])
        group_id_X.append(group_id[x0:x1])
        group_id_Y.append(group_id[y0:y1])
        theta.append(dict['theta'])
        n += 1
        print('***\tReading finished\t***')
    for graph_column in multi_graph:
        print('***\tReading graph kernel matrix block: %s\t***' % graph_column)
        X, group_id = get_Xgroupid_from_df(df, [], [graph_column])
        X, Y = X[x0:x1], X[y0:y1]
        if block_x_id != block_y_id:
            dict_xy = pickle.load(open(os.path.join(
                args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                                 (graph_column, block_x_id, block_y_id)), 'rb'))
            dict_xx = pickle.load(open(os.path.join(
                args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                                 (graph_column, block_x_id, block_x_id)), 'rb'))
            dict_yy = pickle.load(open(os.path.join(
                args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                                 (graph_column, block_y_id, block_y_id)), 'rb'))
            Kxx = dict_xx['K_graph']
            Kyy = dict_yy['K_graph']
            Kxy = dict_xy['K_graph']
            assert (False not in (dict_xx['graph_X'] == dict_xx['graph_Y']))
            assert (False not in (dict_yy['graph_X'] == dict_yy['graph_Y']))
            assert (False not in (dict_xy['graph_X'] == dict_xx['graph_X']))
            assert (False not in (dict_xy['graph_Y'] == dict_yy['graph_Y']))
            assert (False not in ((dict_xx['theta'] == dict_xy['theta']) &
                                  (dict_yy['theta'] == dict_xy['theta'])))
            graph = np.concatenate([dict_xy['graph_X'], dict_xy['graph_Y']])
            K_graph = np.bmat([[Kxx, Kxy], [Kxy.T, Kyy]]).A
            idx = np.argsort(graph)
            graph = graph[idx]
            K_graph = K_graph[idx][:, idx]
        else:
            dict_xy = pickle.load(open(os.path.join(
                args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                                 (graph_column, block_x_id, block_y_id)), 'rb'))
            assert (False not in (dict_xy['graph_X'] == dict_xy['graph_Y']))
            graph = dict_xy['graph_X']
            K_graph = dict_xy['K_graph']
        assert (False not in (K_graph - K_graph.T < 1e-5))
        K.append(ConvolutionKernel()(X, Y, n_process=args.n_jobs,
                                     graph=graph, K_graph=K_graph,
                                     theta=dict_xy['theta']))
        group_id_X.append(group_id[x0:x1])
        group_id_Y.append(group_id[y0:y1])
        theta.append(dict_xy['theta'])
        n += 1
        print('***\tReading finished\t***')

    for i in range(len(K)):
        assert (K[i].shape == K[0].shape)
        assert (False not in (group_id_X[i] == group_id_X[0]))
        assert (False not in (group_id_Y[i] == group_id_Y[0]))
    assert (K[0].shape == (group_id_X[0].shape[0], group_id_Y[0].shape[0]))
    kernel_dict = {
        'group_id_X': group_id_X[0],
        'group_id_Y': group_id_Y[0],
        'K': np.product(K, axis=0),
        'theta': theta[0]
    }
    kernel_pkl = os.path.join(args.result_dir, 'kernel_%d_%d.pkl' %
                              (block_x_id, block_y_id))
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
