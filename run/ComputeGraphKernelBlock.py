#!/usr/bin/env python3
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute the graph kernel matrix of a block',
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
        help='Columns in input data. Only one column contain graphs can be '
             'assigned.\n'
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
    args = parser.parse_args()

    # set graph_property
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    assert (len(single_graph + multi_graph) == 1)
    graph_column = (single_graph + multi_graph)[0]
    # set kernel_config
    kernel_config = set_kernel_config(
        'graph', None, None,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    # set kernel_config
    df = get_df(args.input,
                os.path.join(args.result_dir, '%s.pkl' % ','.join(properties)),
                single_graph, multi_graph, reaction_graph)
    X, group_id = get_Xgroupid_from_df(df, single_graph, multi_graph)
    # set block config
    block_length, block_x_id, block_y_id = set_block_config(args.block_config)
    assert (block_x_id <= block_y_id)
    x0, x1 = block_x_id * block_length, (block_x_id + 1) * block_length
    y0, y1 = block_y_id * block_length, (block_y_id + 1) * block_length
    X, Y = X[x0:x1], X[y0:y1]
    # transform multi graph into unique and sorted single graph
    if multi_graph:
        X = ConvolutionKernel.get_graph(X.ravel())
        Y = ConvolutionKernel.get_graph(Y.ravel())
        kernel_config = set_kernel_config(
            'graph', None, None,
            ['trivial'], [], args.json_hyper,
            args.result_dir,
        )
    else:
        X = np.sort(np.unique(X))
        Y = np.sort(np.unique(Y))
    print('**\tComputing graph kernel matrix block\t**')
    K = kernel_config.kernel(X) if block_x_id == block_y_id else \
        kernel_config.kernel(X, Y)
    kernel_dict = {
        'graph_X': X,
        'graph_Y': Y,
        'K_graph': K,
        'theta': kernel_config.kernel.theta
    }
    print('**\tComputing finished\t**')
    # double check
    assert (False not in (X == np.sort(X)))
    assert (False not in (Y == np.sort(Y)))
    if block_x_id == block_y_id:
        assert (False not in (X == Y))
        assert (False not in (K - K.T < 1e-5))
    # output
    kernel_pkl = os.path.join(args.result_dir, 'graph_kernel_%s_%d_%d.pkl' %
                              (graph_column, block_x_id, block_y_id))
    pickle.dump(kernel_dict, open(kernel_pkl, 'wb'), protocol=4)


if __name__ == '__main__':
    main()
