#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.kernels.GraphKernel import *
from run.GPR import *


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Kernel Matrix and Gradients.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:targets\n'
        'examples: inchi::tt\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional vector features with RBF kernel.\n' 
             'examples:\n'
             'red_T:0.1\n'
             'T,P:100,500'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    args = parser.parse_args()

    single_graph, multi_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    # set kernel_config
    kernel_config = set_kernel_config(
        args.result_dir, 'graph', args.normalized,
        single_graph, multi_graph,
        add_f, add_p,
        json.loads(open(args.json_hyper, 'r').readline())
    )
    params = {
        'train_size': None,
        'train_ratio': 1.0,
        'seed': 0,
    }
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        args.result_dir, args.input, kernel_config, properties, params
    )
    print('**\tCalculating kernel matrix\t**')
    kernel_config.kernel.PreCalculate(train_X, result_dir=args.result_dir)
    print('**\tEnd Calculating kernel matrix\t**')


if __name__ == '__main__':
    main()
