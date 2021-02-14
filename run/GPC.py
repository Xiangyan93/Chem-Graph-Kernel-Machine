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
        description='Gaussian process classification using graph kernel',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--gpc', type=str, default="sklearn",
        help='The GaussianProcessClassifier, optimizer.\n'
             'format: classifier:optimizer\n'
             'examples:\n'
             'sklearn:fmin_l_bfgs_b\n'
             'sklearn:None'
    )
    parser.add_argument(
        '--kernel', type=str,
        help='format: kernel.\n'
             'examples:\n'
             'graph\n'
             'preCalc\n'
             'For preCalc kernel, run KernelCalc.py first.'
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:reaction_graph:targets\n'
        'examples: SMILES:::toxicity\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional vector features with RBF kernel.\n' 
             'examples:\n'
             'Tred:0.1\n'
             'T,P:100,500'
    )
    parser.add_argument(
        '--train_test_config', type=str, help=
        'format: mode:train_size:train_ratio:seed\n'
        'examples:\n'
        'loocv:::0\n'
        'train_test:1000::0\n'
        'train_test::0.8:0\n'
        # 'dynamic::0.8:0:500'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=cpu_count(),
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()

    # set args
    gpc, optimizer = set_gpc_optimizer(args.gpc)
    kernel = args.kernel
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    mode, train_size, train_ratio, seed, _ = \
        set_mode_train_size_ratio_seed(args.train_test_config)

    # set kernel_config
    kernel_config = set_kernel_config(
        kernel, add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )

    # set Gaussian process classifier
    model = set_gpc_model(gpc, kernel_config, optimizer, n_jobs=args.n_jobs)

    # read input
    params = {
        'train_size': train_size,
        'train_ratio': train_ratio,
        'seed': seed,
        'byclass': True
    }
    if mode == 'loocv' or mode == 'lomocv':
        params['train_size'] = None
        params['train_ratio'] = 1.0
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        args.result_dir, args.input, kernel_config, properties, params
    )
    # gpr
    data = {
        'df': df,
        'df_train': df_train,
        'train_X': train_X,
        'train_Y': train_Y,
        'train_id': train_id,
        'test_X': test_X,
        'test_Y': test_Y,
        'test_id': test_id
    }
    gpr_params = {
        'mode': mode,
        'model': model,
    }
    gpc_run(data, args.result_dir, kernel_config, gpr_params, tag=seed)


if __name__ == '__main__':
    main()
