#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor, optimizer.\n'
             'format: regressor:optimizer\n'
             'examples:\n'
             'graphdot:L-BFGS-B\n'
             'sklearn:fmin_l_bfgs_b\n'
             'graphdot_nystrom:None'
    )
    parser.add_argument(
        '--kernel', type=str,
        help='format: kernel:alpha.\n'
             'examples:\n'
             'graph:0.01\n'
             'graph:10.0\n'
             'preCalc:0.01\n'
             'For preCalc kernel, run KernelCalc.py first.'
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:reaction_graph:targets\n'
        'examples: inchi:::tc\n'
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
        'format: mode:train_size:train_ratio:seed:(dynamic train size)\n'
        'examples:\n'
        'loocv:::0\n'
        'train_test:1000::0\n'
        'train_test::0.8:0\n'
        'dynamic::0.8:0:500'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='read existed model file',
    )
    parser.add_argument(
        '--consensus_config', type=str, default=None,
        help='Need to be set if consensus model is used.\n'
        'format: n_estimators:n_sample_per_model:n_jobs:consensus_rule\n'
        'examples: 100:2000:4:smallest_uncertainty\n'
        'examples: 100:2000:4:weight_uncertainty\n'
    )
    parser.add_argument(
        '--nystrom_config', type=str, default='0',
        help='Need to be set if Nystrom approximation is used.\n'
        'format: n_sample_core\n'
        'examples: 2000\n'
    )
    args = parser.parse_args()

    # set args
    gpr, optimizer = set_gpr_optimizer(args.gpr)
    kernel, alpha = set_kernel_alpha(args.kernel)
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    mode, train_size, train_ratio, seed, dynamic_train_size = \
        set_mode_train_size_ratio_seed(args.train_test_config)
    # set kernel_config
    kernel_config = set_kernel_config(
        kernel, add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )

    # set Gaussian process regressor
    model = set_gpr_model(gpr, kernel_config, optimizer, alpha)

    # read input
    params = {
        'train_size': train_size,
        'train_ratio': train_ratio,
        'seed': seed,
        'byclass': False
    }
    if mode == 'loocv' or mode == 'lomocv' or mode == 'all':
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
        'consensus_config': args.consensus_config,
        'nystrom_config': args.nystrom_config,
        'dynamic_train_size': dynamic_train_size
    }
    gpr_run(data, args.result_dir, kernel_config, gpr_params,
            load_model=args.load_model, tag=seed)


if __name__ == '__main__':
    tqdm.pandas()
    main()
