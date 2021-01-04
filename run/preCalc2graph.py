#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import *


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='transform PreCalcKernel model.pkl to GraphKernel model.pkl'
                    ' for prediction',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
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

    # set Gaussian process regressor
    GPR = set_gpr(args.gpr)

    single_graph, multi_graph, _, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    # set kernel_config for graph
    kernel_config = set_kernel_config(
        'graph', add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    # read model
    f_model = os.path.join(args.result_dir, 'model.pkl')
    model = GPR.load_cls(f_model, kernel_config.kernel)
    # read input file
    df = get_df(None, os.path.join(args.result_dir, '%s.pkl' % ','.join(properties)),
                kernel_config.single_graph, kernel_config.multi_graph, [])
    X, Y, id = get_XYid_from_df(
        df,
        kernel_config,
        properties=properties,
    )
    # set kernel_config for preCalc
    kernel_config_ = set_kernel_config(
        'preCalc', add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    X_, Y, id = get_XYid_from_df(
        df,
        kernel_config_,
        properties=properties,
    )
    # change group_id to graph
    idx = [X_.tolist().index(x.tolist()) for x in model.X_train_]
    model.X_train_ = X[idx, :]
    model.save(args.result_dir, overwrite=True)


if __name__ == '__main__':
    main()
