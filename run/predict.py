#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import *


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
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
    parser.add_argument(
        '--f_model', type=str,
        help='model.pkl',
    )
    args = parser.parse_args()

    # set Gaussian process regressor
    GPR = set_gpr(args.gpr)

    single_graph, multi_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)

    # set kernel_config
    kernel_config = set_kernel_config(
        '', 'graph', args.normalized,
        single_graph, multi_graph,
        add_f, add_p,
        json.loads(open(args.json_hyper, 'r').readline())
    )
    model = GPR.load_cls(args.f_model, kernel_config.kernel)
    # read input
    df = get_df(args.input, None, kernel_config.single_graph, kernel_config.multi_graph)
    X, _, _ = get_XYid_from_df(df, kernel_config)
    y, y_std = model.predict(X, return_std=True)
    df = pd.read_csv(args.input, sep='\s+')
    df['predict'] = y
    df['uncertainty'] = y_std
    df.to_csv('predict.csv', sep=' ', index=False)


if __name__ == '__main__':
    main()
