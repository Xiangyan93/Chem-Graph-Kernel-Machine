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
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor, optimizer.\n'
             'The optimizer is useless here\n'
             'format: regressor:optimizer\n'
             'examples:\n'
             'graphdot:L-BFGS-B\n'
             'sklearn:fmin_l_bfgs_b\n'
             'graphdot_nystrom:None'
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:reaction_graph:targets\n'
        'examples: inchi:::tt\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional vector features with RBF kernel.\n' 
             'examples:\n'
             'Tred:0.1\n'
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
    parser.add_argument(
        '-n', '--ntasks', type=int, default=cpu_count(),
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()

    gpr, _ = set_gpr_optimizer(args.gpr)
    single_graph, multi_graph, _, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    # set kernel_config
    kernel_config = set_kernel_config(
        'graph', add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )
    # set Gaussian process regressor
    model = set_gpr_model(gpr, kernel_config, None, 0.01)
    # set consensus model
    consensus, n_estimators, n_sample_per_model, n_jobs, consensus_rule = \
        set_consensus_config(args.consensus_config)
    if consensus:
        model = ConsensusRegressor(
            model,
            n_estimators=n_estimators,
            n_sample_per_model=n_sample_per_model,
            n_jobs=n_jobs,
            consensus_rule=consensus_rule
        )
    # read model
    model.load(args.result_dir)
    # read input
    df = get_df(args.input, None, kernel_config.single_graph,
                kernel_config.multi_graph, [], n_jobs=args.ntasks)
    X, _, _ = get_XYid_from_df(df, kernel_config, properties=None)
    for i in range(len(kernel_config.single_graph)+len(kernel_config.multi_graph)):
        unify_datatype(X[:, i].ravel(), model.X_train_[:, i].ravel())
    y, y_std = model.predict(X, return_std=True)
    df = pd.read_csv(args.input, sep='\s+')
    df['predict'] = y
    df['uncertainty'] = y_std
    df.to_csv(os.path.join(args.result_dir, 'predict.log'), sep=' ', index=False)


if __name__ == '__main__':
    main()
