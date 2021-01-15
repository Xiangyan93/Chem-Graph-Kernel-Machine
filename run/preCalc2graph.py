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
        help='The GaussianProcessRegressor, optimizer.\n'
             'The optimizer is useless here\n'
             'format: regressor:optimizer\n'
             'examples:\n'
             'graphdot:L-BFGS-B\n'
             'sklearn:fmin_l_bfgs_b\n'
             'graphdot_nystrom:None'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
        'format: single_graph:multi_graph:targets\n'
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
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '--consensus_config', type=str, default=None,
        help='Consensus model config.\n'
        'format: n_estimators:n_sample_per_model:n_jobs:consensus_rule\n'
        'examples: 100:2000:4:smallest_uncertainty\n'
        'examples: 100:2000:4:weight_uncertainty\n'
    )
    parser.add_argument(
        '--nystrom_config', type=str, default='0',
        help='Nystrom config.\n'
        'format: n_sample_core\n'
        'examples: 2000\n'
    )
    args = parser.parse_args()

    gpr, _ = set_gpr_optimizer(args.gpr)
    single_graph, multi_graph, _, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    # set kernel_config for graph
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
    # read input file
    df = get_df(None, os.path.join(args.result_dir, '%s.pkl' % ','.join(properties)),
                kernel_config.single_graph, kernel_config.multi_graph, [])
    X, y, id = get_XYid_from_df(
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
    _, y_, id_ = get_XYid_from_df(
        df,
        kernel_config_,
        properties=properties,
    )
    # change group_id to graph
    assert (np.array_equal(np.sort(id), id))
    assert (np.array_equal(id, id_))
    assert (np.array_equal(y, y_))
    def reset_model(m):
        idx = np.searchsorted(id, m.X_id_)
        m.X_train_ = X[idx, :]
        if m.__class__ == LRAGPR:
            idxc = np.searchsorted(id, m.C_id_)
            m._C = X[idxc, :]
    if model.__class__ == ConsensusRegressor:
        for m_ in model.models:
            reset_model(m_)
    else:
        reset_model(model)
    model.save(args.result_dir, overwrite=True)


if __name__ == '__main__':
    main()
