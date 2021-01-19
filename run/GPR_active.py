#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.tools import *
from chemml.regression.active_learner import ActiveLearner


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel, active '
                    'learning',
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
             'graphdot:None'
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
        'examples: inchi:::tt\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional vector features with RBF kernel.\n' 
             'examples:\n'
             'red_T:0.1\n'
             'T,P:100,500'
    )
    parser.add_argument(
        '--train_test_config', type=str, help=
        'format: mode:train_size:train_ratio:seed\n'
        'examples:\n'
        'train_test:1000::0\n'
        'train_test::0.8:0\n'
    )
    parser.add_argument(
        '--active_config', type=str, help=
        'format: learning_mode:add_mode:init_size:add_size:max_size:search_size'
        ':pool_size:stride\n'
        'examples:\n'
        'supervised:nlargest:5:1:200:0:200:100\n'
        'unsupervised:cluster:5:5:200:0:200:100\n'
        'random:nlargest:100:100:200:0:200:100\n'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '--continued', action='store_true',
        help='whether continue training'
    )
    args = parser.parse_args()

    # set args
    kernel, alpha = set_kernel_alpha(args.kernel)
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    learning_mode, add_mode, init_size, add_size, max_size, search_size, \
    pool_size, stride = set_active_config(args.active_config)
    # set kernel_config
    kernel_config = set_kernel_config(
        kernel, add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )

    if args.continued:
        print('***\tLoading checkpoint\t***\n')
        f_checkpoint = os.path.join(args.result_dir, 'checkpoint.pkl')
        activelearner = ActiveLearner.load_checkpoint(f_checkpoint,
                                                      kernel_config)
        activelearner.max_size = max_size
        print("model continued from checkpoint")
    else:
        # set optimizer
        gpr, optimizer = set_gpr_optimizer(args.gpr)
        # set Gaussian process regressor
        model = set_gpr_model(gpr, kernel_config, optimizer, alpha)
        # set train_test
        mode, train_size, train_ratio, seed, dynamic_train_size = \
            set_mode_train_size_ratio_seed(args.train_test_config)
        # read input
        params = {
            'train_size': train_size,
            'train_ratio': train_ratio,
            'seed': seed,
        }
        df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
        test_id = read_input(
            args.result_dir, args.input, kernel_config, properties, params
        )
        activelearner = ActiveLearner(
            train_X, train_Y, train_id, alpha, kernel_config, learning_mode,
            add_mode, init_size, add_size, max_size, search_size, pool_size,
            args.result_dir, GPRLearner, model,
            test_X=test_X, test_Y=test_Y, test_id=test_id,
            optimizer=optimizer, seed=seed, stride=stride
        )

    while True:
        print('***\tStart: active learning, current size = %i\t***\n' %
              activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train():
            if activelearner.current_size % activelearner.stride == 0:
                print('\n**\tstart evaluate\t**\n')
                activelearner.evaluate()
                activelearner.write_training_plot()
                activelearner.save_checkpoint()
            else:
                activelearner.y_pred = None
                activelearner.y_std = None
        else:
            print('Training failed for all alpha')
        if activelearner.stop_sign():
            break
        print('**\tstart add samples**\n')
        activelearner.add_samples()

    print('\n***\tEnd: active learning\t***\n')


if __name__ == '__main__':
    main()
