#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import json
from chemml.kernels.KernelConfig import get_XYid_from_df
from run.txt2pkl import *


def set_gpr_optimizer(gpr):
    gpr, optimizer = gpr.split(':')
    if gpr not in ['graphdot', 'sklearn']:
        raise Exception('Unknown gpr')
    if optimizer in ['None', 'none', '']:
        return gpr, None
    if gpr == 'graphdot' and optimizer != 'L-BFGS-B':
        raise Exception('Please use L-BFGS-B optimizer')
    return gpr, optimizer


def set_kernel_alpha(kernel):
    kernel, alpha = kernel.split(':')
    if kernel not in ['graph', 'preCalc']:
        raise Exception('Unknown kernel')
    return kernel, float(alpha)


def set_add_feature_hyperparameters(add_features):
    if add_features is None:
        return None, None
    add_f, add_p = add_features.split(':')
    add_f = add_f.split(',')
    add_p = list(map(float, add_p.split(',')))
    assert(len(add_f) == len(add_p))
    return add_f, add_p

def set_mode_train_size_ratio_seed(train_test_config):
    mode, train_size, train_ratio, seed = train_test_config.split(':')
    train_size = int(train_size) if train_size else None
    train_ratio = float(train_ratio) if train_ratio else None
    seed = int(seed) if seed else 0
    return mode, train_size, train_ratio, seed


def set_learner(gpr):
    if gpr == 'graphdot':
        from chemml.GPRgraphdot.learner import Learner
    elif gpr == 'sklearn':
        from chemml.GPRsklearn.learner import Learner
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % gpr)
    return Learner


def set_gpr(gpr):
    if gpr == 'graphdot':
        from chemml.GPRgraphdot.gpr import GPR as GaussianProcessRegressor
    elif gpr == 'sklearn':
        from chemml.GPRsklearn.gpr import RobustFitGaussianProcessRegressor as \
            GaussianProcessRegressor
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % gpr)
    return GaussianProcessRegressor


def set_kernel_config(kernel, add_features, add_hyperparameters,
                      single_graph, multi_graph, hyperjson,
                      result_dir):
    if kernel == 'graph':
        hyperdict = [
            json.loads(open(f, 'r').readline()) for f in hyperjson.split(',')
        ]
        params = {
            'single_graph': single_graph,
            'multi_graph': multi_graph,
            'hyperdict': hyperdict
        }
        from chemml.kernels.GraphKernel import GraphKernelConfig as KConfig
    else:
        params = {
            'result_dir': result_dir,
        }
        from chemml.kernels.PreCalcKernel import PreCalcKernelConfig as KConfig
    return KConfig(add_features, add_hyperparameters, params)


def read_input(result_dir, input, kernel_config, properties, params):
    def df_filter(df, train_size=None, train_ratio=None, bygroup=False, seed=0):
        np.random.seed(seed)
        if bygroup:
            gname = 'group_id'
        else:
            gname = 'id'
        unique_ids = df[gname].unique()
        if train_size is None:
            train_size = int(unique_ids.size * train_ratio)
        ids = np.random.choice(unique_ids, train_size, replace=False)
        df_train = df[df[gname].isin(ids)]
        df_test = df[~df[gname].isin(ids)]
        return df_train, df_test

    if params is None:
        params = {
            'train_size': None,
            'train_ratio': 1.0,
            'seed': 0,
        }
    print('***\tStart: Reading input.\t***')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # read input.
    single_graph = kernel_config.single_graph \
        if hasattr(kernel_config, 'single_graph') else []
    multi_graph = kernel_config.multi_graph \
        if hasattr(kernel_config, 'multi_graph') else []
    df = get_df(input,
                os.path.join(result_dir, '%s.pkl' % ','.join(properties)),
                single_graph, multi_graph, [])
    # get df of train and test sets
    df_train, df_test = df_filter(
        df,
        train_size=params['train_size'],
        train_ratio=params['train_ratio'],
        seed=params['seed'],
        bygroup=kernel_config.add_features is not None
    )
    # get X, Y of train and test sets
    train_X, train_Y, train_id = get_XYid_from_df(
        df_train,
        kernel_config,
        properties=properties,
    )
    test_X, test_Y, test_id = get_XYid_from_df(
        df_test,
        kernel_config,
        properties=properties,
    )
    if test_X is None:
        test_X = train_X
        test_Y = np.copy(train_Y)
        test_id = train_id
    print('***\tEnd: Reading input.\t***\n')
    return (df, df_train, df_test, train_X, train_Y, train_id, test_X,
            test_Y, test_id)

'''
def pre_calculate(kernel_config, df, result_dir, load_K):
    if kernel_config.type == 'graph':
        print('***\tStart: Graph kernels calculating\t***')
        print('**\tCalculating kernel matrix\t**')
        kernel = kernel_config.kernel
        if load_K:
            kernel.load(result_dir)
        else:
            X, _, _ = get_XYid_from_df(df, kernel_config)
            kernel.PreCalculate(X, result_dir=result_dir)
        print('***\tEnd: Graph kernels calculating\t***\n')
'''

def gpr_run(data, result_dir, kernel_config, params,
            load_model=False):
    df = data['df']
    df_train = data['df_train']
    train_X = data['train_X']
    train_Y = data['train_Y']
    train_id = data['train_id']
    test_X = data['test_X']
    test_Y = data['test_Y']
    test_id = data['test_id']
    optimizer = params['optimizer']
    mode = params['mode']
    alpha = params['alpha']
    Learner = params['Learner']

    # pre-calculate graph kernel matrix.
    '''
    if params['optimizer'] is None:
        pre_calculate(kernel_config, df, result_dir, load_K)
    '''

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        learner = Learner(train_X, train_Y, train_id, test_X, test_Y,
                          test_id, kernel_config, alpha=alpha,
                          optimizer=optimizer)
        if load_model:
            print('loading existed model')
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir)
            learner.kernel_config.save(result_dir, learner.model)
        r2, ex_var, mse, mae, out = learner.evaluate_loocv()
        print('LOOCV:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        print('mae: %.5f' % mae)
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    else:
        learner = Learner(train_X, train_Y, train_id, test_X, test_Y,
                          test_id, kernel_config, alpha=alpha,
                          optimizer=optimizer)
        learner.train()
        learner.model.save(result_dir)
        learner.kernel_config.save(result_dir, learner.model)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        r2, ex_var, mse, mae, out = learner.evaluate_train()
        print('Training set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        print('mae: %.5f' % mae)
        out.to_csv('%s/train.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
        r2, ex_var, mse, mae, out = learner.evaluate_test()
        print('Test set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        print('mae: %.5f' % mae)
        out.to_csv('%s/test.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')


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
        'loocv:::0\n'
        'train_test:1000::0\n'
        'train_test::0.8:0\n'
    )
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='read existed model.pkl',
    )
    args = parser.parse_args()

    # set args
    gpr, optimizer = set_gpr_optimizer(args.gpr)
    kernel, alpha = set_kernel_alpha(args.kernel)
    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)
    add_f, add_p = set_add_feature_hyperparameters(args.add_features)
    mode, train_size, train_ratio, seed = \
        set_mode_train_size_ratio_seed(args.train_test_config)

    # set Gaussian process regressor
    Learner = set_learner(gpr)

    # set kernel_config
    kernel_config = set_kernel_config(
        kernel, add_f, add_p,
        single_graph, multi_graph, args.json_hyper,
        args.result_dir,
    )

    # read input
    params = {
        'train_size': train_size,
        'train_ratio': train_ratio,
        'seed': seed,
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
        'optimizer': optimizer,
        'alpha': alpha,
        'Learner': Learner
    }
    gpr_run(data, args.result_dir, kernel_config, gpr_params,
            load_model=args.load_model)


if __name__ == '__main__':
    main()
