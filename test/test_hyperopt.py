# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('%s/..' % CWD)
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
)
from chemml.args import HyperoptArgs
from run.HyperOpt import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
])
@pytest.mark.parametrize('num_splits', ['1', '2'])
@pytest.mark.parametrize('metric', ['r2', 'mae', 'rmse'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
])
@pytest.mark.parametrize('optimize_alpha', [True, False])
def test_hyperopt_PureGraph_regression(dataset, testset, num_splits, metric, graph_hyperparameters, optimize_alpha):
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    split, num_folds = testset
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/alpha' % save_dir)
    arguments = [
        '--save_dir', save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', '%s' % graph_hyperparameters,
        '--num_iters', '10',
        '--alpha', '0.01',
        '--num_splits', num_splits
    ]
    if optimize_alpha:
        arguments += [
            '--alpha_bounds', '0.008', '0.02'
        ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    if optimize_alpha:
        assert 0.008 < float(open('%s/alpha' % save_dir).readline()) < 0.02
        os.remove('%s/alpha' % save_dir)
    else:
        assert not os.path.exists('%s/alpha' % save_dir)
    os.remove('%s/hyperparameters_0.json' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('bace', ['smiles'], ['bace']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
@pytest.mark.parametrize('modelset', [
    ('gpr', True),
    ('gpr', False),
    ('gpc', False),
    ('svc', True),
    ('svc', False),
])
@pytest.mark.parametrize('testset', [
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_msnorm
])
def test_hyperopt_PureGraph_binary(dataset, modelset, testset, metric, graph_hyperparameters):
    task = 'binary'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    model, optimize_C = modelset
    split, num_folds = testset
    for i in range(len(pure_columns)):
        assert not os.path.exists('%s/hyperparameters_%d.json' % (save_dir, i))
    assert not os.path.exists('%s/alpha' % save_dir)
    assert not os.path.exists('%s/C' % save_dir)
    arguments = [
        '--save_dir', save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters'] + ['%s' % graph_hyperparameters] * len(pure_columns) + [
        '--num_iters', '10',
        '--C', '1',
    ]
    if model == 'gpr':
        arguments += [
            '--alpha', '0.01'
        ]
        if optimize_C:
            arguments += [
                '--alpha_bounds', '0.001', '0.02'
            ]
    elif model == 'svc':
        arguments += [
            '--C', '1',
        ]
        if optimize_C:
            arguments += [
                '--C_bounds', '0.01', '10.0'
            ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    if optimize_C and model == 'svc':
        assert 0.01 < float(open('%s/C' % save_dir).readline()) < 10.0
        os.remove('%s/C' % save_dir)
    else:
        assert not os.path.exists('%s/C' % save_dir)

    if optimize_C and model == 'gpr':
        assert 0.001 < float(open('%s/alpha' % save_dir).readline()) < 0.02
        os.remove('%s/alpha' % save_dir)
    else:
        assert not os.path.exists('%s/alpha' % save_dir)
    for i in range(len(pure_columns)):
        os.remove('%s/hyperparameters_%d.json' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX']),
])
@pytest.mark.parametrize('modelset', [
    ('gpc', False),
    ('svc', True),
    ('svc', False),
])
@pytest.mark.parametrize('testset', [
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_msnorm
])
def test_hyperopt_PureGraph_multiclass(dataset, modelset, testset, metric, graph_hyperparameters):
    task = 'binary'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    model, optimize_C = modelset
    split, num_folds = testset
    # TODO


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
])
@pytest.mark.parametrize('metric', ['rmse'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_pnorm
])
@pytest.mark.parametrize('features_kernel_type', ['rbf', 'dot_product'])
@pytest.mark.parametrize('features_hyperparameter_fix', [True, False])
def test_hyperopt_PureGraph_FeaturesAdd_regression(dataset, group_reading, features_scaling, testset, metric,
                                         graph_hyperparameters, features_kernel_type, features_hyperparameter_fix):
    optimize_alpha = features_hyperparameter_fix
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    split, num_folds = testset
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/alpha' % save_dir)
    assert not os.path.exists('%s/features_hyperparameters.json' % save_dir)
    arguments = [
        '--save_dir', save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', '%s' % graph_hyperparameters,
        '--num_iters', '10',
        '--alpha', '0.01',
        '--features_kernel_type', features_kernel_type,
    ]
    if features_hyperparameter_fix:
        arguments += [
            '--features_hyperparameters', '1.0',
        ]
    else:
        arguments += [
            '--features_hyperparameters', '1.0',
            '--features_hyperparameters_min', '0.1',
            '--features_hyperparameters_max', '20.0',
        ]
    if optimize_alpha:
        arguments += [
            '--alpha_bounds', '0.008', '0.02'
        ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    if optimize_alpha:
        assert 0.008 < float(open('%s/alpha' % save_dir).readline()) < 0.02
        os.remove('%s/alpha' % save_dir)
    else:
        assert not os.path.exists('%s/alpha' % save_dir)
    os.remove('%s/hyperparameters_0.json' % save_dir)
    os.remove('%s/features_hyperparameters.json' % save_dir)


def test_hyperopt_PureGraph_FeauturesAdd_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeauturesAdd_multiclass():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_regression():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesMol_multiclass():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_regression():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_binary():
    # TODO
    return


def test_hyperopt_PureGraph_FeaturesAddMol_multiclass():
    # TODO
    return


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_pnorm
])
@pytest.mark.parametrize('optimizer', ['L-BFGS-B', 'SLSQP'])
@pytest.mark.parametrize('loss', ['loocv', 'likelihood'])
def test_GradientOpt_PureGraph_regression(dataset, graph_hyperparameters, optimizer, loss):
    model = 'gpr'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--model_type', model,
        '--graph_hyperparameters', graph_hyperparameters,
        '--alpha', '0.01',
        '--optimizer', optimizer,
        '--loss', loss
    ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    os.remove('%s/hyperparameters_0.json' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_pnorm
])
@pytest.mark.parametrize('optimizer', ['L-BFGS-B', 'SLSQP'])
@pytest.mark.parametrize('loss', ['loocv', 'likelihood'])
@pytest.mark.parametrize('features_kernel_type', ['rbf', 'dot_product'])
@pytest.mark.parametrize('features_hyperparameter_fix', [True, False])
def test_GradientOpt_PureGraph_FeaturesAdd_regression(dataset, group_reading, features_scaling, graph_hyperparameters,
                                                      optimizer, loss, features_kernel_type,
                                                      features_hyperparameter_fix):
    model = 'gpr'
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/features_hyperparameters.json' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--model_type', model,
        '--graph_hyperparameters', graph_hyperparameters,
        '--alpha', '0.01',
        '--optimizer', optimizer,
        '--loss', loss,
        '--features_kernel_type', features_kernel_type,
    ]
    if features_hyperparameter_fix:
        arguments += [
            '--features_hyperparameters', '1.0',
        ]
    else:
        arguments += [
            '--features_hyperparameters', '1.0',
            '--features_hyperparameters_min', '0.1',
            '--features_hyperparameters_max', '20.0',
        ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    os.remove('%s/hyperparameters_0.json' % save_dir)
    os.remove('%s/features_hyperparameters.json' % save_dir)


def test_GradientOpt_PureGraph_FeaturesMol_regression():
    # TODO
    return


def test_GradientOpt_PureGraph_FeaturesAddMol_regression():
    # TODO
    return
