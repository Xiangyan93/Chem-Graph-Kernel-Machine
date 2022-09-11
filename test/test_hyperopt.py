# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('%s/..' % CWD)
import json
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
)
from chemml.args import HyperoptArgs
from run.HyperOpt import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'], None),
])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['r2', 'mae', 'rmse'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
])
@pytest.mark.parametrize('optimize_alpha', [True, False])
def test_hyperopt_pure_graph_regression(dataset, testset, metric, graph_hyperparameters, optimize_alpha):
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns, features_columns = dataset
    split, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
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

"""
@pytest.mark.parametrize('dataset', [
    ('bace', ['smiles'], ['bace'], None),
])
@pytest.mark.parametrize('testset', [
    ('random', '10'),
])
@pytest.mark.parametrize('model', ['gpc', 'svc'])
@pytest.mark.parametrize('metric', ['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_pnorm
])
@pytest.mark.parametrize('optimize_C', [True, False])
def test_hyperopt_pure_graph_binary(dataset, testset, model, metric, graph_hyperparameters, optimize_C):
    task = 'binary'
    dataset, pure_columns, target_columns, features_columns = dataset
    split, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/C' % save_dir)
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
        '--C', '1',
    ]
    if optimize_C:
        arguments += [
            '--C_bounds', '0.01', '10.0'
        ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    if optimize_C:
        assert 0.01 < float(open('%s/C' % save_dir).readline()) < 10.0
        os.remove('%s/C' % save_dir)
    else:
        assert not os.path.exists('%s/C' % save_dir)
    os.remove('%s/hyperparameters_0.json' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['r2', 'mae', 'rmse'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
])
@pytest.mark.parametrize('optimize_alpha', [True, False])
def test_hyperopt_pure_graph_regression(dataset, testset, metric, graph_hyperparameters, optimize_alpha):
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns, features_columns = dataset
    split, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
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
"""

"""
@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['rmse'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_msnorm
])
@pytest.mark.parametrize('features_kernel_type', ['rbf', 'dot_product'])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('features_hyperparameter_fix', [True, False])
def test_hyperopt_pure_graph_regression(dataset, testset, metric, graph_hyperparameters,
                                        features_kernel_type, features_scaling, features_hyperparameter_fix):
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns, features_columns = dataset
    split, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
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
    if features_columns is not None:
        arguments += [
            '--feature_columns'
        ] + features_columns
    if features_hyperparameter_fix:
        arguments += [
            '--features_hyperparameters', '1.0',
        ]
    else:
        arguments += [
            '--features_hyperparameters', '1.0',
            '--features_hyperparameters_min', '0.1',
            '--features_hyperparameters_max', '20.0',
            '--alpha_bounds', '0.008', '0.02'
        ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    if not features_hyperparameter_fix:
        assert 0.008 < float(open('%s/alpha' % save_dir).readline()) < 0.02
        os.remove('%s/alpha' % save_dir)
    else:
        assert not os.path.exists('%s/alpha' % save_dir)
    os.remove('%s/hyperparameters_0.json' % save_dir)
"""

"""
@pytest.mark.parametrize('testset', [
    ('freesolv', 'regression', 'gpr', 'loocv', 'rmse', '1'),
    ('bace', 'binary', 'gpc', 'random', 'roc-auc', '10'),
    ('bace', 'binary', 'svc', 'random', 'roc-auc', '10')
])
def test_hyperopt_pure_graph_classification(testset):


@pytest.mark.parametrize('testset', [
    ('freesolv', 'regression', 'gpr', 'loocv', 'rmse', '1'),
])
@pytest.mark.parametrize('optimizer', ['L-BFGS-B', 'SLSQP'])
@pytest.mark.parametrize('loss', ['loocv', 'likelihood'])
def test_GradientOpt_pure_graph(testset, optimizer, loss):
    dataset, task, model, split, metric, num_folds = testset
    save_dir = '%s/data/_%s' % (CWD, dataset)
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/alpha' % save_dir)
    assert not os.path.exists('%s/C' % save_dir)
    assert not os.path.exists('%s/features_hyperparameters.json' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', '%s/../hyperparameters/additive-PNorm.json' % CWD,
        '--alpha', '0.01',
        '--optimizer', optimizer,
        '--loss', loss
    ]
    args = HyperoptArgs().parse_args(arguments)
    main(args)
    os.remove('%s/hyperparameters_0.json' % save_dir)


@pytest.mark.parametrize('testset', [
    ('freesolv', 'regression', 'gpr', 'loocv', 'rmse', '1'),
    ('bace', 'binary', 'gpc', 'random', 'roc-auc', '10'),
    ('bace', 'binary', 'svc', 'random', 'roc-auc', '10')
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d'],
                                                ['morgan_count'],
                                                ['rdkit_2d_normalized', 'morgan']])
@pytest.mark.parametrize('features_kernel_type', ['rbf', 'dot_product'])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('features_hyperparameter_fix', [True, False])
def test_hyperopt_pure_graph_features(testset, features_generator, features_kernel_type,
                                      features_scaling, features_hyperparameter_fix):
    dataset, task, model, split, metric, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(features_generator), features_scaling)
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/alpha' % save_dir)
    assert not os.path.exists('%s/C' % save_dir)
    assert not os.path.exists('%s/features_hyperparameters.json' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', '%s/../hyperparameters/additive-PNorm.json' % CWD,
        '--num_iters', '10',
        '--features_kernel_type', features_kernel_type,
        '--features_generator',
    ] + features_generator
    if task == 'regression':
        if features_hyperparameter_fix:
            arguments += [
                '--alpha', '0.01',
            ]
        else:
            arguments += [
                '--alpha', '0.01',
                '--alpha_bounds', '0.008', '0.02'
            ]
    if model == 'svc':
        if features_hyperparameter_fix:
            arguments += [
                '--C', '1',
            ]
        else:
            arguments += [
                '--C', '1',
                '--C_bounds', '0.01', '10.0'
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
    f = json.load(open('%s/features_hyperparameters.json' % save_dir))
    if not features_hyperparameter_fix:
        if task == 'regression':
            assert 0.008 < float(open('%s/alpha' % save_dir).readline()) < 0.02
            os.remove('%s/alpha' % save_dir)
        if model == 'svc':
            assert 0.01 < float(open('%s/C' % save_dir).readline()) < 10.0
            os.remove('%s/C' % save_dir)
        assert f['features_hyperparameters'][0] != pytest.approx(1.0, 1e-10)
        assert f['features_hyperparameters_bounds'][0] == [0.1, 20.0]
    else:
        assert f['features_hyperparameters'][0] == pytest.approx(1.0, 1e-10)
        assert f['features_hyperparameters_bounds'] == "fixed"
    os.remove('%s/hyperparameters_0.json' % save_dir)
    os.remove('%s/features_hyperparameters.json' % save_dir)


@pytest.mark.parametrize('testset', [
    ('freesolv', 'regression', 'gpr', 'loocv', 'rmse', '1'),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d'],
                                                ['morgan_count'],
                                                ['rdkit_2d_normalized', 'morgan']])
@pytest.mark.parametrize('features_kernel_type', ['dot_product'])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('features_hyperparameter_fix', [True, False])
@pytest.mark.parametrize('optimizer', ['L-BFGS-B', 'SLSQP'])
@pytest.mark.parametrize('loss', ['likelihood'])
def test_GradientOpt_pure_graph_features(testset, features_generator, features_kernel_type,
                                         features_scaling, features_hyperparameter_fix, optimizer, loss):
    dataset, task, model, split, metric, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(features_generator), features_scaling)
    assert not os.path.exists('%s/hyperparameters_0.json' % save_dir)
    assert not os.path.exists('%s/alpha' % save_dir)
    assert not os.path.exists('%s/C' % save_dir)
    assert not os.path.exists('%s/features_hyperparameters.json' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', '%s/../hyperparameters/additive-PNorm.json' % CWD,
        '--alpha', '0.01',
        '--optimizer', optimizer,
        '--features_kernel_type', features_kernel_type,
        '--loss', loss,
        '--features_generator',
    ] + features_generator
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
    f = json.load(open('%s/features_hyperparameters.json' % save_dir))
    if not features_hyperparameter_fix:
        assert f['features_hyperparameters'][0] == pytest.approx(1.0, 1e-10)
        assert f['features_hyperparameters_bounds'][0] == [0.1, 20.0]
    else:
        assert f['features_hyperparameters'][0] == pytest.approx(1.0, 1e-10)
        assert f['features_hyperparameters_bounds'] == "fixed"
    os.remove('%s/hyperparameters_0.json' % save_dir)
    os.remove('%s/features_hyperparameters.json' % save_dir)
"""