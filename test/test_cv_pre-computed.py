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
from chemml.args import KernelArgs, TrainArgs


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
    ('scaffold_order', '10'),
])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
])
def test_cv_PreComputed_PureGraph_Regression(dataset, testset, graph_hyperparameters):
    task = 'regression'
    model = 'gpr'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    split, num_folds = testset
    # kernel computation
    assert not os.path.exists('%s/kernel.pkl' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', '%s' % graph_hyperparameters,
    ]
    args = KernelArgs().parse_args(arguments)
    from run.KernelCalc import main
    main(args)
    assert os.path.exists('%s/kernel.pkl' % save_dir)
    # cross validation
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'pre-computed',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--split_sizes', '0.8', '0.2',
        '--alpha', '0.01',
        '--metric', 'rmse',
        '--extra_metrics', 'r2', 'mae',
        '--num_folds', num_folds
    ]
    args = TrainArgs().parse_args(arguments)
    from run.ModelEvaluate import main
    main(args)
    os.remove('%s/kernel.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('bace', ['smiles'], ['bace']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
@pytest.mark.parametrize('model', ['gpr' , 'gpc', 'svc'])
@pytest.mark.parametrize('testset', [
    ('random', '10'),
])
@pytest.mark.parametrize('metric', ['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score', 'mcc'])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive_msnorm
])
def test_cv_PreComputed_PureGraph_Binary(dataset, model, testset, metric, graph_hyperparameters):
    task = 'binary'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    split, num_folds = testset
    # kernel computation
    assert not os.path.exists('%s/kernel.pkl' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters'] + ['%s' % graph_hyperparameters] * len(pure_columns)
    args = KernelArgs().parse_args(arguments)
    from run.KernelCalc import main
    main(args)
    assert os.path.exists('%s/kernel.pkl' % save_dir)
    # cross validation
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'pre-computed',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--split_sizes', '0.8', '0.2',
        '--metric', metric,
        '--num_folds', num_folds
    ]
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    elif model == 'svc':
        arguments += ['--C', '1.0']
    args = TrainArgs().parse_args(arguments)
    from run.ModelEvaluate import main
    main(args)
    os.remove('%s/kernel.pkl' % save_dir)
