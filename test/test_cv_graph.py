# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os

CWD = os.path.dirname(os.path.abspath(__file__))
import sys
import pandas as pd
sys.path.append('%s/..' % CWD)
from mgktools.hyperparameters import (
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
)
from chemml.args import TrainArgs
from run.ModelEvaluate import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('testset', [
    ('loocv', '1'),
    ('random', '10'),
])
def test_cv_PureGraph_regression(dataset, model, testset):
    task = 'regression'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    split, num_folds = testset
    metric = 'rmse'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', additive_msnorm,
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--alpha', '0.01'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    if split == 'loocv':
        df = pd.read_csv('%s/loocv.log' % save_dir)
        assert len(df) > 0
        os.remove('%s/loocv.log' % save_dir)
    elif split == 'random':
        for i in range(int(num_folds)):
            df = pd.read_csv('%s/test_%d.log' % (save_dir, i))
            assert len(df) > 0
            os.remove('%s/test_%d.log' % (save_dir, i))

"""
@pytest.mark.parametrize('testset', [
    ('freesolv', 'regression', 'gpr', 'loocv', 'rmse', ['mae', 'r2'], '1'),
    ('freesolv', 'regression', 'gpr', 'random', 'rmse', ['mae', 'r2'], '10'),
    ('bace', 'binary', 'gpc', 'random', 'roc-auc', ['accuracy', 'precision', 'recall', 'f1_score', 'mcc'], '10'),
    ('bace', 'binary', 'svc', 'random', 'roc-auc', ['accuracy', 'precision', 'recall', 'f1_score', 'mcc'], '10')
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d'],
                                                ['morgan_count'],
                                                ['rdkit_2d_normalized', 'morgan']])
@pytest.mark.parametrize('features_kernel_type', ['rbf', 'dot_product'])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_cv_pure_graph_features(testset, features_generator, features_kernel_type, features_scaling):
    dataset, task, model, split, metric, extra_metric, num_folds = testset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(features_generator), features_scaling)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', '%s/../hyperparameters/tMGR.json' % CWD,
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--extra_metrics'
    ] + extra_metric + [
        '--features_kernel_type', features_kernel_type,
        '--features_hyperparameters', '1.0',
        '--features_generator',
    ] + features_generator
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    elif model == 'svc':
        arguments += ['--C', '1.0']

    args = TrainArgs().parse_args(arguments)
    main(args)
    if split == 'loocv':
        df = pd.read_csv('%s/loocv.log' % save_dir)
        assert len(df) > 0
        os.remove('%s/loocv.log' % save_dir)
    elif split == 'random':
        for i in range(num_folds):
            df = pd.read_csv('%s/test_%d.log' % (save_dir, i))
            assert len(df) > 0
            os.remove('%s/test_%d.log' % (save_dir, i))
"""
