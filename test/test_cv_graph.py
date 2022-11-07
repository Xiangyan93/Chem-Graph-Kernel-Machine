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
@pytest.mark.parametrize('split_set', [
    ('loocv', '1'),
    ('random', '10'),
])
@pytest.mark.parametrize('graph_hyperparameters', [
    additive, additive_pnorm, additive_msnorm, additive_norm,
    product, product_pnorm, product_msnorm, product_norm
])
def test_cv_PureGraph_Regression(dataset, model, split_set, graph_hyperparameters):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    task = 'regression'
    split, num_folds = split_set
    metric = 'rmse'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', graph_hyperparameters,
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
        df = pd.read_csv('%s/loocv.csv' % save_dir)
        assert len(df) > 0
        os.remove('%s/loocv.csv' % save_dir)
    elif split == 'random':
        for i in range(int(num_folds)):
            df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
            assert len(df) > 0
            os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('modelset', [('gpr', False, None),
                                      ('gpr', True, '4'),
                                      ('gpr_nystrom', False, '4'),
                                      ('gpr_nle', False, '4')])
def test_cv_PureGraph_regression_ExtTest(dataset, modelset):
    task = 'regression'
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    model, ensemble, n_core = modelset
    split, num_folds = None, '1'
    metric = 'rmse'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', additive_msnorm,
        '--task_type', task,
        '--model_type', model,
        '--metric', metric,
        '--num_folds', num_folds,
        '--alpha', '0.01',
        '--separate_test_path', '%s/data/%s_test.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns
    if ensemble:
        arguments += ['--ensemble',
                      '--n_estimator', '2',
                      '--n_sample_per_model', n_core]
    if model == 'gpr_nystrom':
        arguments += ['--n_core', n_core]
    elif model == 'gpr_nle':
        arguments += ['--n_local', n_core]

    args = TrainArgs().parse_args(arguments)
    main(args)
    if split == 'loocv':
        df = pd.read_csv('%s/loocv.csv' % save_dir)
        assert len(df) > 0
        os.remove('%s/loocv.csv' % save_dir)
    elif split == 'random':
        for i in range(int(num_folds)):
            df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
            assert len(df) > 0
            os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_PureGraph_Regression_FeaturesAdd(dataset, group_reading, features_scaling, model, split_set):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    task = 'regression'
    split, num_folds = split_set
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
        '--alpha', '0.01',
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '100.0'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_cv_PureGraph_Regression_FeaturesMol(dataset, model, split_set, features_generator, features_scaling):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            ','.join(features_generator), features_scaling)
    task = 'regression'
    split, num_folds = split_set
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
        '--alpha', '0.01',
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '1.0'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_PureGraph_Regression_FeaturesAddMol(dataset, group_reading, features_scaling, features_generator,
                                                model, split_set):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               group_reading, ','.join(features_generator), features_scaling)
    task = 'regression'
    split, num_folds = split_set
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
        '--alpha', '0.01',
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '1.0'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('bace', ['smiles'], ['bace']),
    # ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
@pytest.mark.parametrize('model', ['gpr', 'gpc'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_PureGraph_Binary(dataset, model, split_set):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    task = 'binary'
    split, num_folds = split_set
    metric = 'roc-auc'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
    ]
    if dataset == 'np':
        arguments += ['--graph_hyperparameters', additive_msnorm, additive_msnorm]
    else:
        arguments += ['--graph_hyperparameters', additive_msnorm]
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


def test_cv_PureGraph_Binary_FeaturesAdd():
    # TODO
    return


@pytest.mark.parametrize('dataset', [
    ('bace', ['smiles'], ['bace']),
    # ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
@pytest.mark.parametrize('model', ['gpr', 'gpc'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_cv_PureGraph_Binary_FeaturesMol(dataset, model, split_set, features_generator, features_scaling):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            ','.join(features_generator), features_scaling)
    task = 'binary'
    split, num_folds = split_set
    metric = 'roc-auc'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '1.0'
    ]
    if dataset == 'np':
        arguments += ['--graph_hyperparameters', additive_msnorm, additive_msnorm]
    else:
        arguments += ['--graph_hyperparameters', additive_msnorm]
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


def test_cv_PureGraph_Binary_FeaturesAddMol():
    # TODO
    return


def test_cv_MixtureGraph_Regression():
    # TODO
    return


@pytest.mark.parametrize('dataset', [
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_MixtureGraph_Regression_FeaturesAdd(dataset, group_reading, features_scaling, model, split_set):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    task = 'regression'
    split, num_folds = split_set
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
        '--alpha', '0.01',
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '100.0'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


def test_cv_MixtureGraph_Regression_FeaturesMol():
    # TODO
    return


@pytest.mark.parametrize('dataset', [
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('model', ['gpr'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
def test_cv_MixtureGraph_Regression_FeaturesAddMol(dataset, group_reading, features_scaling, model, split_set,
                                                   features_generator, features_combination):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                                  group_reading, ','.join(features_generator), features_combination,
                                                  features_scaling)
    task = 'regression'
    split, num_folds = split_set
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
        '--alpha', '0.01',
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '100.0'
    ]
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np']),
])
@pytest.mark.parametrize('model', ['gpr', 'gpc'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_MixtureGraph_Binary(dataset, model, split_set):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    task = 'binary'
    split, num_folds = split_set
    metric = 'roc-auc'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', additive_pnorm
    ]
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


def test_cv_MixtureGraph_Binary_FeaturesAdd():
    # TODO
    return


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np']),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
@pytest.mark.parametrize('features_scaling', [True, False])
@pytest.mark.parametrize('model', ['gpr', 'gpc'])
@pytest.mark.parametrize('split_set', [
    ('random', '10'),
])
def test_cv_MixtureGraph_Binary_FeaturesMol(dataset, features_generator, features_combination, features_scaling,
                                            model, split_set):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               ','.join(features_generator), features_combination, features_scaling)
    task = 'binary'
    split, num_folds = split_set
    metric = 'roc-auc'
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--task_type', task,
        '--model_type', model,
        '--split_type', split,
        '--metric', metric,
        '--num_folds', num_folds,
        '--graph_hyperparameters', additive_pnorm,
        '--features_kernel_type', 'rbf',
        '--features_hyperparameters', '10.0'
    ]
    if model == 'gpr':
        arguments += ['--alpha', '0.01']
    args = TrainArgs().parse_args(arguments)
    main(args)
    for i in range(int(num_folds)):
        df = pd.read_csv('%s/test_%d.csv' % (save_dir, i))
        assert len(df) > 0
        os.remove('%s/test_%d.csv' % (save_dir, i))


def test_cv_MixtureGraph_Binary_FeaturesAddMol():
    # TODO
    return
