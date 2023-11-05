# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
import shutil
sys.path.append('%s/..' % CWD)
from chemml.args import CommonArgs
from run.ReadData import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
    ('bace', ['smiles'], ['bace']),
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
def test_ReadData_PureGraph(dataset):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
    ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_PureGraph_FeaturesAdd(dataset, group_reading, features_scaling):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--feature_columns'
    ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    if features_scaling:
        arguments += [
            '--features_add_normalize'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv']),
    ('bace', ['smiles'], ['bace']),
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX']),
    ('np', ['smiles1', 'smiles2'], ['np']),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_PureGraph_FeaturesMol(dataset, features_generator, features_scaling):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            ','.join(features_generator), features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--features_generator',
    ] + features_generator
    if features_scaling:
        arguments += [
            '--features_mol_normalize'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('st', ['smiles'], ['st'], ['T']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_PureGraph_FeaturesAddMol(dataset, group_reading, features_generator, features_scaling):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               group_reading, ','.join(features_generator), features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--features_generator',
    ] + features_generator + [
        '--feature_columns'
    ] + features_columns
    if features_scaling:
        arguments += [
            '--features_mol_normalize',
            '--features_add_normalize'
        ]
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np']),
])
def test_ReadData_MixtureGraph(dataset):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
    ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_MixtureGraph_FeaturesAdd(dataset, group_reading, features_scaling):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                            group_reading, features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--feature_columns',
    ] + features_columns + [
        '--feature_columns'
    ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    if features_scaling:
        arguments += [
            '--features_add_normalize'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np']),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_MixtureGraph_FeaturesMol(dataset, features_generator, features_combination, features_scaling):
    dataset, pure_columns, target_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               ','.join(features_generator), features_combination, features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--features_combination', features_combination,
        '--features_generator',
    ] + features_generator
    if features_scaling:
        arguments += [
            '--features_mol_normalize'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


@pytest.mark.parametrize('dataset', [
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P']),
])
@pytest.mark.parametrize('group_reading', [True, False])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_ReadData_MixtureGraph_FeaturesMolAdd(dataset, group_reading, features_generator, features_combination,
                                              features_scaling):
    dataset, pure_columns, target_columns, features_columns = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                                  group_reading, ','.join(features_generator), features_combination,
                                                  features_scaling)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
        '--features_combination', features_combination,
        '--features_generator',
    ] + features_generator + [
        '--feature_columns'
    ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    if features_scaling:
        arguments += [
            '--features_mol_normalize',
            '--features_add_normalize'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)
    assert os.path.exists('%s/dataset.pkl' % save_dir)


def test_ReadData_reaction():
    # TODO
    return
    """
    arguments = [
        '--save_dir', '%s/data/_%s' % (CWD, dataset),
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--reaction_columns', 'good_smarts',
        '--target_columns', 'reaction_type',
        '--n_jobs', '6',
    ]
    args = CommonArgs().parse_args(arguments)
    from run.ReadData import main
    main(args)
    """
