# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('%s/..' % CWD)
from chemml.args import CommonArgs
from run.ReadData import main


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'], None, False),
    ('bace', ['smiles'], ['bace'], None, False),
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX'], None, False),
    ('np', ['smiles1', 'smiles2'], ['np'], None, False),
    ('st', ['smiles'], ['st'], ['T'], True),
    ('st', ['smiles'], ['st'], ['T'], False),
])
def test_read_data_pure_graph(dataset):
    dataset, pure_columns, target_columns, features_columns, group_reading = dataset
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns), group_reading)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
    ]
    if features_columns is not None:
        arguments += [
            '--feature_columns'
        ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('freesolv', ['smiles'], ['freesolv'], None, False),
    ('bace', ['smiles'], ['bace'], None, False),
    ('clintox', ['smiles'], ['FDA_APPROVED', 'CT_TOX'], None, False),
    ('np', ['smiles1', 'smiles2'], ['np'], None, False),
    ('st', ['smiles'], ['st'], ['T'], True),
    ('st', ['smiles'], ['st'], ['T'], False),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_read_data_pure_graph_features(dataset, features_generator, features_scaling):
    dataset, pure_columns, target_columns, features_columns, group_reading = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                               ','.join(features_generator), features_scaling, group_reading)
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
        if features_columns is not None:
            arguments += [
                '--features_add_normalize'
            ]
    if features_columns is not None:
        arguments += [
            '--feature_columns'
        ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np'], None, False),
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P'], True),
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P'], False),
])
def test_read_data_mixture_graph(dataset):
    dataset, pure_columns, target_columns, features_columns, group_reading = dataset
    save_dir = '%s/data/_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns), group_reading)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns'] + pure_columns + [
        '--target_columns'] + target_columns + [
        '--n_jobs', '6',
    ]
    if features_columns is not None:
        arguments += [
            '--feature_columns'
        ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)


@pytest.mark.parametrize('dataset', [
    ('np', ['mixture'], ['np'], None, False),
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P'], True),
    ('solubility', ['mixture'], ['Solubility'], ['T', 'P'], False),
])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d', 'morgan_count']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_read_data_mixture_graph_features(dataset, features_generator, features_combination, features_scaling):
    dataset, pure_columns, target_columns, features_columns, group_reading = dataset
    save_dir = '%s/data/_%s_%s_%s_%s_%s_%s_%s' % (CWD, dataset, ','.join(pure_columns), ','.join(target_columns),
                                                  ','.join(features_generator), features_combination, features_scaling,
                                                  group_reading)
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
        if features_columns is not None:
            arguments += [
                '--features_add_normalize'
            ]
    if features_columns is not None:
        arguments += [
            '--group_reading',
            '--feature_columns'
        ] + features_columns
    if group_reading:
        arguments += [
            '--group_reading'
        ]
    args = CommonArgs().parse_args(arguments)
    main(args)


def test_read_data_reaction():
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
