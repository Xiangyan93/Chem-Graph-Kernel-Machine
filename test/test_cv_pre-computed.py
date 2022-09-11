# !/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
CWD = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('%s/..' % CWD)
from chemml.args import KernelArgs, TrainArgs


@pytest.mark.parametrize('dataset', ['freesolv', 'bace', 'st'])
def test_kernel_data_pure_graph(dataset):
    save_dir = '%s/data/_%s' % (CWD, dataset)
    assert not os.path.exists('%s/kernel.pkl' % save_dir)
    arguments = [
        '--save_dir', '%s' % save_dir,
        '--graph_kernel_type', 'graph',
        '--graph_hyperparameters', '%s/../hyperparameters/tMGR.json' % CWD
    ]
    args = KernelArgs().parse_args(arguments)
    from run.KernelCalc import main
    main(args)
    assert os.path.exists('%s/kernel.pkl' % save_dir)


@pytest.mark.parametrize('dataset', ['freesolv', 'bace', 'st'])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d'],
                                                ['morgan_count'],
                                                ['rdkit_2d_normalized', 'morgan']])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_kernel_pure_graph_features(dataset, features_generator, features_scaling):
    save_dir = '%s_%s_%s' % (dataset, ','.join(features_generator), features_scaling)
    arguments = [
        '--save_dir', '%s/data/_%s' % (CWD, save_dir),
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--pure_columns', 'smiles',
        '--target_columns', dataset,
        '--n_jobs', '6',
        '--features_generator',
    ] + features_generator
    if features_scaling:
        arguments += [
            '--features_mol_normalize'
        ]
        if dataset == 'st':
            arguments += [
                '--features_add_normalize'
            ]
    if dataset == 'st':
        arguments += [
            '--feature_columns', 'T'
        ]
    args = KernelArgs().parse_args(arguments)
    from run.ReadData import main
    main(args)


@pytest.mark.parametrize('dataset', ['np', 'solubility'])
def test_read_data_mixture_graph(dataset):
    save_dir = dataset
    arguments = [
        '--save_dir', '%s/data/_%s' % (CWD, save_dir),
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns', 'mixture',
        '--target_columns', dataset,
        '--n_jobs', '6',
    ]
    if dataset == 'solubility':
        arguments += [
            '--feature_columns', 'T', 'P'
        ]
    args = CommonArgs().parse_args(arguments)
    from run.ReadData import main
    main(args)


@pytest.mark.parametrize('dataset', ['np', 'solubility'])
@pytest.mark.parametrize('features_generator', [['rdkit_2d_normalized'],
                                                ['morgan'],
                                                ['rdkit_2d'],
                                                ['morgan_count'],
                                                ['rdkit_2d_normalized', 'morgan']])
@pytest.mark.parametrize('features_combination', ['mean', 'concat'])
@pytest.mark.parametrize('features_scaling', [True, False])
def test_read_data_mixture_graph_features(dataset, features_generator, features_combination, features_scaling):
    save_dir = '%s_%s_%s_%s' % (dataset, ','.join(features_generator), features_combination, features_scaling)
    arguments = [
        '--save_dir', '%s/data/_%s' % (CWD, save_dir),
        '--data_path', '%s/data/%s.csv' % (CWD, dataset),
        '--mixture_columns', 'mixture',
        '--target_columns', dataset,
        '--n_jobs', '6',
        '--features_combination', features_combination,
        '--features_generator',
    ] + features_generator
    if features_scaling:
        arguments += [
            '--features_mol_normalize'
        ]
        if dataset == 'solubility':
            arguments += [
                '--features_add_normalize'
            ]
    if dataset == 'solubility':
        arguments += [
            '--feature_columns', 'T', 'P'
        ]
    args = CommonArgs().parse_args(arguments)
    from run.ReadData import main
    main(args)


"""
@pytest.mark.parametrize('dataset', ['react'])
def test_read_data_reaction(dataset):
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
