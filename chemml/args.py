#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np


Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score',
                 'rmse', 'mae', 'mse', 'r2']


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    pure_columns: List[str] = None
    """
    For pure compounds.
    Name of the columns containing single SMILES or InChI string.
    """
    mixture_columns: List[str] = None
    """
    For mixtures.
    Name of the columns containing multiple SMILES or InChI string and 
    corresponding concentration.
    example: ['C', 0.5, 'CC', 0.3]
    """
    mixture_type: Literal['single_graph', 'multi_graph'] = 'single_graph'
    """How the mixture is represented."""
    reaction_columns: List[str] = None
    """
    For chemical reactions.
    Name of the columns containing single reaction smarts string.
    """
    reaction_type: Literal['reaction', 'agent', 'reaction+agent'] = \
        'reaction'
    """How the chemical reaction is represented."""
    feature_columns: List[str] = None
    """
    Name of the columns containing additional molfeatures such as temperature, 
    pressuer.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional molfeatures."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    unique_reading: bool = False
    """Find unique input strings first, then read the data."""
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)


class KernelArgs(CommonArgs):
    kernel_type: Literal['graph', 'preCalc'] = 'graph'
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters file for graph kernel."""
    molfeatures_hyperparameters: float = 1.0
    """hyperparameters for molecular molfeatures."""
    molfeatures_normalize: bool = False
    """Nomralize the molecular molfeatures."""
    addfeatures_hyperparameters: float = 1.0
    """hyperparameters for additional molfeatures."""
    addfeatures_normalize: bool = False
    """omral the additonal molfeatures."""


class TrainArgs(KernelArgs):
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of dataset. This determines the loss function used during training.
    """
    model_type: Literal['gpr', 'svc', 'gpc', 'gpr_nystrom']
    """Type of model to use"""
    optimizer: Literal['L-BFGS-B', 'fmin_l_bfgs_b', 'bayesian'] = None
    """Optimizer"""
    loss: Literal['loocv', 'likelihood'] = 'loocv'
    """The target loss function to minimize or maximize."""
    split_type: Literal['random', 'scaffold_balanced', 'loocv'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: Tuple[float, float] = (0.8, 0.2)
    """Split proportions for train/validation/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    alpha: float = 0.01
    """data noise used in gpr."""
    C: float = 1.0
    """C parameter used in Support Vector Machine."""
    seed: int = 0
    """Random seed."""

    ensemble: bool = False
    """use ensemble model."""
    n_estimator: int = 1
    """Ensemble model with n estimators."""
    n_sample_per_model: int = None
    """The number of samples use in each estimator."""
    ensemble_rule: Literal['smallest_uncertainty', 'weight_uncertainty',
                           'mean'] = 'weight_uncertainty'
    """The rule to combining prediction from estimators."""
    metric: str
    """metric"""
    extra_metric: List[str] = None
    """Metrics"""
    evaluate_train: bool = False
    """"""
    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self.check()

    @property
    def metrics(self):
        return [self.metric] + self.extra_metric

    def check(self):
        if self.split_type == 'loocv':
            assert self.dataset_type == 'regression'

    def kernel_args(self):
        return super()

class HyperoptArgs(TrainArgs):
    num_iters: int = 20
