#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
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

    @property
    def graph_columns(self):
        graph_columns = []
        if self.pure_columns is not None:
            graph_columns += self.pure_columns
        if self.mixture_columns is not None:
            graph_columns += self.mixture_columns
        if self.reaction_columns is not None:
            graph_columns += self.reaction_columns
        return graph_columns


class KernelArgs(CommonArgs):
    kernel_type: Literal['graph', 'preCalc'] = 'graph'
    """The type of kernel to use."""
    graph_hyperparameters: List[str] = None
    """hyperparameters file for graph kernel."""
    features_hyperparameters: List[float] = None
    """hyperparameters for molecular features."""
    features_hyperparameters_min: List[float] = None
    """hyperparameters for molecular features."""
    features_hyperparameters_max: List[float] = None
    """hyperparameters for molecular features."""
    features_hyperparameters_file: str = None
    """JSON file contains features hyperparameters"""

    molfeatures_normalize: bool = False
    """Nomralize the molecular molfeatures."""
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
    alpha: str = None
    """data noise used in gpr."""
    C: str = None
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
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    evaluate_train: bool = False
    """"""
    def __init__(self, *args, **kwargs) -> None:
        super(TrainArgs, self).__init__(*args, **kwargs)
        self.check()

    @property
    def metrics(self) -> List[str]:
        return [self.metric] + self.extra_metrics

    @property
    def alpha_(self) -> float:
        if isinstance(self.alpha, float):
            return self.alpha
        if os.path.exists(self.alpha):
            return float(open(self.alpha, 'r').read())
        else:
            return float(self.alpha)

    @property
    def C_(self) -> float:
        if isinstance(self.C, float):
            return self.C
        elif os.path.exists(self.C):
            return float(open(self.C, 'r').read())
        else:
            return float(self.C)

    def check(self):
        if self.split_type == 'loocv':
            assert self.dataset_type == 'regression'

    def kernel_args(self):
        return super()

    def process_args(self) -> None:
        if self.dataset_type == 'regression':
            assert self.model_type in ['gpr', 'gpr_nystrom']
        else:
            assert self.model_type in ['gpc', 'svc']

        if self.split_type == 'loocv':
            assert self.num_folds == 1
            assert self.model_type == 'gpr'

        if self.model_type in ['gpr', 'gpr_nystrom']:
            assert self.alpha is not None

        if self.model_type == 'svc':
            assert self.C is not None


class HyperoptArgs(TrainArgs):
    num_iters: int = 20
    """Number of hyperparameter choices to try."""
    alpha_bounds: Tuple[float, float] = (1e-3, 1e2)
    """Bounds of alpha used in GPR."""
    C_bounds: Tuple[float, float] = (1e-3, 1e3)
    """Bounds of C used in SVC."""

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'r2'}

    def process_args(self) -> None:
        super().process_args()
