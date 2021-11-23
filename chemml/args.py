#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import numpy as np


Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score',
                 'rmse', 'mae', 'mse', 'r2', 'max']


class CommonArgs(Tap):
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""
    data_public: Literal['qm7', 'qm9'] = None
    """Use public data sets."""
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
    reaction_type: Literal['reaction', 'agent', 'reaction+agent'] = 'reaction'
    """How the chemical reaction is represented."""
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    group_reading: bool = False
    """Find unique input strings first, then read the data."""
    def __init__(self, *args, **kwargs):
        super(CommonArgs, self).__init__(*args, **kwargs)

    @property
    def graph_columns(self) -> List[str]:
        graph_columns = []
        if self.pure_columns is not None:
            graph_columns += self.pure_columns
        if self.mixture_columns is not None:
            graph_columns += self.mixture_columns
        if self.reaction_columns is not None:
            graph_columns += self.reaction_columns
        return graph_columns

    def update_columns(self, keys: List[str]):
        """Add all undefined columns to target_columns"""
        if self.target_columns is not None:
            return
        else:
            used_columns = self.graph_columns
            if self.feature_columns is not None:
                used_columns += self.feature_columns
            for key in used_columns:
                keys.remove(key)
            self.target_columns = keys

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.group_reading:
            if self.feature_columns is None:
                raise ValueError('feature_columns must be assigned when using group_reading.')


class KernelArgs(CommonArgs):
    graph_kernel_type: Literal['graph', 'preCalc'] = None
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
    features_mol_normalize: bool = False
    """Nomralize the molecular features_mol."""
    features_add_normalize: bool = False
    """Nomralize the additonal features_mol."""
    single_features_hyperparameter: bool = True
    """Use the same hyperparameter for all features."""

    @property
    def features_hyperparameters_bounds(self):
        if self.features_hyperparameters_min is None or self.features_hyperparameters_max is None:
            return 'fixed'
        else:
            return [(self.features_hyperparameters_min[i], self.features_hyperparameters_max[i])
                    for i in range(len(self.features_hyperparameters))]

    @property
    def ignore_features_add(self) -> bool:
        if self.feature_columns is None and \
                self.features_hyperparameters is None and \
                self.features_hyperparameters_file is None:
            return True
        else:
            return False

    def process_args(self) -> None:
        super().process_args()


class KernelBlockArgs(KernelArgs):
    block_size: int = 5000
    """"""
    block_id: Tuple[int, int]
    """"""

    @property
    def X_idx(self):
        return self.block_id[0] * self.block_size, (self.block_id[0] + 1) * self.block_size

    @property
    def Y_idx(self):
        return self.block_id[1] * self.block_size, (self.block_id[1] + 1) * self.block_size

    def process_args(self) -> None:
        super().process_args()
        assert self.block_id[1] >= self.block_id[0]


class TrainArgs(KernelArgs):
    dataset_type: Literal['regression', 'classification', 'multiclass'] = None
    """
    Type of dataset. This determines the loss function used during training.
    """
    model_type: Literal['gpr', 'svc', 'gpc', 'gpr_nystrom']
    """Type of model to use"""
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
    no_proba: bool = False
    """Use predict_proba for classification task."""
    evaluate_train: bool = False
    """If set True, evaluate the model on training set."""
    detail: bool = False
    """If set True, 5 most similar molecules in the training set will be save in the test_*.log."""
    save_model: bool = False
    """Save the trained model file."""

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

    def kernel_args(self):
        return super()

    def process_args(self) -> None:
        super().process_args()
        if self.dataset_type == 'regression':
            assert self.model_type in ['gpr', 'gpr_nystrom']
            for metric in self.metrics:
                assert metric in ['rmse', 'mae', 'mse', 'r2', 'max']
        elif self.dataset_type == 'classification':
            assert self.model_type in ['gpc', 'svc']
            for metric in self.metrics:
                assert metric in ['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score']
        else:
            assert self.model_type in ['gpc', 'svc']
            for metric in self.metrics:
                assert metric in ['accuracy', 'precision', 'recall', 'f1_score']

        if 'accuracy' in self.metrics:
            assert self.no_proba

        if self.split_type == 'loocv':
            assert self.num_folds == 1
            assert self.model_type == 'gpr'

        if self.model_type in ['gpr', 'gpr_nystrom']:
            assert self.alpha is not None

        if self.model_type == 'svc':
            assert self.C is not None

        if self.split_type == 'loocv':
            assert self.dataset_type == 'regression'

        if not hasattr(self, 'optimizer'):
            self.optimizer = None
        if not hasattr(self, 'batch_size'):
            self.batch_size = None

        if self.save_model:
            assert self.num_folds == 1
            assert self.split_sizes[0] > 0.99999
            assert self.model_type == 'gpr'


class PredictArgs(TrainArgs):
    test_path: str
    """Path to CSV file containing testing data for which predictions will be made."""
    preds_path: str = 'test.log'
    """Path to CSV file where predictions will be saved."""

    def process_args(self) -> None:
        super().process_args()


class HyperoptArgs(TrainArgs):
    num_iters: int = 20
    """Number of hyperparameter choices to try."""
    alpha_bounds: Tuple[float, float] = None
    """Bounds of alpha used in GPR."""
    alpha_uniform: float = None
    """"""
    C_bounds: Tuple[float, float] = (1e-3, 1e3)
    """Bounds of C used in SVC."""
    C_uniform: float = None
    """"""
    optimizer: Literal['SLSQP', 'L-BFGS-B', 'BFGS', 'fmin_l_bfgs_b', 'sgd', 'rmsprop', 'adam'] = None
    """Optimizer"""
    batch_size: int = None
    """batch_size"""

    @property
    def minimize_score(self) -> bool:
        """Whether the model should try to minimize the score metric or maximize it."""
        return self.metric in {'rmse', 'mae', 'mse', 'r2'}

    @property
    def opt_alpha(self) -> bool:
        if self.alpha_bounds is not None and \
                self.model_type in ['gpr', 'gpr_nystrom']:
            return True
        else:
            return False

    @property
    def opt_C(self) -> bool:
        if self.C_bounds is not None and \
                self.model_type == 'svc':
            return True
        else:
            return False

    def process_args(self) -> None:
        super().process_args()
        assert self.graph_kernel_type != 'preCalc'
        if self.optimizer in ['L-BFGS-B']:
            assert self.model_type == 'gpr'


class ActiveLearningArgs(TrainArgs):
    learning_algorithm: Literal['unsupervised', 'supervised', 'random'] = \
        'unsupervised'
    """Active learning algorithm."""
    sample_add_algorithm: Literal['nlargest', 'cluster', 'random'] = 'nlargest'
    """Sample adding algorithm."""
    initial_size: int = 5
    """Initial sample size of active learning."""
    add_size: int = 1
    """Number of samples added per active learning step."""
    pool_size: int = None
    """
    A subset of the sample pool is randomly selected for active learning. 
    None means all samples are selected.
    """
    cluster_size: int = None
    """If sample_add_algorithm='cluster', N worst samples are selected for 
    clustering."""
    stop_uncertainty: List[float] = None
    """If learning_algorithm='unsupervised', stop active learning if the 
    uncertainty is smaller than stop_uncertainty."""
    stop_size: int = None
    """Stop active learning when N samples are selected."""
    evaluate_stride: int = None
    """Evaluate the model performance every N samples."""
    surrogate_kernel: str = None
    """Specify the kernel pickle file for surrogate model."""
    def process_args(self) -> None:
        super().process_args()
        # active learning is only valid for GPR
        assert self.dataset_type == 'regression'
        assert self.model_type == 'gpr'
        assert self.split_type == 'random'
        if self.stop_uncertainty is not None:
            assert self.learning_algorithm == 'unsupervised'
        if self.cluster_size is not None:
            assert self.sample_add_algorithm == 'cluster'
        if self.sample_add_algorithm == 'nlargest':
            self.cluster_size = self.add_size
        assert self.initial_size >= 2
        if self.surrogate_kernel is not None:
            assert self.graph_kernel_type == 'preCalc'

        if self.stop_uncertainty is None:
            self.stop_uncertainty = [-1.0]
        else:
            self.stop_uncertainty.sort(reverse=True)


class EmbeddingArgs(KernelArgs):
    embedding_algorithm: Literal['tSNE', 'kPCA'] = 'tSNE'
    """Algorithm for data embedding."""
    n_components: int = 2
    """Dimension of the embedded space."""
    perplexity: float = 30.0
    """
    The perplexity is related to the number of nearest neighbors that
    is used in other manifold learning algorithms. Larger datasets
    usually require a larger perplexity. Consider selecting a value
    between 5 and 50. Different values can result in significantly
    different results.
    """
    n_iter: int = 1000
    """Maximum number of iterations for the optimization. Should be at least 250."""
    save_png: bool = False
    """If True, save the png file of the data embedding."""

    def process_args(self) -> None:
        super().process_args()
        if self.save_png:
            assert self.n_components == 2
