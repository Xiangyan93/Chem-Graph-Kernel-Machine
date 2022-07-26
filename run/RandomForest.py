#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.evaluators.cross_validation import Evaluator


Metric = Literal['roc-auc', 'accuracy', 'precision', 'recall', 'f1_score',
                 'rmse', 'mae', 'mse', 'r2', 'max']


class RandomForestArgs(Tap):
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
    feature_columns: List[str] = None
    """
    Name of the columns containing additional features_mol such as temperature, 
    pressuer.
    """
    features_generator: List[str] = None
    """Method(s) of generating additional features_mol."""
    features_combination: Literal['concat', 'mean'] = None
    """How to combine features vector for mixtures."""
    target_columns: List[str] = None
    """
    Name of the columns containing target values.
    """
    group_reading: bool = False
    """Find unique input strings first, then read the data."""
    task_type: Literal['regression', 'binary', 'multi-class'] = None
    """
    Type of task. This determines the loss function used during training.
    """
    split_type: Literal['random', 'scaffold_balanced', 'loocv'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: Tuple[float, float] = (0.8, 0.2)
    """Split proportions for train/validation/test sets."""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    metric: Metric = None
    """metric"""
    extra_metrics: List[Metric] = []
    """Metrics"""
    seed: int = 0
    """Random seed."""

    @property
    def metrics(self) -> List[Metric]:
        return [self.metric] + self.extra_metrics


class RFClassifier(RandomForestClassifier):
    def predict_proba(self, X):
        return super().predict_proba(X)[:, 1]


def main(args: RandomForestArgs) -> None:
    if os.path.exists('%s/dataset.pkl' % args.save_dir):
        dataset = Dataset.load(path=args.save_dir)
    else:
        dataset = Dataset.from_df(df=pd.read_csv(args.data_path),
                                  pure_columns=args.pure_columns,
                                  mixture_columns=args.mixture_columns,
                                  feature_columns=args.feature_columns,
                                  target_columns=args.target_columns,
                                  features_generator=args.features_generator,
                                  features_combination=args.features_combination,
                                  group_reading=args.group_reading,
                                  n_jobs=args.n_jobs)
        dataset.save(args.save_dir)
    if args.task_type == 'regression':
        model = RandomForestRegressor()
    else:
        model = RFClassifier()
    Evaluator(save_dir=args.save_dir,
              dataset=dataset,
              model=model,
              task_type=args.task_type,
              metrics=args.metrics,
              split_type=args.split_type,
              split_sizes=args.split_sizes,
              num_folds=args.num_folds,
              return_proba=True if args.task_type != 'regression' else False,
              evaluate_train=False,
              n_similar=None,
              kernel=None,
              seed=args.seed,
              verbose=True).evaluate()


if __name__ == '__main__':
    main(args=RandomForestArgs().parse_args())
