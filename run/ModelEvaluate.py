#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.evaluators.cross_validation import Evaluator
from chemml.args import TrainArgs
from chemml.evaluator import set_model


def main(args: TrainArgs) -> None:
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = 'pre-computed'
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      features_hyperparameters=args.features_hyperparameters,
                                      features_hyperparameters_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters,
                                      kernel_pkl=os.path.join(args.save_dir, 'kernel.pkl'))
    model = set_model(args, kernel=kernel_config.kernel)
    Evaluator(save_dir=args.save_dir,
              dataset=dataset,
              model=model,
              task_type=args.task_type,
              metrics=args.metrics,
              split_type=args.split_type,
              split_sizes=args.split_sizes,
              num_folds=args.num_folds,
              return_std=True if args.model_type == 'gpr' else False,
              return_proba=True if args.task_type == 'binary' else False,
              evaluate_train=False,
              n_similar=None,
              kernel=None,
              n_core=args.n_core,
              seed=args.seed,
              verbose=True).evaluate()


if __name__ == '__main__':
    main(args=TrainArgs().parse_args())
