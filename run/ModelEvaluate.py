#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import pandas as pd
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.evaluators.cross_validation import Evaluator
from chemml.args import TrainArgs
from chemml.model import set_model


def main(args: TrainArgs) -> None:
    dataset = Dataset.load(path=args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      features_kernel_type=args.features_kernel_type,
                                      features_hyperparameters=args.features_hyperparameters,
                                      features_hyperparameters_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters,
                                      kernel_pkl=os.path.join(args.save_dir, 'kernel.pkl'))
    model = set_model(args, kernel=kernel_config.kernel)
    if args.separate_test_path is not None:
        df = pd.read_csv(args.separate_test_path)
        if args.target_columns is None:
            df['null_target'] = 0.
        dataset_test = Dataset.from_df(df=df,
                                       pure_columns=args.pure_columns,
                                       mixture_columns=args.mixture_columns,
                                       reaction_columns=args.reaction_columns,
                                       feature_columns=args.feature_columns,
                                       target_columns=args.target_columns or ['null_target'],
                                       features_generator=args.features_generator,
                                       features_combination=args.features_combination,
                                       mixture_type=args.mixture_type,
                                       reaction_type=args.reaction_type,
                                       group_reading=args.group_reading,
                                       n_jobs=args.n_jobs)
        dataset_test.graph_kernel_type = args.graph_kernel_type
        dataset.unify_datatype(dataset_test.X_graph)
    else:
        dataset_test = None
    evaluator = Evaluator(save_dir=args.save_dir,
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
                          verbose=True)

    if args.separate_test_path is not None and args.target_columns is None:
        evaluator.fit(X=dataset.X, y=dataset.y)
        evaluator.predict(X=dataset_test.X, y=None, repr=dataset_test.repr.ravel()).to_csv(
            '%s/pred_ext.csv' % args.save_dir, sep='\t', index=False, float_format='%15.10f')
    else:
        evaluator.evaluate(external_test_dataset=dataset_test)


if __name__ == '__main__':
    main(args=TrainArgs().parse_args())
