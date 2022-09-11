#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from mgktools.data import Dataset
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters.hyperopt import bayesian_optimization
from chemml.model import set_model
from chemml.args import HyperoptArgs


def main(args: HyperoptArgs) -> None:
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    print(dataset.X[:, 1])
    # set kernel_config
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      features_kernel_type=args.features_kernel_type,
                                      features_hyperparameters=args.features_hyperparameters,
                                      features_hyperparameters_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters)
    if args.optimizer is None:
        bayesian_optimization(save_dir=args.save_dir,
                              dataset=dataset,
                              kernel_config=kernel_config,
                              task_type=args.task_type,
                              model_type=args.model_type,
                              metric=args.metric,
                              split_type=args.split_type,
                              num_iters=args.num_iters,
                              alpha=args.alpha_,
                              alpha_bounds=args.alpha_bounds,
                              C=args.C_,
                              C_bounds=args.C_bounds,
                              seed=args.seed)
    else:
        model = set_model(args, kernel=kernel_config.kernel)
        model.fit(dataset.X, dataset.y, loss=args.loss, verbose=True)
        kernel_config.update_from_theta()
        kernel_config.save_hyperparameters(args.save_dir)


if __name__ == '__main__':
    main(args=HyperoptArgs().parse_args())
