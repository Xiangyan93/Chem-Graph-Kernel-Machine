#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from mgktools.data import Dataset
from mgktools.data.split import dataset_split
from mgktools.kernels.utils import get_kernel_config
from mgktools.hyperparameters.hyperopt import bayesian_optimization
from chemml.model import set_model
from chemml.args import HyperoptArgs


def main(args: HyperoptArgs) -> None:
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    if args.num_splits == 1:
        datasets = [dataset]
    else:
        datasets = dataset_split(dataset=dataset, split_type='random',
                                 sizes=tuple([1 / args.num_splits] * args.num_splits))
    # set kernel_config
    kernel_config = get_kernel_config(dataset=dataset,
                                      graph_kernel_type=args.graph_kernel_type,
                                      features_kernel_type=args.features_kernel_type,
                                      features_hyperparameters=args.features_hyperparameters,
                                      features_hyperparameters_bounds=args.features_hyperparameters_bounds,
                                      features_hyperparameters_file=args.features_hyperparameters_file,
                                      mgk_hyperparameters_files=args.graph_hyperparameters)
    if args.optimizer is None:
        best_hyperdict, results, hyperdicts = bayesian_optimization(save_dir=args.save_dir,
                                                                    datasets=datasets,
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
        if args.save_all:
            for i, hyperdict in enumerate(hyperdicts):
                if not os.path.exists('%s/%d' % (args.save_dir, i)):
                    os.mkdir('%s/%d' % (args.save_dir, i))
                kernel_config.update_from_space(hyperdict)
                kernel_config.save_hyperparameters('%s/%d' % (args.save_dir, i))
                open('%s/%d/loss' % (args.save_dir, i), 'w').write(str(results[i]))
    else:
        model = set_model(args, kernel=kernel_config.kernel)
        model.fit(dataset.X, dataset.y, loss=args.loss, verbose=True)
        kernel_config.update_from_theta()
        kernel_config.save_hyperparameters(args.save_dir)


if __name__ == '__main__':
    main(args=HyperoptArgs().parse_args())
