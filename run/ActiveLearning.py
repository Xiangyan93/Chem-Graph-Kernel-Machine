#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.args import ActiveLearningArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config
from chemml.model import ActiveLearner
from chemml.data import dataset_split


def main(args: ActiveLearningArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(args, dataset, kernel_pkl=os.path.join(args.save_dir, 'kernel.pkl'))
    if args.surrogate_kernel is not None:
        kernel_config_surrogate = get_kernel_config(args, dataset,
                                                    kernel_pkl=os.path.join(args.save_dir, 'kernel_surrogate.pkl'))
    else:
        kernel_config_surrogate = kernel_config
    dataset, dataset_pool = dataset_split(
        dataset,
        split_type="random", seed=args.seed,
        sizes=(args.initial_size/len(dataset),
               1 - args.initial_size/len(dataset)),
    )
    ActiveLearner(args, dataset, dataset_pool, kernel_config, kernel_config_surrogate).run()


if __name__ == '__main__':
    main(args=ActiveLearningArgs().parse_args())
