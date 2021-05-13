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
from chemml.evaluator import ActiveLearner


def main(args: ActiveLearningArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    kernel_config = get_kernel_config(args, dataset)
    dataset, dataset_pool = dataset.split(
        split_type="random",
        sizes=(args.initial_size/len(dataset),
               1 - args.initial_size/len(dataset)))
    ActiveLearner(args, dataset, dataset_pool, kernel_config).run()


if __name__ == '__main__':
    main(args=ActiveLearningArgs().parse_args())
