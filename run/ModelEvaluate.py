#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.args import TrainArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config
from chemml.evaluator import Evaluator


def main(args: TrainArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.kernel_type = args.kernel_type
    kernel_config = get_kernel_config(args, dataset)
    # print(kernel_config.graph_hyperparameters)
    Evaluator(args, dataset, kernel_config.kernel).evaluate()


if __name__ == '__main__':
    main(args=TrainArgs().parse_args())
