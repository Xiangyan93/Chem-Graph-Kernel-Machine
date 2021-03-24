#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.args import TrainArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import set_kernel
from chemml.evaluator import Evaluator


def main(args: TrainArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.kernel_type = args.kernel_type
    evaluator = Evaluator(args, dataset).evaluate()


if __name__ == '__main__':
    main(args=TrainArgs().parse_args())
