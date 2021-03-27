#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pickle
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from typing import Dict, Union, List
from hyperopt import fmin, hp, tpe
import numpy as np
from chemml.evaluator import Evaluator
from chemml.args import TrainArgs, HyperoptArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config


def main(args: HyperoptArgs) -> None:
    dataset = Dataset.load(args.save_dir)
    dataset.kernel_type = args.kernel_type
    kernel_config = get_kernel_config(args, dataset)

    def objective(hyperparams: Dict[str, List[Union[int, float]]]) -> float:
        if args.model_type == 'gpr':
            args.alpha = hyperparams.pop('alpha')
        kernel_config.update_space(hyperparams)
        evaluator = Evaluator(args, dataset, kernel_config.kernel)
        return evaluator.evaluate()

    SPACE = kernel_config.get_space()
    if args.model_type == 'gpr':
        SPACE['alpha'] = hp.quniform('alpha', low=0.001, high=0.02, q=0.001)
    best = fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters,
                rstate=np.random.RandomState(args.seed))
    if args.model_type == 'gpr':
        print('Optimal alpha: ', best.pop('alpha'))
    kernel_config.update_space(best)
    kernel_config.save(args.save_dir)


if __name__ == '__main__':
    main(args=HyperoptArgs().parse_args())
