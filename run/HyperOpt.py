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
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.kernel_type = args.kernel_type
    # get kernel config
    kernel_config = get_kernel_config(args, dataset)

    hyperdicts = []
    results = []

    def objective(hyperdict: Dict[str, List[Union[int, float]]]) -> float:
        print('\nHyperopt Step')
        hyperdicts.append(hyperdict.copy())
        if args.model_type == 'gpr':
            args.alpha = hyperdict.pop('alpha')
        elif args.model_type == 'svc':
            args.C = hyperdict.pop('C')
        kernel_config.update_space(hyperdict)
        evaluator = Evaluator(args, dataset, kernel_config)
        result = evaluator.evaluate()
        if not args.minimize_score:
            result = - result
        results.append(result)
        dataset.kernel_type = 'graph'
        return result

    SPACE = kernel_config.get_space()

    # add adjust hyperparameters of model
    if args.model_type == 'gpr':
        SPACE['alpha'] = hp.loguniform('alpha',
                                       low=np.log(args.alpha_bounds[0]),
                                       high=np.log(args.alpha_bounds[1]))
    elif args.model_type == 'svc':
        SPACE['C'] = hp.loguniform('C',
                                   low=np.log(args.C_bounds[0]),
                                   high=np.log(args.C_bounds[1]))

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters,
         rstate=np.random.RandomState(args.seed))
    # get best hyperparameters.
    best_idx = np.where(results == np.min(results))[0][0]
    best = hyperdicts[best_idx]
    #
    if args.model_type == 'gpr':
        open('%s/alpha' % args.save_dir, 'w').write('%s' % best.pop('alpha'))
    elif args.model_type == 'svc':
        open('%s/C' % args.save_dir, 'w').write('%s' % best.pop('C'))
    kernel_config.update_space(best)
    kernel_config.save(args.save_dir)


if __name__ == '__main__':
    main(args=HyperoptArgs().parse_args())
