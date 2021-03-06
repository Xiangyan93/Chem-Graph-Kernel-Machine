#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from typing import Dict, Union, List
from hyperopt import fmin, hp, tpe
import numpy as np
from chemml.evaluator import Evaluator
from chemml.args import HyperoptArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config


def save_best_params(results: List[float],
                     hyperdicts: List[Dict],
                     kernel_config,
                     args):
    best_idx = np.where(results == np.min(results))[0][0]
    best = hyperdicts[best_idx].copy()
    #
    if args.opt_alpha:
        open('%s/alpha' % args.save_dir, 'w').write('%s' % best.pop('alpha'))
    elif args.opt_C:
        open('%s/C' % args.save_dir, 'w').write('%s' % best.pop('C'))
    kernel_config.update_from_space(best)
    kernel_config.save_hyperparameters(args.save_dir)


def Bayesian(args: HyperoptArgs, dataset: Dataset, kernel_config):
    hyperdicts = []
    results = []

    def objective(hyperdict: Dict[str, float]) -> float:
        print('\nHyperopt Step')
        hyperdicts.append(hyperdict.copy())
        args.alpha = hyperdict.pop('alpha', None)
        args.C = hyperdict.pop('C', None)
        kernel_config.update_from_space(hyperdict)
        evaluator = Evaluator(args, dataset, kernel_config)
        result = evaluator.evaluate()
        if not args.minimize_score:
            result = - result
        results.append(result)
        dataset.graph_kernel_type = args.graph_kernel_type
        save_best_params(results, hyperdicts, kernel_config, args)
        return result

    SPACE = kernel_config.get_space()

    # add adjust hyperparameters of model
    if args.model_type == 'gpr':
        if args.alpha_bounds is None:
            pass
        elif args.alpha_uniform is None:
            SPACE['alpha'] = hp.loguniform('alpha',
                                           low=np.log(args.alpha_bounds[0]),
                                           high=np.log(args.alpha_bounds[1]))
        else:
            SPACE['alpha'] = hp.quniform('alpha',
                                         low=args.alpha_bounds[0],
                                         high=args.alpha_bounds[1],
                                         q=args.alpha_uniform)

    elif args.model_type == 'svc':
        if args.C_bounds is None:
            pass
        elif args.C_uniform is None:
            SPACE['C'] = hp.loguniform('C',
                                       low=np.log(args.C_bounds[0]),
                                       high=np.log(args.C_bounds[1]))
        else:
            SPACE['C'] = hp.loguniform('C',
                                       low=args.C_bounds[0],
                                       high=args.C_bounds[1],
                                       q=args.C_uniform)

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters,
         rstate=np.random.RandomState(args.seed))
    # get best hyperparameters.
    save_best_params(results, hyperdicts, kernel_config, args)


def main(args: HyperoptArgs) -> None:
    # read data
    dataset = Dataset.load(args.save_dir)
    dataset.graph_kernel_type = args.graph_kernel_type
    # get kernel config
    kernel_config = get_kernel_config(args, dataset)
    if args.optimizer is None:
        Bayesian(args, dataset, kernel_config)
    else:
        Evaluator(args, dataset, kernel_config).evaluate()
        kernel_config.update_from_theta()
        kernel_config.save_hyperparameters(args.save_dir)


if __name__ == '__main__':
    main(args=HyperoptArgs().parse_args())
