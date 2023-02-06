#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .args import TrainArgs
from mgktools.models import GPR, GPC, LRAGPR, NLEGPR, SVC, SVR, ConsensusRegressor
from mgktools.interpret.gpr import InterpretableGaussianProcessRegressor as IGPR


def set_model(args: TrainArgs,
              kernel):
    if args.model_type == 'gpr':
        if args.atomic_attribution:
            model = IGPR(
                kernel=kernel,
                optimizer=args.optimizer,
                alpha=args.alpha_,
                normalize_y=False,
            )
        else:
            model = GPR(
                kernel=kernel,
                optimizer=args.optimizer,
                alpha=args.alpha_,
                normalize_y=True,
            )
        if args.ensemble:
            model = ConsensusRegressor(
                model,
                n_estimators=args.n_estimator,
                n_sample_per_model=args.n_sample_per_model,
                n_jobs=args.n_jobs,
                consensus_rule=args.ensemble_rule
            )
    elif args.model_type == 'gpr_nystrom':
        model = LRAGPR(
            kernel=kernel,
            optimizer=args.optimizer,
            alpha=args.alpha_,
            normalize_y=True,
        )
    elif args.model_type == 'gpr_nle':
        n_jobs = 1 if args.graph_kernel_type == 'graph' else args.n_jobs
        model = NLEGPR(
            kernel=kernel,
            alpha=args.alpha_,
            n_local=args.n_local,
            n_jobs=n_jobs
        )
    elif args.model_type == 'gpc':
        model = GPC(
            kernel=kernel,
            optimizer=args.optimizer,
            n_jobs=args.n_jobs
        )
    elif args.model_type == 'svc':
        model = SVC(
            kernel=kernel,
            C=args.C_,
            probability=True
        )
    elif args.model_type == 'svr':
        model = SVR(
            kernel=kernel,
            C=args.C_,
        )
    else:
        raise RuntimeError(f'Unsupport model:{args.model_type}')
    return model
