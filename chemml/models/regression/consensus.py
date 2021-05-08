#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
import threading
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
from ...models.regression.GPRgraphdot import GPR as GPRgraphdot
from ...models.regression.GPRsklearn.gpr import GPR as GPRsklearn
from .GPRgraphdot.gpr import _predict


def _parallel_build_models(model, models, X, y, id, model_idx, n_models,
                           verbose=0):
    """
    Private function used to fit a consensus model in parallel."""
    if verbose > 1:
        print("building model %d of %d" % (model_idx + 1, n_models))
    np.random.seed(model_idx)
    idx = np.random.choice(np.arange(len(X)), models.n_sample_per_model,
                           replace=False)
    if X.ndim == 1:
        model.fit(X[idx], y[idx])
    elif X.ndim == 2:
        model.fit(X[idx, :], y[idx])
    else:
        raise RuntimeError(
            'X must be 1 or 2 dimensional'
        )
    model.X_id_ = id[idx]
    return model


def _accumulate_prediction(predict, X, out, out_u, lock, return_std=False):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    if return_std:
        prediction, uncertainty = predict(X, return_std=True)
        with lock:
            out.append(prediction)
            out_u.append(uncertainty)
    else:
        prediction = predict(X, return_std=False)

        with lock:
            if len(out) == 1:
                out[0].append(prediction)
            else:
                for i in range(len(out)):
                    out[i].append(prediction)


class ConsensusRegressor:
    def __init__(self, model, n_estimators=100, n_sample_per_model=2000,
                 n_jobs=1, verbose=0, consensus_rule='smallest_uncertainty',):
        self.model = model
        self.models = []
        self.n_estimators = n_estimators
        self.n_sample_per_model = n_sample_per_model
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.consensus_rule = consensus_rule
        assert (n_estimators > 0)

    def fit(self, X, y, id):
        models = [copy.copy(self.model) for i in range(self.n_estimators)]
        models = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          **_joblib_parallel_args(prefer='processes'))(
            delayed(_parallel_build_models)(
                m, self, X, y, id, i, len(models), verbose=self.verbose)
            for i, m in enumerate(models))
        self.models.extend(models)

    def predict(self, X, return_std=False, return_cov=False):
        assert (not return_cov)
        if self.model.__class__ in [GPRgraphdot, GPRsklearn]:
            return _predict(self.predict_gpr, X, return_std=return_std,
                            return_cov=return_cov)
        else:
            raise RuntimeError(
                f'The regressor {self.model} are not supported for '
                f'ConsensusRegressor yet'
            )

    def predict_gpr(self, X, return_std=False, return_cov=False):
        y_hat = []
        u_hat = []
        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(prefer='processes'))(
            delayed(_accumulate_prediction)(m.predict, X, y_hat, u_hat, lock,
                                            return_std=return_std)
            for m in self.models)
        y, u = self.majority_vote(np.asarray(y_hat), np.asarray(u_hat),
                                  self.consensus_rule)
        y_hat = y
        u_hat = u
        if return_std:
            return y_hat, u_hat
        else:
            return y_hat

    def majority_vote(self, y, u, rule):
        if rule == 'smallest_uncertainty':
            idx = u.argmin(axis=0)
            return np.array([y[idx[I]][I] for I in np.lib.index_tricks.ndindex(idx.shape)]), \
                   np.array([u[idx[I]][I] for I in np.lib.index_tricks.ndindex(idx.shape)])
        elif rule == 'weight_uncertainty':
            sigma = 10
            weight = np.exp(-sigma*u) / np.exp(-sigma*u).sum(axis=0)
            return (y * weight).sum(axis=0), (u * weight).sum(axis=0)
        elif rule == 'mean':
            return y.mean(axis=0), u.mean(axis=0)
        else:
            raise RuntimeError(
                f'Unknown predict_rule for ConsensusRegressor{rule}'
            )

    def save(self, path, overwrite=False):
        for i, m in enumerate(self.models):
            m.save(path, filename='model_%d.pkl' % i, overwrite=overwrite)

    def load(self, path):
        models = [copy.copy(self.model) for i in range(self.n_estimators)]
        for i, m in enumerate(models):
            m.load(path, filename='model_%d.pkl' % i)
        self.models = models

    @property
    def X_train_(self):
        return np.concatenate([m.X_train_ for m in self.models])