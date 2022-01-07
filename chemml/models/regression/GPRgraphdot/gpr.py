#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
import math
from graphdot.model.gaussian_process.nystrom import *
from graphdot.model.gaussian_process.gpr import minimize
from .sgd import *


def predict_(predict, X, return_std=False, return_cov=False, memory_save=True,
             n_memory_save=10000):
    if return_cov or not memory_save:
        return predict(X, return_std=return_std, return_cov=return_cov)
    else:
        N = X.shape[0]
        y_mean = []
        y_std = []
        for i in range(math.ceil(N / n_memory_save)):
            X_ = X[i * n_memory_save:(i + 1) * n_memory_save]
            if return_std:
                [y_mean_, y_std_] = predict(
                    X_, return_std=return_std, return_cov=return_cov)
                y_std.append(y_std_)
            else:
                y_mean_ = predict(
                    X_, return_std=return_std, return_cov=return_cov)
            y_mean.append(y_mean_)
        if return_std:
            return np.concatenate(y_mean), np.concatenate(y_std)
        else:
            return np.concatenate(y_mean)


class GPR(GaussianProcessRegressor):
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop('batch_size')
        super().__init__(*args, **kwargs)
        if self.optimizer == 'sgd':
            self.optimizer = sgd
        elif self.optimizer == 'rmsprop':
            self.optimizer = rmsprop
        elif self.optimizer == 'adam':
            self.optimizer = adam

    def fit(self, X, y, loss='likelihood', tol=1e-5, repeat=1,
            theta_jitter=1.0, verbose=False):
        """Train a GPR model. If the `optimizer` argument was set while
        initializing the GPR object, the hyperparameters of the kernel will be
        optimized using the specified loss function.

        Parameters
        ----------
        X: list of objects or feature vectors.
            Input values of the training data.
        y: 1D array
            Output/target values of the training data.
        loss: 'likelihood' or 'loocv'
            The loss function to be minimzed during training. Could be either
            'likelihood' (negative log-likelihood) or 'loocv' (mean-square
            leave-one-out cross validation error).
        tol: float
            Tolerance for termination.
        repeat: int
            Repeat the hyperparameter optimization by the specified number of
            times and return the best result.
        theta_jitter: float
            Standard deviation of the random noise added to the initial
            logscale hyperparameters across repeated optimization runs.
        verbose: bool
            Whether or not to print out the optimization progress and outcome.

        Returns
        -------
        self: GaussianProcessRegressor
            returns an instance of self.
        """
        self.X = X
        self.y = y

        '''hyperparameter optimization'''
        if self.optimizer:

            if loss == 'likelihood':
                objective = self.log_marginal_likelihood
            elif loss == 'loocv':
                objective = self.squared_loocv_error

            opt = self._hyper_opt(
                lambda theta, objective=objective: objective(
                    theta, eval_gradient=True, clone_kernel=False,
                    verbose=verbose
                ),
                self.kernel.theta.copy(),
                tol, repeat, theta_jitter, verbose
            )
            if verbose:
                print(f'Optimization result:\n{opt}')

            if opt.success:
                self.kernel.theta = opt.x
            else:
                raise RuntimeError(
                    f'Training using the {loss} loss did not converge, got:\n'
                    f'{opt}'
                )

        '''build and store GPR model'''
        self.K = K = self._gramian(self.X)
        self.Kinv, _ = self._invert(K)
        self.Ky = self.Kinv @ self.y
        return self

    def squared_loocv_error(self, theta=None, X=None, y=None,
                            eval_gradient=False, clone_kernel=True,
                            verbose=False):
        """Returns the squared LOOCV error of a given set of log-scale
        hyperparameters.

        Parameters
        ----------
        theta: array-like
            Kernel hyperparameters for which the log-marginal likelihood is
            to be evaluated. If None, the current hyperparameters will be used.
        X: list of objects or feature vectors.
            Input values of the training data. If None, `self.X` will be used.
        y: 1D array
            Output/target values of the training data. If None, `self.y` will
            be used.
        eval_gradient: boolean
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta will be returned
            alongside.
        clone_kernel: boolean
            If True, the kernel is copied so that probing with theta does not
            alter the trained kernel. If False, the kernel hyperparameters will
            be modified in-place.
        verbose: boolean
            If True, the log-likelihood value and its components will be
            printed to the screen.

        Returns
        -------
        squared_error: float
            Squared LOOCV error of theta for training data.
        squared_error_gradient: 1D array
            Gradient of the Squared LOOCV error with respect to the kernel
            hyperparameters at position theta. Only returned when eval_gradient
            is True.
        """
        theta = theta if theta is not None else self.kernel.theta
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        batch_size = self.batch_size

        if batch_size is not None and batch_size < len(X):
            idx = np.random.choice(list(range(len(X))), batch_size, replace=True)
            X = X[idx]
            y = y[idx]

        if clone_kernel is True:
            kernel = self.kernel.clone_with_theta(theta)
        else:
            kernel = self.kernel
            kernel.theta = theta

        t_kernel = time.perf_counter()
        if eval_gradient is True:
            K, dK = self._gramian(X, kernel=kernel, jac=True)
        else:
            K = self._gramian(X, kernel=kernel)
        t_kernel = time.perf_counter() - t_kernel

        t_linalg = time.perf_counter()

        Kinv, logdet = self._invert(K)
        if not isinstance(Kinv, np.ndarray):
            Kinv = Kinv @ np.eye(len(X))
        Kinv_diag = Kinv.diagonal()
        Ky = Kinv @ y
        e = Ky / Kinv_diag
        squared_error = 0.5 * np.sum(e ** 2)

        if eval_gradient is True:
            D_theta = np.zeros_like(theta)
            for i, t in enumerate(theta):
                dk = dK[:, :, i]
                KdK = Kinv @ dk
                D_theta[i] = (
                                     - (e / Kinv_diag) @ (KdK @ Ky)
                                     + (e ** 2 / Kinv_diag) @ (KdK @ Kinv).diagonal()
                             ) * np.exp(t)
            retval = (squared_error, D_theta)
        else:
            retval = squared_error

        t_linalg = time.perf_counter() - t_linalg

        if verbose:
            mprint.table(
                ('Sq.Err.', '%12.5g', squared_error),
                ('d(SqErr)', '%12.5g', squared_error),
                ('log|K| ', '%12.5g', logdet),
                ('Cond(K)', '%12.5g', np.linalg.cond(K)),
                ('t_GPU (s)', '%10.2g', t_kernel),
                ('t_CPU (s)', '%10.2g', t_linalg),
            )

        return retval

    def predict_(self, Z, return_std=False, return_cov=False):
        if not hasattr(self, 'Kinv'):
            raise RuntimeError('Model not trained.')
        Ks = self._gramian(Z, self.X)
        ymean = (Ks @ self.Ky) * self.y_std + self.y_mean
        if return_std is True:
            Kss = self._gramian(Z, diag=True)
            Kss.flat[::len(Kss) + 1] -= self.alpha
            std = np.sqrt(
                np.maximum(0, Kss - (Ks @ (self.Kinv @ Ks.T)).diagonal())
            )
            return (ymean, std)
        elif return_cov is True:
            Kss = self._gramian(Z)
            Kss.flat[::len(Kss) + 1] -= self.alpha
            cov = np.maximum(0, Kss - Ks @ (self.Kinv @ Ks.T))
            return (ymean, cov)
        else:
            return ymean

    def predict(self, X, return_std=False, return_cov=False):
        return predict_(self.predict_, X, return_std=return_std,
                        return_cov=return_cov)

    def predict_loocv(self, Z, z, return_std=False):
        assert(len(Z) == len(z))
        z = np.asarray(z)
        if self.normalize_y is True:
            z_mean, z_std = np.mean(z, axis=0), np.std(z, axis=0)
            z = (z - z_mean) / z_std
        else:
            z_mean, z_std = 0, 1
        Kinv, _ = self._invert(self._gramian(Z))
        Kinv_diag = (Kinv @ np.eye(len(Z))).diagonal()

        ymean = (z - ((Kinv @ z).T / Kinv_diag).T) * z_std + z_mean
        if return_std is True:
            std = np.sqrt(1 / np.maximum(Kinv_diag, 1e-14))
            return (ymean, std)
        else:
            return ymean

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        model = cls(kernel)
        model.__dict__.update(**store_dict)
        return model

    """sklearn GPR parameters"""

    @property
    def kernel_(self):
        return self.kernel

    @property
    def X_train_(self):
        return self._X

    @X_train_.setter
    def X_train_(self, value):
        self._X = value

