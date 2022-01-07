#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.gaussian_process._gpr import *
from sklearn.preprocessing import StandardScaler
from ..GPRgraphdot.gpr import predict_
import pickle
import os


class GPR(GaussianProcessRegressor):
    def __init__(self, kernel=None, *, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 copy_X_train=True, random_state=None, y_scale=True):
        super().__init__(kernel=kernel, alpha=alpha, optimizer=optimizer,
                         n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=False, copy_X_train=copy_X_train,
                         random_state=random_state)
        self.y_scale = y_scale
        self.kernel_ = clone(self.kernel)

    def fit(self, X, y):
        # scale y according to train y and save the scalar
        if self.y_scale:
            self.scaler = StandardScaler().fit(y.reshape(-1, 1))
            super().fit(X, self.scaler.transform(y.reshape(-1, 1)).ravel())
        else:
            super().fit(X, y)
        return self

    def predict_(self, X, return_std=False, return_cov=False):
        """ Copy from sklearn but remove the check_array part since the graphs
        input is not valid in that function.
        """
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.
            if self.y_scale:
                y_mean = self.scaler.inverse_transform(y_mean.reshape(-1, 1))\
                    .ravel()
            if return_cov:
                v = cho_solve((self.L_, True), K_trans.T)  # Line 5
                y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # cache result of K_inv computation
                if self._K_inv is None:
                    # compute inverse K_inv of K based on its Cholesky
                    # decomposition L and its inverse L_inv
                    L_inv = solve_triangular(self.L_.T,
                                             np.eye(self.L_.shape[0]))
                    self._K_inv = L_inv.dot(L_inv.T)

                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i",
                                   np.dot(K_trans, self._K_inv), K_trans)
                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def predict(self, X, return_std=False, return_cov=False):
        return predict_(self.predict_, X, return_std=return_std,
                        return_cov=return_cov)

    def predict_loocv(self, X, y, return_std=False):  # return loocv prediction
        if not hasattr(self, 'kernel_'):
            self.kernel_ = self.kernel
        K = self.kernel_(X)
        if self.y_scale:
            self.scaler = StandardScaler().fit(y.reshape(-1, 1))
            y_ = self.scaler.transform(y.reshape(-1, 1)).flatten()
        else:
            y_ = y - self._y_train_mean
        K[np.diag_indices_from(K)] += self.alpha
        I_mat = np.eye(K.shape[0])
        K_inv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(K,lower=True), I_mat)
        # K_inv = np.linalg.inv(K)
        y_pred = y_ - (K_inv.dot(y_).T / K_inv.diagonal()).T
        if self.y_scale:
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_pred += self._y_train_mean
        if return_std:
            y_std = np.sqrt(1 / K_inv.diagonal())
            return y_pred, y_std
        else:
            return y_pred

    def save(self, path, filename='model.pkl', overwrite=False):
        """Save the trained GaussianProcessRegressor with the associated data
        as a pickle.

        Parameters
        ----------
        path: str
            The directory to store the saved model.
        filename: str
            The file name for the saved model.
        overwrite: bool
            If True, a pre-existing file will be overwritten. Otherwise, a
            runtime error will be raised.
        """
        f_model = os.path.join(path, filename)
        if os.path.isfile(f_model) and not overwrite:
            raise RuntimeError(
                f'Path {f_model} already exists. To overwrite, set '
                '`overwrite=True`.'
            )
        store = self.__dict__.copy()
        store['theta'] = self.kernel.theta
        store['theta_'] = self.kernel_.theta
        store.pop('kernel', None)
        store.pop('kernel_', None)
        pickle.dump(store, open(f_model, 'wb'), protocol=4)

    def load(self, path, filename='model.pkl'):
        """Load a stored GaussianProcessRegressor model from a pickle file.

        Parameters
        ----------
        path: str
            The directory where the model is saved.
        filename: str
            The file name for the saved model.
        """
        f_model = os.path.join(path, filename)
        store = pickle.load(open(f_model, 'rb'))
        theta = store.pop('theta')
        theta_ = store.pop('theta_')
        self.__dict__.update(**store)
        self.kernel.theta = theta
        self.kernel_.theta = theta_

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        if 'theta_' not in store_dict:
            raise RuntimeError(
                f'Please use sklearn GPR for {f_model}'
            )
        kernel_ = kernel.clone_with_theta(store_dict.pop('theta_'))
        model = cls(kernel=kernel)
        model.kernel_ = kernel_
        model.__dict__.update(**store_dict)
        return model
