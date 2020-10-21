"""
Gaussian processes regression using Nystrom approximation.

The hyperparameter training process is designed to be self-consistent process. The K_core selection of Nystrom is
dependent on the kernel, and the kernel hyperparameter optimization is dependent on K_core.

The NystromGaussianProcessRegressor.fit_robust() need to be call several time to ensure convergence.

Drawbacks:
************************************************************************************************************************
The self-consistent process is not always converged. So how many loops are used is quite tricky.

For critical density prediction, it is not converged.
************************************************************************************************************************

Examples:
************************************************************************************************************************
N = 3  # N=1 for critical density.
for i in range(N):
    model = NystromGaussianProcessRegressor(kernel=kernel, random_state=0,
                                            kernel_cutoff=0.95, normalize_y=True,
                                            alpha=alpha).fit_robust(X, y)
    kernel = model.kernel_
************************************************************************************************************************
"""
from sklearn.gaussian_process._gpr import *
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eigh
import pickle
import os


class GPR(GaussianProcessRegressor):
    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.

        Parameters
        ----------
        X : sequence of length n_samples
            Query points where the GP is evaluated.
            Could either be array-like with shape = (n_samples, n_features)
            or a list of objects.

        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
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

    def save(self, dir):
        f_model = os.path.join(dir, 'model.pkl')
        store_dict = self.__dict__.copy()
        store_dict['theta'] = self.kernel.theta
        store_dict.pop('kernel', None)
        store_dict['theta_'] = self.kernel_.theta
        store_dict.pop('kernel_', None)
        pickle.dump(store_dict, open(f_model, 'wb'), protocol=4)

    def load(self, dir):
        f_model = os.path.join(dir, 'model.pkl')
        return self.load_cls(f_model, self.kernel)

    @classmethod
    def load_cls(cls, f_model, kernel):
        store_dict = pickle.load(open(f_model, 'rb'))
        kernel = kernel.clone_with_theta(store_dict.pop('theta'))
        kernel_ = kernel.clone_with_theta(store_dict.pop('theta_'))
        model = cls(kernel)
        model.kernel_ = kernel_
        model.__dict__.update(**store_dict)
        return model


class RobustFitGaussianProcessRegressor(GPR):
    def __init__(self, y_scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_scale = y_scale

    def fit(self, X, y):
        # scale y according to train y and save the scalar
        if self.y_scale:
            self.scaler = StandardScaler().fit(y.reshape(-1, 1))
            super().fit(X, self.scaler.transform(y.reshape(-1, 1)).flatten())
        else:
            super().fit(X, y)
        return self

    def predict(self, *args, **kwargs):
        result = super().predict(*args, **kwargs)
        if self.y_scale:
            if type(result) is tuple:
                y_back = self.scaler.inverse_transform(result[0].reshape(-1, 1)).flatten()
                return y_back, result[1]
            else:
                return self.scaler.inverse_transform(result.reshape(-1, 1)).flatten()
        else:
            return result

    def fit_robust(self, X, y, core_predict=True, cycle=1):
        self.fit(X, y)
        for i in range(cycle):
            try:
                print('Try to fit the data with alpha = ', self.alpha)
                self.fit(X, y)
                print('Success fit the data with %i-th cycle alpha' % (i + 1))
            except Exception as e:
                print('error info: ', e)
                self.alpha *= 1.1
                if i == cycle - 1:
                    print('Attempted alpha failed in %i-th cycle. The training is terminated for unstable numerical '
                          'issues may occur.' % (cycle + 1))
                    return None
            else:
                return self

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
