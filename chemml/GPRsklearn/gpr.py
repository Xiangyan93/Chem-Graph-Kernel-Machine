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
from chemml.solver import CholSolver
from chemml.optimizer import (
    ensemble,
    sequential_threshold,
    l1_regularization,
    vanilla_lbfgs
)


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
        model = cls(kernel=kernel)
        model.kernel_ = kernel_
        model.__dict__.update(**store_dict)
        return model

    def unpack_theta(self, theta):
        xi = np.exp(theta[0])
        t = np.exp(theta[1:1 + len(self.kernel.theta)])
        s = np.exp(theta[1 + len(self.kernel.theta):])
        return xi, t, s

    @staticmethod
    def pack_theta(xi, t, s):
        return np.log(np.concatenate(([xi], t, s)))

    def get_raw_kernel_matrix(self, theta, jac=True):

        xi, t, s = self.unpack_theta(theta)

        self.kernel.theta = np.log(t)

        if jac is True:
            R, dR = self.kernel(self.X_train_, eval_gradient=True)
            return R.astype(np.float), dR.astype(np.float)
        else:
            R = self.kernel(self.X_train_)
            return R.astype(np.float)

    def likelihood(self, theta, jac=True, verbose=False):
        xi, t, s = self.unpack_theta(theta)
        y = np.copy(self.y_train_)
        scaler_forward = lambda z, mean=np.mean(y), std=np.std(y): (z - mean) / std
        y = scaler_forward(y)
        if jac is True:
            R, dR = self.get_raw_kernel_matrix(theta, jac=True)

            # normalization
            diag = R.diagonal()
            diag_rsqrt = diag ** -0.5
            diag_r15rt = diag ** -1.5
            K = diag_rsqrt[:, None] * R * diag_rsqrt[None, :]
            KK = K ** xi
            # derivative after normalization
            dK = []
            for i in range(dR.shape[-1]):
                dr = dR[:, :, i]
                ddiag = dr.diagonal()
                dk = (diag_rsqrt[:, None] * dr * diag_rsqrt[None, :]
                      - 0.5 * diag_r15rt[:, None] * R * diag_r15rt[None, :] * (
                                  np.outer(ddiag, diag) + np.outer(diag, ddiag)))
                dK.append(dk)
            dK = np.array(dK)
        else:
            R = self.get_raw_kernel_matrix(theta, jac=False)
            diag = R.diagonal()
            diag_rsqrt = diag ** -0.5
            K = diag_rsqrt[:, None] * R * diag_rsqrt[None, :]
            KK = K ** xi

        KK.flat[::len(self.X_train_) + 1] += s ** 2
        self.KK = KK
        KKinv = CholSolver(KK)

        KKy = KKinv(y)
        yKKy = y.dot(KKy)
        logdet = np.prod(np.linalg.slogdet(KK))
        if verbose:
            print('L %.5f = %.5f + %.5f' % (yKKy + logdet, yKKy, logdet), xi, t, s.mean())

        if jac is True:
            dKK_dxi = K ** xi * np.log(K)
            D_xi = (KKinv(dKK_dxi).trace() - KKy.dot(dKK_dxi).dot(KKy)) * xi
            D_s = (KKinv(np.eye(len(self.X_train_))).diagonal() - KKy ** 2) * (2 * s ** 2)
            D_theta = []
            for i in range(dK.shape[0]):
                dk = dK[i, :, :]
                dkk = dk * xi * K ** (xi - 1)
                dt = (KKinv(dkk).trace() - KKy.dot(dkk).dot(KKy)) * t[i]
                D_theta.append(dt)
            return yKKy + logdet, np.concatenate(([D_xi], D_theta, D_s))
        else:
            return yKKy + logdet

    def get_alpha(self, seed, opt='l1reg'):
        np.random.seed(seed)
        lmin = 1e-2
        lmax = 10
        xi0 = 2.0
        theta0 = np.exp(self.kernel.theta)
        if opt == 'l1reg':
            alpha = 1.0
            opt_ = ensemble(
                lambda: l1_regularization(
                    fun=lambda theta, gpr=self: gpr.likelihood(theta),
                    x0=self.pack_theta(xi0, theta0, np.random.uniform(lmin, 1.0, len(self.y_train_))),
                    strengths=np.concatenate(([0.0], np.zeros_like(self.kernel.theta), alpha * np.ones_like(self.y_train_))),
                    bounds_lower=self.pack_theta(1.05, np.exp(self.kernel.bounds[:, 0]), lmin * np.ones_like(self.y_train_)),
                    bounds_upper=self.pack_theta(16.0, np.exp(self.kernel.bounds[:, 1]), lmax * np.ones_like(self.y_train_)),
                    jac=True
                ),
                10
            )
        elif opt == 'seqth':
            th = 2.0
            opt_ = ensemble(
                lambda: sequential_threshold(
                    fun=lambda theta, gpr=self: gpr.likelihood(theta),
                    x0=self.pack_theta(xi0, theta0, np.random.uniform(lmin, 1.0, len(self.y_train_))),
                    thresholds=np.concatenate(
                        ([1e-7], 1e-7 * np.ones_like(self.kernel.theta), th * np.ones_like(self.y_train_))),
                    bounds_lower=self.pack_theta(1.05, np.exp(self.kernel.bounds[:, 0]), lmin * np.ones_like(self.y_train_)),
                    bounds_upper=self.pack_theta(16.0, np.exp(self.kernel.bounds[:, 1]), lmax * np.ones_like(self.y_train_)),
                    jac=True
                ),
                10
            )
        else:
            raise Exception('unknown optimizer')
        _, _, u = self.unpack_theta(opt_.x)
        return u**2


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

    def fit_robust(self, X, y, cycle=1):
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


'''
# This class cannot be used directly.
class NystromPreGaussianProcessRegressor(RobustFitGaussianProcessRegressor):
    def __init__(self, off_diagonal_cutoff=0.9, core_max=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_diagonal_cutoff = off_diagonal_cutoff
        self.core_max = core_max

    @staticmethod
    def Nystrom_solve(K_core, K_cross, eigen_cutoff=1e-10, debug=False):
        Wcc, Ucc = np.linalg.eigh(K_core)
        mask = Wcc > eigen_cutoff * max(Wcc)  # alpha  # !!!
        if debug:
            print('%i / %i eigenvalues are used in Nystrom Kcc' % (len(Wcc[mask]), len(Wcc)))
        Wcc = Wcc[mask]  # !!!
        Ucc = Ucc[:, mask]  # !!!
        Kccinv = (Ucc / Wcc).dot(Ucc.T)
        Uxx, Sxx, Vxx = np.linalg.svd(K_cross.T.dot((Ucc / Wcc ** 0.5).dot(Ucc.T)), full_matrices=False)
        mask = Sxx > eigen_cutoff * max(Sxx)  # !!!
        Uxx = Uxx[:, mask]  # !!!
        Sxx = Sxx[mask]  # !!!
        Kxx_ihalf = Uxx / Sxx
        return Kccinv, Kxx_ihalf

    @staticmethod
    def _nystrom_predict(kernel, C, X, Y, y, alpha=1e-10, return_std=False, return_cov=False, y_shift=0.0,
                         normalize_y=True):
        if normalize_y:
            y_mean = y.mean()
            y = np.copy(y) - y_mean
        else:
            y_mean = 0.
        Kcc = kernel(C)
        Kcx = kernel(C, X)
        Kcc[np.diag_indices_from(Kcc)] += alpha
        Kccinv, Kxx_ihalf = NystromPreGaussianProcessRegressor.Nystrom_solve(Kcc, Kcx, eigen_cutoff=alpha)
        Kyc = kernel(Y, C)
        left = Kyc.dot(Kccinv).dot(Kcx.dot(Kxx_ihalf))  # y*c
        right = Kxx_ihalf.T.dot(y)  # c*o
        y_mean += left.dot(right) + y_shift
        if return_cov:
            y_cov = kernel(Y) - left.dot(left.T)  # Line 6
            return y_mean, y_cov
        elif return_std:
            y_var = kernel.diag(Y)
            y_var -= np.einsum("ij,ij->i", left, left)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                print('%i predicted variances smaller than 0' % len(y_var[y_var_negative]))
                # print('They are: ', y_var[y_var_negative])
                print('most negative value: %e' % min(y_var[y_var_negative]))
                warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean

    def nystrom_predict(self, X, return_std=False, return_cov=False):
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if self.kernel is None or self.kernel.requires_vector_input:
            X = check_array(X, ensure_2d=True, dtype="numeric")
        else:
            X = check_array(X, ensure_2d=False, dtype=None)

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
            kernel = self.kernel_
            Kyc = kernel(X, self.core_X)
            left = Kyc.dot(self.left)
            y_mean = left.dot(self.right) + self._y_train_mean_full
            if return_cov:
                y_cov = kernel(X) - left.dot(left.T)  # Line 6
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                y_var -= np.einsum("ij,ij->i", left, left)
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    print('%i predicted variances smaller than 0' % len(y_var[y_var_negative]))
                    # print('They are: ', y_var[y_var_negative])
                    print('most negative value: %e' % min(y_var[y_var_negative]))
                    warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    @staticmethod
    def get_core_X(X, kernel, off_diagonal_cutoff=0.9, y=None, core_max=500, method='random'):
        C_idx = get_core_idx(X, kernel, off_diagonal_cutoff=off_diagonal_cutoff, core_max=core_max, method=method)
        print('%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), X.shape[0]))
        X = X[X.index.isin(C_idx)] if X.__class__ == pd.DataFrame else X[C_idx]
        if y is not None:
            return X, y[C_idx]
        else:
            return X

    def y_normalise(self, y):
        # Normalize target value
        if self.normalize_y:
            self._y_train_mean_full = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean_full
        else:
            self._y_train_mean_full = np.zeros(1)
        return y


class NystromGaussianProcessRegressor(NystromPreGaussianProcessRegressor):
    def __init__(self, core_predict=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_predict = core_predict

    def fit_robust(self, X, y, Xc=None, yc=None):
        print('Start a new fit process')
        if Xc is not None and yc is not None:
            X_ = Xc
            y_ = yc
        elif hasattr(self, 'kernel_'):
            X_, y_ = self.get_core_X(X, self.kernel_, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        else:
            X_, y_ = self.get_core_X(X, self.kernel, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        if super().fit_robust(X_, y_, core_predict=self.core_predict) is None:
            return None
        if self.optimizer is not None:
            X_, y_ = self.get_core_X(X, self.kernel_, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        y = self.y_normalise(y)
        self.core_X = np.copy(X_)
        self.core_y = np.copy(y_)
        self.full_X = np.copy(X) if self.copy_X_train else X
        self.full_y = np.copy(y) if self.copy_X_train else y
        print('hyperparameter: ', self.kernel_.hyperparameters, '\n')
        Kcc = self.kernel_(X_)
        Kcx = self.kernel_(X_, X)
        Kccinv, Kxx_ihalf = self.Nystrom_solve(Kcc, Kcx, eigen_cutoff=Config.NystromPara.alpha, debug=Config.DEBUG)
        self.left = Kccinv.dot(Kcx.dot(Kxx_ihalf))  # c*c
        self.right = Kxx_ihalf.T.dot(y)  # c*o
        return self

    def core_predict(self, X, return_std=False, return_cov=False):
        if not self.core_predict:
            raise Exception('core_prediction can only used by set core_predict=True in fit_robust()')
        return super().predict(X, return_std, return_cov)

    def predict(self, X, return_std=False, return_cov=False):
        if return_cov:
            return super().nystrom_predict(X, return_std=return_std, return_cov=return_cov)
        else:
            if X.__class__ != np.ndarray:
                X = X.to_numpy()
            N = X.shape[0]
            y_mean = np.array([])
            y_std = np.array([])
            for i in range(math.ceil(N / 1000)):
                X_ = X[i * 1000:(i + 1) * 1000]
                if return_std:
                    y_mean_, y_std_ = super().nystrom_predict(X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = super().nystrom_predict(X_, return_std=return_std, return_cov=return_cov)
                y_mean = np.r_[y_mean, y_mean_]
            if return_std:
                return y_mean, y_std
            else:
                return y_mean
'''