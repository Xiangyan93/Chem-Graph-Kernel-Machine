#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.gaussian_process._gpc import (
    GaussianProcessClassifier as _GPC,
    _BinaryGaussianProcessClassifierLaplace,
    OneVsOneClassifier,
    OneVsRestClassifier as OVR,
    C, RBF, check_random_state, LabelEncoder, itemgetter
)
from sklearn.multiclass import (
    LabelBinarizer, warnings, _ConstantPredictor, Parallel, delayed
)
from sklearn.utils.validation import _deprecate_positional_args
import numpy as np
import copy


@_deprecate_positional_args
def clone(estimator, *, safe=True):
    """Constructs a new unfitted estimator with the same parameters.

    Clone does a deep copy of the model in an estimator
    without actually copying attached data. It yields a new estimator
    with the same parameters that has not been fitted on any data.

    If the estimator's `random_state` parameter is an integer (or if the
    estimator doesn't have a `random_state` parameter), an *exact clone* is
    returned: the clone and the original estimator will give the exact same
    results. Otherwise, *statistical clone* is returned: the clone might
    yield different results from the original estimator. More details can be
    found in :ref:`randomness`.

    Parameters
    ----------
    estimator : {list, tuple, set} of estimator instance or a single \
            estimator instance
        The estimator or group of estimators to be cloned.

    safe : bool, default=True
        If safe is False, clone will fall back to a deep copy on objects
        that are not estimators.

    """
    estimator_type = type(estimator)
    # XXX: not handling dictionaries
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            if isinstance(estimator, type):
                raise TypeError("Cannot clone object. " +
                                "You should provide an instance of " +
                                "scikit-learn estimator instead of a class.")
            else:
                raise TypeError("Cannot clone object '%s' (type %s): "
                                "it does not seem to be a scikit-learn "
                                "estimator as it does not implement a "
                                "'get_params' method."
                                % (repr(estimator), type(estimator)))

    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)
    """
    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor '
                               'either does not set or modifies parameter %s' %
                               (estimator, name))
    """
    return new_object


def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)
    return estimator


class OneVsRestClassifier(OVR):
    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

        return self


class BinaryGaussianProcessClassifierLaplace(_BinaryGaussianProcessClassifierLaplace):
    def fit(self, X, y):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self.rng = check_random_state(self.random_state)

        self.X_train_ = np.copy(X) if self.copy_X_train else X

        # Encode class labels and check that it is a binary classification
        # problem
        label_encoder = LabelEncoder()
        self.y_train_ = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        if self.classes_.size > 2:
            raise ValueError("%s supports only binary classification. "
                             "y contains classes %s"
                             % (self.__class__.__name__, self.classes_))
        elif self.classes_.size == 1:
            raise ValueError("{0:s} requires 2 classes; got {1:d} class"
                             .format(self.__class__.__name__,
                                     self.classes_.size))

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta,
                                                         clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [self._constrained_optimization(obj_func,
                                                     self.kernel_.theta,
                                                     self.kernel_.bounds)]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0],
                                                            bounds[:, 1]))
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)

        _, (self.pi_, self.W_sr_, self.L_, _, _) = \
            self._posterior_mode(K, return_temporaries=True)

        return self


class GPC(_GPC):
    def fit(self, X, y):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None or self.kernel.requires_vector_input:
            X, y = self._validate_data(X, y, multi_output=False,
                                       ensure_2d=True, dtype="numeric")
        else:
            X, y = self._validate_data(X, y, multi_output=False,
                                       ensure_2d=False, dtype=None)

        self.base_estimator_ = BinaryGaussianProcessClassifierLaplace(
            kernel=self.kernel,
            optimizer=self.optimizer,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            copy_X_train=self.copy_X_train,
            random_state=self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("GaussianProcessClassifier requires 2 or more "
                             "distinct classes; got %d class (only class %s "
                             "is present)"
                             % (self.n_classes_, self.classes_[0]))
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = \
                    OneVsRestClassifier(self.base_estimator_,
                                        n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = \
                    OneVsOneClassifier(self.base_estimator_,
                                       n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [estimator.log_marginal_likelihood()
                 for estimator in self.base_estimator_.estimators_])
        else:
            self.log_marginal_likelihood_value_ = \
                self.base_estimator_.log_marginal_likelihood()

        return self


class GaussianProcessClassifier(GPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._GPC = GPC(*args, **kwargs)

    @staticmethod
    def _remove_nan_X_y(X, y):
        if None in y:
            idx = np.where(y!=None)[0]
        elif y.dtype == float:
            idx = ~np.isnan(y)
        else:
            idx = np.arange(len(y))
        return np.asarray(X)[idx], y[idx]  #.astype(int)

    def fit(self, X, y):
        self.GPCs = []
        if y.ndim == 1:
            X_, y_ = self._remove_nan_X_y(X, y)
            super().fit(X_, y_)
        else:
            for i in range(y.shape[1]):
                GPC = copy.deepcopy(self._GPC)
                X_, y_ = self._remove_nan_X_y(X, y[:, i])
                GPC.fit(X_, y_)
                self.GPCs.append(GPC)

    def predict(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                y_mean.append(GPC.predict(X))
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict(X)

    def predict_proba(self, X):
        if self.GPCs:
            y_mean = []
            for GPC in self.GPCs:
                y_mean.append(GPC.predict_proba(X)[:, 1])
            return np.concatenate(y_mean).reshape(len(y_mean), len(X)).T
        else:
            return super().predict_proba(X)[:, 1]