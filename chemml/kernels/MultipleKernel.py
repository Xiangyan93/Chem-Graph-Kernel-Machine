import copy
import numpy as np


class MultipleKernel:
    def __init__(self, kernel_list, composition, combined_rule='product'):
        self.kernel_list = kernel_list
        self.composition = composition
        self.combined_rule = combined_rule

    @property
    def nkernel(self):
        return len(self.kernel_list)

    def get_X_list(self, X):
        def f(c):
            return X[:, c]
        X_list = list(map(f, self.composition))
        return X_list

    def __call__(self, X, Y=None, eval_gradient=False):
        X_list = self.get_X_list(X)
        Y_list = self.get_X_list(Y) if Y is not None else None
        if eval_gradient:
            covariance_matrix = 1
            gradient_matrix_list = list(
                map(int, np.ones(self.nkernel).tolist()))
            for i, kernel in enumerate(self.kernel_list):
                Xi = X_list[i]
                Yi = Y_list[i] if Y is not None else None
                output = kernel(Xi, Y=Yi, eval_gradient=True)
                if self.combined_rule == 'product':
                    covariance_matrix *= output[0]
                    for j in range(self.nkernel):
                        if j == i:
                            gradient_matrix_list[j] = gradient_matrix_list[j] * \
                                                      output[1]
                        else:
                            shape = output[0].shape + (1,)
                            gradient_matrix_list[j] = gradient_matrix_list[j] * \
                                                      output[0].reshape(shape)
            gradient_matrix = gradient_matrix_list[0]
            for i, gm in enumerate(gradient_matrix_list):
                if i != 0:
                    gradient_matrix = np.c_[
                        gradient_matrix, gradient_matrix_list[i]]
            return covariance_matrix, gradient_matrix
        else:
            covariance_matrix = 1
            for i, kernel in enumerate(self.kernel_list):
                Xi = X_list[i]
                Yi = Y_list[i] if Y is not None else None
                output = kernel(Xi, Y=Yi, eval_gradient=False)
                if self.combined_rule == 'product':
                    covariance_matrix *= output
            return covariance_matrix

    def diag(self, X):
        X_list = self.get_X_list(X)
        diag_list = [self.kernel_list[i].diag(X_list[i]) for i in
                     range(len(self.kernel_list))]
        if self.combined_rule == 'product':
            return np.product(diag_list, axis=0)
        else:
            raise Exception('Unknown combined rule %s' % self.combined_rule)

    def is_stationary(self):
        return False

    @property
    def requires_vector_input(self):
        return False

    @property
    def n_dims_list(self):
        return [kernel.n_dims for kernel in self.kernel_list]

    @property
    def n_dims(self):
        return sum(self.n_dims_list)

    @property
    def hyperparameters(self):
        return np.exp(self.theta)

    @property
    def theta(self):
        for i, kernel in enumerate(self.kernel_list):
            if i == 0:
                theta = self.kernel_list[0].theta
            else:
                theta = np.r_[theta, kernel.theta]
        return theta

    @theta.setter
    def theta(self, value):
        if len(value) != self.n_dims:
            raise Exception('The length of n_dims and theta must the same')
        s = 0
        e = 0
        for i, kernel in enumerate(self.kernel_list):
            e += self.n_dims_list[i]
            kernel.theta = value[s:e]
            s += self.n_dims_list[i]

    @property
    def bounds(self):
        for i, kernel in enumerate(self.kernel_list):
            if i == 0:
                bounds = self.kernel_list[0].bounds
            else:
                bounds = np.r_[bounds, kernel.bounds]
        return bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            kernel_list=self.kernel_list,
            composition=self.composition,
            combined_rule=self.combined_rule,
        )
