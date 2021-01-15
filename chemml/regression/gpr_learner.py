from chemml.base_learner import KernelRegressionBaseLearner
from chemml.regression.consensus import ConsensusRegressor
from chemml.regression.GPRgraphdot.gpr import GPR as GPRgraphdot
from chemml.regression.GPRgraphdot.gpr import LRAGPR
from chemml.regression.GPRsklearn.gpr import GPR as GPRsklearn
import numpy as np
import sys


class GPRLearner(KernelRegressionBaseLearner):
    def train(self, train_X=None, train_y=None, train_id=None):
        if train_X is None:
            train_X = self.train_X
        if train_y is None:
            train_y = self.train_Y
        if train_id is None:
            train_id = self.train_id
        if self.model.__class__ == ConsensusRegressor:
            self.model.fit(train_X, train_y, train_id)
        elif self.model.__class__ == GPRgraphdot:
            self.model.fit(train_X, train_y, loss='loocv',
                           verbose=True, repeat=1)
            self.model.X_id_ = train_id
            print('hyperparameter: ', self.model.kernel_.hyperparameters)
        elif self.model.__class__ == LRAGPR:
            idx = np.random.choice(
                len(train_X), self.n_nystrom_core, replace=True)
            C = train_X[idx]
            self.model.fit(C, train_X, train_y, loss='loocv',
                           verbose=True, repeat=1)
            self.model.X_id_ = train_id
            self.model.C_id_ = train_id[idx]
            print('hyperparameter: ', self.model.kernel_.hyperparameters)
        elif self.model.__class__ == GPRsklearn:
            self.model.fit(train_X, train_y)
            self.model.X_id_ = train_id
            print('hyperparameter: ', self.model.kernel_.hyperparameters)
        else:
            raise RuntimeError(f'Unknown regressor {self.model}')

    def evaluate_df__(self, *args, **kwargs):
        y_std = kwargs.pop('y_std', None)
        alpha = kwargs.pop('alpha', None)
        df_out, r2, ex_var, mae, rmse, mse = self.evaluate_df_(*args, **kwargs)
        if y_std is not None:
            df_out.loc[:, 'uncertainty'] = y_std
        if alpha is not None:
            df_out.loc[:, 'alpha'] = alpha
        return df_out, r2, ex_var, mae, rmse, mse

    def evaluate_test(self, alpha=None):
        x = self.test_X
        y = self.test_Y
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df__(x, y, y_pred, self.test_id, y_std=y_std,
                                  alpha=alpha, n_most_similar=5)

    def evaluate_train(self, alpha=None):
        x = self.train_X
        y = self.train_Y
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df__(x, y, y_pred, self.train_id, y_std=y_std,
                                  alpha=alpha)

    def evaluate_loocv(self, alpha=None):
        x = self.train_X
        y = self.train_Y
        y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        return self.evaluate_df__(x, y, y_pred, self.train_id, y_std=y_std,
                                  alpha=alpha, n_most_similar=5)

    def evaluate_test_dynamic(self, dynamic_train_size=500):
        assert (self.model_.optimizer is None)
        K = self.model_.kernel_(self.test_X, self.train_X)
        kindex = self.get_most_similar_graphs(K, n=dynamic_train_size)
        y_pred = []
        y_std = []
        for i in range(len(self.test_X)):
            sys.stdout.write('\r %i / %i' % (i, len(self.test_X)))
            tx = self.test_X[i:i+1]
            self.train(train_X=self.train_X[kindex[i]],
                       train_y=self.train_Y[kindex[i]])
            y_pred_, y_std_ = self.model.predict(tx, return_std=True)
            y_pred.append(y_pred_)
            y_std.append(y_std_)
        return self.evaluate_df__(
            self.test_X, self.test_Y, np.concatenate(y_pred), self.test_id,
            np.concatenate(y_std), K=K)
