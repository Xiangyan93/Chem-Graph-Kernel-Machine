import numpy as np
import pandas as pd
from chemml.regression.consensus import ConsensusRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class BaseLearner:
    def __init__(self, model, train_X, train_Y, train_id, test_X, test_Y,
                 test_id):
        assert (self.__class__ != BaseLearner)
        self.model_ = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_id = train_id
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_id = test_id


class RegressionBaseLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        consensus = kwargs.pop('consensus', None) or False
        n_estimators = kwargs.pop('n_estimators', None) or 100
        n_sample_per_model = kwargs.pop('n_sample_per_model', None) or 2000
        n_jobs = kwargs.pop('n_jobs', None) or 1
        consensus_rule = kwargs.pop('consensus_rule', None) or \
                         'smallest_uncertainty'
        super().__init__(*args, **kwargs)
        if consensus:
            self.model = ConsensusRegressor(
                self.model_,
                n_estimators=n_estimators,
                n_sample_per_model=n_sample_per_model,
                n_jobs=n_jobs,
                consensus_rule=consensus_rule
            )
        else:
            self.model = self.model_

    @staticmethod
    def evaluate_df(y, y_pred, id):
        r2 = r2_score(y, y_pred, multioutput='raw_values')
        ex_var = explained_variance_score(y, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
        df_out = pd.DataFrame({
            '#target': y,
            'predict': y_pred,
            'abs_dev': abs(y - y_pred),
            'id': id
            # 'rel_dev': abs((y - y_pred) / y)
        })
        return df_out, r2, ex_var, mae, rmse, mse


class ClassificationBaseLearner(BaseLearner):
    @staticmethod
    def evaluate_df(y, y_pred, id):
        accuracy = (np.asarray(y) == np.asarray(y_pred)).sum() / len(y)
        df_out = pd.DataFrame({
            '#target': y,
            'predict': y_pred,
            'id': id
        })
        return df_out, accuracy


class KernelRegressionBaseLearner(RegressionBaseLearner):
    def evaluate_df_(self, x, y, y_pred, id, K=None, n_most_similar=None):
        df_out, r2, ex_var, mae, rmse, mse = self.evaluate_df(y, y_pred, id)
        if n_most_similar is not None:
            if K is None:
                K = self.model_.kernel_(x, self.train_X)
            assert (K.shape == (len(x), len(self.train_X)))
            similar_info = []
            kindex = self.get_most_similar_graphs(K, n=n_most_similar)
            for i, index in enumerate(kindex):
                def round5(x):
                    return ',%.5f' % x
                k = list(map(round5, K[i][index]))
                id = list(map(str, self.train_id[index].tolist()))
                info = ';'.join(list(map(str.__add__, id, k)))
                similar_info.append(info)
            df_out.loc[:, 'similar_mols'] = similar_info
        return df_out, r2, ex_var, mae, rmse, mse

    @staticmethod
    def get_most_similar_graphs(K, n=5):
        return np.argsort(-K)[:, :min(n, K.shape[1])]
