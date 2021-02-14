import math
import numpy as np
import pandas as pd
from chemml.regression.consensus import ConsensusRegressor
from chemml.regression.GPRgraphdot.gpr import LRAGPR
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
        self.n_nystrom_core = kwargs.pop('n_nystrom_core', None) or 0
        super().__init__(*args, **kwargs)
        if self.model_.__class__ == LRAGPR:
            assert (not consensus)
        if consensus and n_estimators != 1:
            self.model = ConsensusRegressor(
                self.model_,
                n_estimators=n_estimators,
                n_sample_per_model=n_sample_per_model,
                n_jobs=n_jobs,
                consensus_rule=consensus_rule
            )
        elif n_estimators == 1:
            self.model = self.model_
            idx = np.random.choice(
                len(self.train_X), n_sample_per_model, replace=True)
            self.train_X = self.train_X[idx]
            self.train_Y = self.train_Y[idx]
            self.train_id = self.train_id[idx]
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.model_

    @staticmethod
    def evaluate_df(y, y_pred, id):
        print(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        print(accuracy)
        precision = precision_score(y, y_pred, average='micro')
        recall = recall_score(y, y_pred, average='micro')
        f1 = f1_score(y, y_pred, average='micro')
        df_out = pd.DataFrame({
            '#target': y,
            'predict': y_pred,
            'id': id
        })
        return df_out, accuracy, precision, recall, f1


class KernelRegressionBaseLearner(RegressionBaseLearner):
    def evaluate_df_(self, X, y, y_pred, id, n_most_similar=None,
                     memory_save=True, n_memory_save=1000):
        df_out, r2, ex_var, mae, rmse, mse = self.evaluate_df(y, y_pred, id)
        if n_most_similar is not None:
            if memory_save:
                similar_info = []
                N = X.shape[0]
                for i in range(math.ceil(N / n_memory_save)):
                    X_ = X[i * n_memory_save:(i + 1) * n_memory_save]
                    similar_info.extend(
                        self.get_similar_info(self, X_, n_most_similar))
            else:
                similar_info = self.get_similar_info(self, X, n_most_similar)

            df_out.loc[:, 'similar_mols'] = similar_info
        return df_out, r2, ex_var, mae, rmse, mse

    @staticmethod
    def get_similar_info(self, X, n_most_similar):
        K = self.model_.kernel_(X, self.train_X)
        assert (K.shape == (len(X), len(self.train_X)))
        similar_info = []
        kindex = KernelRegressionBaseLearner.get_most_similar_graphs(
            K, n=n_most_similar)
        for i, index in enumerate(kindex):
            def round5(x):
                return ',%.5f' % x

            k = list(map(round5, K[i][index]))
            id = list(map(str, self.train_id[index].tolist()))
            info = ';'.join(list(map(str.__add__, id, k)))
            similar_info.append(info)
        return similar_info

    @staticmethod
    def get_most_similar_graphs(K, n=5):
        return np.argsort(-K)[:, :min(n, K.shape[1])]


class KernelClassificationBaseLearner(ClassificationBaseLearner):
    def evaluate_df_(self, X, y, y_pred, id, n_most_similar=None,
                     memory_save=True, n_memory_save=1000):
        df_out, accuracy, precision, recall, f1 = self.evaluate_df(y, y_pred, id)
        if n_most_similar is not None:
            if memory_save:
                similar_info = []
                N = X.shape[0]
                for i in range(math.ceil(N / n_memory_save)):
                    X_ = X[i * n_memory_save:(i + 1) * n_memory_save]
                    similar_info.extend(
                        KernelRegressionBaseLearner.get_similar_info(
                            self, X_, n_most_similar))
            else:
                similar_info = KernelRegressionBaseLearner.get_similar_info(
                    self, X, n_most_similar)

            df_out.loc[:, 'similar_mols'] = similar_info
        return df_out, accuracy, precision, recall, f1
