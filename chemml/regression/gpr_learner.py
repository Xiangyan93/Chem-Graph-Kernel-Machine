from chemml.baselearner import BaseLearner
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class GPRLearner(BaseLearner):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs['alpha']
        kwargs.pop('alpha')
        super().__init__(*args, **kwargs)

    def evaluate_df(self, x, y, id, y_pred, y_std, alpha=None, K=None,
                    n_most_similar=None):
        r2 = r2_score(y, y_pred, multioutput='raw_values')
        ex_var = explained_variance_score(y, y_pred, multioutput='raw_values')
        mse = mean_squared_error(y, y_pred, multioutput='raw_values')
        mae = mean_absolute_error(y, y_pred, multioutput='raw_values')
        out = pd.DataFrame({
            '#target': y,
            'predict': y_pred,
            'uncertainty': y_std,
            'abs_dev': abs(y - y_pred),
            # 'rel_dev': abs((y - y_pred) / y)
        })
        if alpha is not None:
            out.loc[:, 'alpha'] = alpha
        out.loc[:, 'id'] = id

        if n_most_similar is not None:
            if K is None:
                K = self.model.kernel_(x, self.train_X)
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
            out.loc[:, 'similar_mols'] = similar_info
        return out.sort_values(by='abs_dev', ascending=False), \
               r2, ex_var, mse, mae

    def evaluate_test(self, alpha=None):
        x = self.test_X
        y = self.test_Y
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df(x, y, self.test_id, y_pred, y_std,
                                alpha=alpha, n_most_similar=5)

    def evaluate_train(self, alpha=None):
        x = self.train_X
        y = self.train_Y
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df(x, y, self.train_id, y_pred, y_std,
                                alpha=alpha)

    def evaluate_loocv(self, alpha=None):
        x = self.train_X
        y = self.train_Y
        y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        return self.evaluate_df(x, y, self.train_id, y_pred, y_std,
                                alpha=alpha, n_most_similar=5)

    def evaluate_test_dynamic(self, dynamic_train_size=500):
        assert (self.optimizer is None)
        K = self.kernel(self.test_X, self.train_X)
        kindex = self.get_most_similar_graphs(K, n=dynamic_train_size)
        y_pred = []
        y_std = []
        for i in range(len(self.test_X)):
            sys.stdout.write('\r %i / %i' % (i, len(self.test_X)))
            tx = self.test_X[i:i+1]
            self.train(train_X=self.train_X[kindex[i]],
                       train_Y=self.train_Y[kindex[i]])
            y_pred_, y_std_ = self.model.predict(tx, return_std=True)
            y_pred.append(y_pred_)
            y_std.append(y_std_)
        return self.evaluate_df(self.test_X, self.test_Y, self.test_id,
                                np.concatenate(y_pred), np.concatenate(y_std),
                                K=K)
