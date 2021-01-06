from chemml.base_learner import KernelLearner
import pandas as pd
import numpy as np


class GPCLearner(KernelLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_df(self, x, y, id, y_pred, y_proba, K=None,
                    n_most_similar=None):
        correct_ratio = (y == y_pred).sum() / len(y)
        out = pd.DataFrame({
            '#target': y,
            'predict': y_pred,
            'predict_proba_1': y_proba[:, 0],
            'predict_proba_2': y_proba[:, 1],
            'confidence': np.abs(y_proba[:, 0] - 0.5)
        })
        out.loc[:, 'id'] = id

        if n_most_similar is not None:
            if K is None:
                K = self.kernel_(x, self.train_X)
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
        return out.sort_values(by='confidence', ascending=True), correct_ratio

    def evaluate_test(self):
        x = self.test_X
        y = self.test_Y
        y_pred = self.model.predict(x)
        y_proba = self.model.predict_proba(x)
        return self.evaluate_df(x, y, self.test_id, y_pred, y_proba,
                                n_most_similar=5)

    def evaluate_train(self):
        x = self.train_X
        y = self.train_Y
        y_pred = self.model.predict(x)
        y_proba = self.model.predict_proba(x)
        return self.evaluate_df(x, y, self.train_id, y_pred, y_proba)

