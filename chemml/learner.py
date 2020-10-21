import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class BaseLearner:
    def __init__(self, train_X, train_Y, train_smiles, test_X, test_Y,
                 test_smiles, kernel_config, optimizer=None, alpha=0.01):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_smiles = train_smiles
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_smiles = test_smiles
        self.kernel_config = kernel_config
        self.kernel = kernel_config.kernel
        self.optimizer = optimizer
        self.alpha = alpha

    def get_out(self, x, smiles):
        out = pd.DataFrame({'smiles': smiles})
        if self.kernel_config.features is not None:
            for i, feature in enumerate(self.kernel_config.features):
                column_id = -len(self.kernel_config.features)+i
                out.loc[:, feature] = x[:, column_id]
        return out

    def evaluate_df(self, x, y, smiles, y_pred, y_std, kernel=None,
                    debug=False, alpha=None):
        r2 = r2_score(y, y_pred)
        ex_var = explained_variance_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        if len(y.shape) == 1:
            out = pd.DataFrame({'#target': y,
                                'predict': y_pred,
                                'uncertainty': y_std,
                                'abs_dev': abs(y - y_pred),
                                'rel_dev': abs((y - y_pred) / y)})
        else:
            out = pd.DataFrame({})
            for i in range(y.shape[1]):
                out['c%i' % i] = y[:, i]
                out['c%i_pred' % i] = y_pred[:, i]
            out['uncertainty'] = y_std
            out['abs_dev'] = abs(y - y_pred).mean(axis=1)
            out['rel_dev'] = abs((y - y_pred) / y).mean(axis=1)
        if alpha is not None:
            out.loc[:, 'alpha'] = alpha
        out = pd.concat([out, self.get_out(x, smiles)], axis=1)
        if debug:
            K = kernel(x, self.train_X)
            xout = self.get_out(self.train_X, self.train_smiles)
            info_list = []
            kindex = np.argsort(-K)[:, :min(5, len(self.train_X))]
            for i, smiles in enumerate(self.train_smiles):
                s = [smiles]
                if self.kernel_config.features is not None:
                    for feature in self.kernel_config.features:
                        s.append(xout[feature][i])
                s = list(map(str, s))
                info_list.append(','.join(s))
            info_list = np.array(info_list)
            similar_data = []
            for i, index in enumerate(kindex):
                info = info_list[index]

                def round5(x):
                    return ',%.5f' % x

                k = list(map(round5, K[i][index]))
                info = ';'.join(list(map(str.__add__, info, k)))
                similar_data.append(info)
            out.loc[:, 'similar_mols'] = similar_data
        return r2, ex_var, mse, out.sort_values(by='abs_dev', ascending=False)

    def evaluate_test(self, debug=True, alpha=None):
        x = self.test_X
        y = self.test_Y
        smiles = self.test_smiles
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)

    def evaluate_train(self, debug=False, alpha=None):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)

    def evaluate_loocv(self, debug=True, alpha=None):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)


class ActiveLearner:
    ''' for active learning, basically do selection for users '''

    def __init__(self, train_X, train_Y, train_smiles, alpha, kernel_config,
                 learning_mode, add_mode, initial_size, add_size, max_size,
                 search_size, pool_size, result_dir, Learner, test_X=None,
                 test_Y=None, test_smiles=None, optimizer=None, stride=100,
                 seed=0):
        '''
        search_size: Random chose samples from untrained samples. And are
                     predicted based on current model.
        pool_size: The largest mse or std samples in search_size.
        nystrom_active: If True, using train_X as training set and active
                        learning the K_core of corresponding nystrom
                        approximation.
        nystrom_predict: If True, no nystrom approximation is used in the active
                         learning process. But output the nystrom prediction
                         using train_X as X and train_x as C.
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_smiles = train_smiles
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_smiles = test_smiles
        self.kernel_config = kernel_config
        self.alpha = alpha
        self.learning_mode = learning_mode
        self.add_mode = add_mode
        if initial_size <= 1:
            raise Exception('initial_size must be larger than 1.')
        self.current_size = initial_size
        self.add_size = add_size
        self.max_size = max_size
        self.search_size = search_size
        self.pool_size = pool_size
        self.optimizer = optimizer
        self.seed = seed
        self.result_dir = result_dir
        self.Learner = Learner
        self.train_IDX = np.linspace(
            0,
            len(train_X) - 1,
            len(train_X),
            dtype=int
        )
        np.random.seed(seed)
        self.train_idx = np.random.choice(
            self.train_IDX,
            initial_size,
            replace=False
        )
        self.stride = stride
        self.learning_log = pd.DataFrame({
            '#size': [], 'r2': [], 'mse': [], 'ex-var': [], 'search_size': []
        })

    def stop_sign(self):
        if self.current_size >= self.max_size \
                or self.current_size == len(self.train_X):
            return True
        else:
            return False

    def __get_train_X_y(self):
        train_x = self.train_X[self.train_idx]
        train_y = self.train_Y[self.train_idx]
        smiles = self.train_smiles[self.train_idx]
        return train_x, train_y, smiles

    def __get_untrain_X_y(self):
        untrain_idx = np.delete(self.train_IDX, self.train_idx)
        if self.search_size != 0 and self.search_size < len(untrain_idx):
            untrain_idx = np.random.choice(
                untrain_idx,
                self.search_size,
                replace=False
            )
        untrain_x = self.train_X[untrain_idx]
        untrain_y = self.train_Y[untrain_idx]
        return untrain_x, untrain_y, untrain_idx

    def train(self):
        np.random.seed(self.seed)
        # print('%s' % (time.asctime(time.localtime(time.time()))))
        train_x, train_y, smiles = self.__get_train_X_y()
        self.learner = self.Learner(
            train_x,
            train_y,
            smiles,
            self.test_X,
            self.test_Y,
            self.test_smiles,
            self.kernel_config,
            optimizer=self.optimizer,
            alpha=self.alpha,
        )
        self.learner.train(verbose=False)
        return True

    @staticmethod
    def __to_df(x):
        if x.__class__ == pd.Series:
            return pd.DataFrame({x.name: x})
        elif x.__class__ == np.ndarray:
            return pd.DataFrame({'graph': x})
        else:
            return x

    def add_samples(self):
        # print('%s' % (time.asctime(time.localtime(time.time()))))
        import warnings
        warnings.filterwarnings("ignore")
        untrain_x, untrain_y, untrain_idx = self.__get_untrain_X_y()
        if self.learning_mode == 'supervised':
            y_pred = self.learner.model.predict(untrain_x)
            y_abse = abs(y_pred - untrain_y)
            add_idx = self._get_samples_idx(untrain_x, y_abse, untrain_idx)
            self.train_idx = np.r_[self.train_idx, add_idx]
            self.current_size = self.train_idx.size
        elif self.learning_mode == 'unsupervised':
            y_pred, y_std = self.learner.model.predict(untrain_x, return_std=True)
            add_idx = self._get_samples_idx(untrain_x, y_std, untrain_idx)
            self.train_idx = np.r_[self.train_idx, add_idx]
            self.current_size = self.train_idx.size
        elif self.learning_mode == 'random':
            np.random.seed(self.seed)
            if untrain_idx.shape[0] < self.add_size:
                add_idx = untrain_idx
                self.train_idx = np.r_[self.train_idx, add_idx]
            else:
                add_idx = np.random.choice(
                    untrain_idx,
                    self.add_size,
                    replace=False
                )
                self.train_idx = np.r_[self.train_idx, add_idx]
            self.current_size = self.train_idx.size
        else:
            raise ValueError(
                "unrecognized method. Could only be one of ('supervised',"
                "'unsupervised','random').")

    def _get_samples_idx(self, x, error, idx):
        ''' get a sample idx list from the pooling set using add mode method
        :df: dataframe constructed
        :target: should be one of mse/std
        :return: list of idx
        '''
        # print('%s' % (time.asctime(time.localtime(time.time()))))
        if len(x) < self.add_size:  # add all if end of the training set
            return idx
        if self.add_mode == 'random':
            return np.random.choice(idx, self.add_size, replace=False)
        elif self.add_mode == 'cluster':
            search_idx = self.__get_search_idx(
                error,
                idx,
                pool_size=self.pool_size
            )
            search_K = self.learner.model.kernel(self.train_X[search_idx])
            add_idx = self._find_add_idx_cluster(search_K)
            return np.array(search_idx)[add_idx]
        elif self.add_mode == 'nlargest':
            return self.__get_search_idx(error, idx, pool_size=self.add_size)
        else:
            raise ValueError(
                "unrecognized method. Could only be one of ('random','cluster','nlargest', 'threshold).")

    def __get_search_idx(self, error, idx, pool_size=0):
        if pool_size == 0 or len(error) < pool_size:
            # from all remaining samples
            return idx
        else:
            return idx[np.argsort(error)[-pool_size:]]

    def _find_add_idx_cluster(self, gram_matrix):
        ''' find representative samp-les from a pool using clustering method
        :gram_matrix: gram matrix of the pool samples
        :return: list of idx
        '''
        embedding = SpectralEmbedding(
            n_components=self.add_size,
            affinity='precomputed'
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=self.add_size,
            random_state=0
        ).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(self.add_size)])
        total_distance = defaultdict(
            dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(
                embedding[i] - np.delete(center, cluster_class, axis=0))).sum(
                axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(
                       self.add_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def evaluate(self, train_output=True, debug=True):
        # print('%s' % (time.asctime(time.localtime(time.time()))))
        r2, ex_var, mse, out = self.learner.evaluate_test(debug=debug)
        print("R-square:%.3f\nMSE:%.3g\nexplained_variance:%.3f\n" %
              (r2, mse, ex_var))
        self.learning_log.loc[self.current_size] = (
            self.current_size, r2, mse,
            ex_var,
            self.search_size
        )
        out.to_csv(
            '%s/%i.log' % (self.result_dir, self.current_size),
            sep='\t',
            index=False,
            float_format='%15.10f'
        )

        if train_output:
            r2, ex_var, mse, out = self.learner.evaluate_train(debug=debug)
            out.to_csv(
                '%s/%i-train.log' % (self.result_dir, self.current_size),
                sep='\t',
                index=False,
                float_format='%15.10f'
            )

    def write_training_plot(self):
        self.learning_log.reset_index().drop(columns='index').to_csv(
            '%s/%s-%s-%d.out' % (
                self.result_dir,
                self.learning_mode,
                self.add_mode,
                self.add_size
            ),
            sep=' ',
            index=False
        )

    def save_checkpoint(self):
        f_checkpoint = os.path.join(self.result_dir, 'checkpoint.pkl')
        # store all attributes instead of model
        store_dict = self.__dict__.copy()
        store_dict.pop('learner', None)
        store_dict.pop('kernel_config', None)
        # store model
        self.learner.model.save(self.result_dir)
        pickle.dump(store_dict, open(f_checkpoint, 'wb'), protocol=4)

    @classmethod
    def load_checkpoint(cls, f_checkpoint, kernel_config):
        d = pickle.load(open(f_checkpoint, 'rb'))
        activelearner = cls(
            d['train_X'], d['train_Y'], d['train_smiles'], d['alpha'],
            kernel_config, d['learning_mode'], d['add_mode'], 10,
            d['add_size'], d['max_size'], d['search_size'], d['pool_size'],
            d['result_dir'], d['Learner'], d['test_X'], d['test_Y'],
            d['test_smiles'], d['optimizer'], d['stride'], d['seed'])
        # restore params
        for key in d.keys():
            setattr(activelearner, key, d[key])
        return activelearner

    def __str__(self):
        return 'parameter of current active learning checkpoint:\n' + \
               'current_size:%s  max_size:%s  learning_mode:%s  add_mode:%s  search_size:%d  pool_size:%d  add_size:%d\n' % (
                   self.current_size, self.max_size, self.learning_mode,
                   self.add_mode, self.search_size,
                   self.pool_size,
                   self.add_size)
