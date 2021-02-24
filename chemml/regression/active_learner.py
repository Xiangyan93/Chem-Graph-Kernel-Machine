#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding


class ActiveLearner:
    ''' for active learning, basically do selection for users '''

    def __init__(self, train_X, train_Y, train_id, alpha, kernel_config,
                 learning_mode, add_mode, initial_size, add_size, max_size,
                 search_size, pool_size, result_dir, Learner, model,
                 test_X=None, test_Y=None, test_id=None, optimizer=None,
                 stride=100, seed=0):
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
        self.train_id = train_id
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_id = test_id
        self.kernel_config = kernel_config
        if np.iterable(alpha):
            self.init_alpha = alpha
        else:
            self.init_alpha = np.ones(len(train_X)) * alpha
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
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.Learner = Learner
        self.model = model
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
            '#size': [], 'r2': [], 'mse': [], 'mae': [], 'ex-var': [],
            'search_size': []
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
        id = self.train_id[self.train_idx]
        alpha = self.init_alpha[self.train_idx]
        return train_x, train_y, id, alpha

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
        train_x, train_y, id, alpha = self.__get_train_X_y()
        self.learner = self.Learner(
            self.model,
            train_x,
            train_y,
            id,
            self.test_X,
            self.test_Y,
            self.test_id,
        )
        self.learner.train()
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

    def evaluate(self, train_output=True):
        # print('%s' % (time.asctime(time.localtime(time.time()))))
        out, r2, ex_var, mae, rmse, mse = self.learner.evaluate_test()
        print("R-square:%.3f\nMSE:%.5g\nexplained_variance:%.3f\n" %
              (r2, mse, ex_var))
        self.learning_log.loc[self.current_size] = (
            self.current_size, r2, mse, mae,
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
            out, r2, ex_var, mae, rmse, mse = self.learner.evaluate_train()
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
        self.learner.model.save(self.result_dir, overwrite=True)
        # self.learner.kernel_config.save(self.result_dir, self.learner.model)
        pickle.dump(store_dict, open(f_checkpoint, 'wb'), protocol=4)

    @classmethod
    def load_checkpoint(cls, f_checkpoint, kernel_config):
        d = pickle.load(open(f_checkpoint, 'rb'))
        activelearner = cls(
            d['train_X'], d['train_Y'], d['train_id'], d['init_alpha'],
            kernel_config, d['learning_mode'], d['add_mode'], 10,
            d['add_size'], d['max_size'], d['search_size'], d['pool_size'],
            d['result_dir'], d['Learner'], d['test_X'], d['test_Y'],
            d['test_id'], d['optimizer'], d['stride'], d['seed'])
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
