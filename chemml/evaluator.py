#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
from .args import TrainArgs, ActiveLearningArgs
from .data import Dataset
from .models.regression.GPRgraphdot import GPR, LRAGPR
from .models.classification import GPC
from .models.classification import SVC
from .models.regression import ConsensusRegressor
from .kernels import PreCalcKernel, PreCalcKernelConfig


class Evaluator:
    def __init__(self, args: TrainArgs,
                 dataset: Dataset,
                 kernel_config):
        self.args = args
        self.dataset = dataset
        self.kernel_config = kernel_config
        self.kernel = kernel_config.kernel
        self.set_model(args)

    def evaluate(self):
        # Leave-One-Out cross validation
        if self.args.split_type == 'loocv':
            return self._evaluate_loocv()

        # Transform graph kernel to preCalc kernel.
        if self.args.num_folds != 1 and self.kernel.__class__ != PreCalcKernel \
                and self.args.graph_kernel_type == 'graph':
            self.make_kernel_precalc()

        # Initialization
        train_results = dict()
        test_results = dict()
        for metric in self.args.metrics:
            train_results[metric] = []
            test_results[metric] = []

        for i in range(self.args.num_folds):
            # data splits
            dataset_train, dataset_test = self.dataset.split(
                self.args.split_type,
                self.args.split_sizes,
                seed=self.args.seed + i)
            train_metrics, test_metrics = self._evaluate_train_test(dataset_train, dataset_test,
                                                                    test_log='test_%d.log' % i)
            for j, metric in enumerate(self.args.metrics):
                if self.args.evaluate_train:
                    train_results[metric].append(train_metrics[j])
                test_results[metric].append(test_metrics[j])

        if self.args.evaluate_train:
            print('\nTraining set:')
            for metric, result in train_results.items():
                print(metric,
                      ': %.5f +/- %.5f' % (np.nanmean(result), np.nanstd(result)))
                # print(np.asarray(result).ravel())
        print('\nTest set:')
        for metric, result in test_results.items():
            print(metric, ': %.5f +/- %.5f' % (np.nanmean(result), np.nanstd(result)))
        return np.nanmean(test_results[self.args.metric])

    def _evaluate_train_test(self, dataset_train: Dataset,
                             dataset_test: Dataset,
                             test_log: str = 'test.log') -> Tuple[List[float], List[float]]:
        X_train = dataset_train.X
        y_train = dataset_train.y
        repr_train = dataset_train.repr.ravel()
        X_test = dataset_test.X
        y_test = dataset_test.y
        repr_test = dataset_test.repr.ravel()
        # Find the most similar sample in training sets.
        if self.args.detail:
            y_similar = self.get_similar_info(X_test, X_train, repr_train, 5)
        else:
            y_similar = None

        train_metrics = []
        test_metrics = []
        if self.args.dataset_type == 'regression':
            self.model.fit(X_train, y_train, loss=self.args.loss, verbose=True)
            y_pred, y_std = self.model.predict(X_test, return_std=True)
            # save results test_0.log
            self._output_df(df=pd.DataFrame({
                'target': y_test.tolist(),
                'predict': y_pred.tolist(),
                'uncertainty': y_std.tolist(),
                'repr': repr_test}), y_similar=y_similar). \
                to_csv('%s/%s' % (self.args.save_dir, test_log), sep='\t',
                       index=False, float_format='%15.10f')
            # save results metric
            for metric in self.args.metrics:
                test_metrics.append(
                    self._evaluate(y_test, y_pred, metric))

            if self.args.evaluate_train:
                y_pred, y_std = self.model.predict(X_train, return_std=True)
                self._output_df(df=pd.DataFrame({
                    'target': y_train.tolist(),
                    'predict': y_pred.tolist(),
                    'uncertainty': y_std.tolist(),
                    'repr': repr_train})). \
                    to_csv('%s/%s' % (self.args.save_dir, test_log.replace('test', 'train')), sep='\t',
                           index=False, float_format='%15.10f')
                for metric in self.args.metrics:
                    train_metrics.append(
                        self._evaluate(y_train, y_pred, metric))
        else:
            self.model.fit(X_train, y_train)
            if self.args.no_proba:
                y_pred = self.model.predict(X_test)
            else:
                y_pred = self.model.predict_proba(X_test)
            self._output_df(df=pd.DataFrame({
                'target': y_test.tolist(),
                'predict': y_pred.tolist(),
                'repr': repr_test}), y_similar=y_similar). \
                to_csv('%s/%s' % (self.args.save_dir, test_log), sep='\t',
                       index=False, float_format='%15.10f')
            for metric in self.args.metrics:
                test_metrics.append(
                    self._evaluate(y_test, y_pred, metric))

            if self.args.evaluate_train:
                y_pred = self.model.predict(X_train)
                for metric in self.args.metrics:
                    train_metrics.append(
                        self._evaluate(y_train, y_pred, metric))
        return train_metrics, test_metrics

    def _evaluate_loocv(self):
        X, y, X_repr = self.dataset.X, self.dataset.y, self.dataset.X_repr.ravel()
        if self.args.optimizer is not None:
            self.model.fit(X, y, loss='loocv', verbose=True)
        y_pred, y_std = self.model.predict_loocv(X, y, return_std=True)
        print('LOOCV:')
        for metric in self.args.metrics:
            print('%s: %.5f' % (metric, self._evaluate(y, y_pred, metric)))
        if self.args.detail:
            y_similar = self.get_similar_info(X, X, X_repr, 5)
        else:
            y_similar = None
        self._output_df(df=pd.DataFrame({
            'target': y.tolist(),
            'predict': y_pred.tolist(),
            'uncertainty': y_std.tolist()}), y_similar=y_similar).to_csv(
            '%s/loocv.log' % self.args.save_dir, sep='\t', index=False,
            float_format='%15.10f')
        return self._evaluate(y, y_pred, self.args.metric)

    def get_similar_info(self, X, X_train, X_repr, n_most_similar):
        K = self.kernel(X, X_train)
        assert (K.shape == (len(X), len(X_train)))
        similar_info = []
        kindex = self.get_most_similar_graphs(K, n=n_most_similar)
        for i, index in enumerate(kindex):
            def round5(x):
                return ',%.5f' % x

            k = list(map(round5, K[i][index]))
            repr = np.asarray(X_repr)[index]
            info = ';'.join(list(map(str.__add__, repr, k)))
            similar_info.append(info)
        return similar_info

    @staticmethod
    def get_most_similar_graphs(K, n=5):
        return np.argsort(-K)[:, :min(n, K.shape[1])]

    def make_kernel_precalc(self):
        self.kernel_config = self.kernel_config.get_preCalc_kernel_config(self.args, self.dataset)
        self.kernel = self.kernel_config.kernel

    def set_model(self, args: TrainArgs):
        if args.model_type == 'gpr':
            self.model = GPR(
                kernel=self.kernel,
                optimizer=args.optimizer,
                alpha=args.alpha_,
                normalize_y=True,
                batch_size=args.batch_size
            )
            if args.ensemble:
                self.model = ConsensusRegressor(
                    self.model,
                    n_estimators=args.n_estimator,
                    n_sample_per_model=args.n_sample_per_model,
                    n_jobs=args.n_jobs,
                    consensus_rule=args.ensemble_rule
                )
        elif args.model_type == 'gpr_nystrom':
            self.model = LRAGPR(
                kernel=self.kernel,
                optimizer=args.optimizer,
                alpha=args.alpha_,
                normalize_y=True
            )
        elif args.model_type == 'gpc':
            self.model = GPC(
                kernel=self.kernel,
                optimizer=args.optimizer,
                n_jobs=args.n_jobs
            )
        elif args.model_type == 'svc':
            self.model = SVC(
                kernel=self.kernel,
                C=args.C_,
                probability=True
            )
        else:
            raise RuntimeError(f'Unsupport model:{args.model_type}')

    @staticmethod
    def _output_df(df: pd.DataFrame, y_similar: List = None):
        if y_similar is not None:
            df['y_similar'] = y_similar
        return df

    def _evaluate(self, y, y_pred, metrics):
        if y.ndim == 2 and y_pred.ndim == 2:
            num_tasks = y.shape[1]
            results = []
            for i in range(num_tasks):
                results.append(self._metric_func(y[:, i], y_pred[:, i],
                                                      metrics))
            return np.nanmean(results)
        else:
            return self._metric_func(y, y_pred, metrics)

    def _metric_func(self, y, y_pred, metrics):
        # y_pred has nan may happen when train_y are all 1 or 0.
        if y_pred.dtype != object and True in np.isnan(y_pred):
            return np.nan
        # y may be unlabeled in some index. Select index of labeled data.
        if y.dtype == float:
            idx = ~np.isnan(y)
            y = y[idx]
            y_pred = y_pred[idx]
        if self.args.dataset_type == 'classification':
            if 0 not in y or 1 not in y:
                return np.nan

        if metrics == 'roc-auc':
            return roc_auc_score(y, y_pred)
        elif metrics == 'accuracy':
            return accuracy_score(y, y_pred)
        elif metrics == 'precision':
            return precision_score(y, y_pred, average='macro')
        elif metrics == 'recall':
            return recall_score(y, y_pred, average='macro')
        elif metrics == 'f1_score':
            return f1_score(y, y_pred, average='macro')
        elif metrics == 'precision':
            return precision_score(y, y_pred, average='macro')
        elif metrics == 'r2':
            return r2_score(y, y_pred)
        elif metrics == 'mae':
            return mean_absolute_error(y, y_pred)
        elif metrics == 'mse':
            return mean_squared_error(y, y_pred)
        elif metrics == 'rmse':
            return np.sqrt(self._metric_func(y, y_pred, 'mse'))
        else:
            raise RuntimeError(f'Unsupported metrics {metrics}')


class ActiveLearner:
    def __init__(self, args: ActiveLearningArgs,
                 dataset_train: Dataset,
                 dataset_pool: Dataset,
                 kernel_config, kernel_config_surrogate):
        self.args = args
        self.evaluator = Evaluator(args, dataset_train, kernel_config)
        self.surrogate = Evaluator(args, dataset_train, kernel_config_surrogate)
        self.kernel = kernel_config_surrogate.kernel
        self.dataset = dataset_train
        self.dataset_pool = dataset_pool
        self.max_uncertainty = 1.
        self.log_df = pd.DataFrame({'training_size': []})
        for metric in args.metrics:
            self.log_df[metric] = []

    @property
    def current_size(self):
        return len(self.dataset)

    def run(self) -> None:
        while True:
            print('***\tStart: active learning, current size = %i\t***\n' %
                  self.current_size)
            print('**\tStart train\t**\n')
            if self.current_size % self.args.evaluate_stride == 0:
                print('\n**\tstart evaluate\t**\n')
                self.evaluate()
                print('\n**\tend evaluate\t**\n')
            else:
                self.train()
            print('**\tadding samples**\n')
            if self.stop():
                break
            self.add_sample()
        if self.args.save_dir is not None:
            self.log_df.to_csv('%s/active_learning.log' % self.args.save_dir,
                               sep='\t', index=False, float_format='%15.10f')
            pd.DataFrame({'smiles': self.dataset.X_repr.ravel()}).to_csv('%s/training_smiles.csv' % self.args.save_dir,
                                                                 index=False)
        self.evaluate()

        print('\n***\tEnd: active learning\t***\n')

    def stop(self) -> bool:
        # stop active learning when reach stop size.
        if len(self.dataset) >= self.args.stop_size:
            return True
        # stop active learning when pool data set is empty.
        elif len(self.dataset_pool) == 0:
            return True
        elif self.args.stop_uncertainty is not None and self.max_uncertainty < self.args.stop_uncertainty:
            return True
        else:
            return False

    def train(self):
        X_train, y_train = self.dataset.X, self.dataset.y
        self.evaluator.model.fit(X_train, y_train, loss=self.args.loss, verbose=True)
        self.surrogate.model.fit(X_train, y_train, loss=self.args.loss, verbose=True)

    def evaluate(self):
        train_metrics, test_metrics = self.evaluator._evaluate_train_test(
            self.dataset, self.dataset_pool, test_log='test_active_%d.log' % self.current_size)
        self.log_df.loc[len(self.log_df)] = [self.current_size] + test_metrics
        for i, metric in enumerate(self.args.metrics):
            print('%s: %.5f' % (metric, test_metrics[i]))

    def add_sample(self):
        pool_idx = self.pool_idx()
        X, y = self.dataset_pool.X[pool_idx], self.dataset_pool.y[pool_idx]
        if self.args.learning_algorithm == 'supervised':
            y_pred = self.surrogate.model.predict(X)
            y_abse = abs(y_pred - y)
            add_idx = self._get_add_samples_idx(y_abse, pool_idx)
        elif self.args.learning_algorithm == 'unsupervised':
            y_pred, y_std = self.surrogate.model.predict(X, return_std=True)
            self.max_uncertainty = y_std.max()
            print('Add sample with maximum uncertainty: %f' % self.max_uncertainty)
            add_idx = self._get_add_samples_idx(y_std, pool_idx)
        elif self.args.learning_algorithm == 'random':
            if len(pool_idx) < self.args.add_size:
                add_idx = pool_idx
            else:
                add_idx = np.random.choice(pool_idx, self.args.add_size, replace=False)
        else:
            raise Exception
        for i in sorted(add_idx, reverse=True):
            self.dataset.data.append(self.dataset_pool.data.pop(i))

    def pool_idx(self) -> List[int]:
        idx = np.arange(len(self.dataset_pool))
        if self.args.pool_size is not None and self.args.pool_size < len(self.dataset_pool):
            return np.random.choice(idx, self.args.pool_size, replace=False)
        else:
            return idx

    def _get_add_samples_idx(self, error: List[float], pool_idx: List[int]) -> List[int]:
        # add all if the pool set is small.
        if len(error) < self.args.add_size:
            return pool_idx
        elif self.args.sample_add_algorithm == 'random':
            return np.random.choice(pool_idx, self.args.add_size, replace=False)
        elif self.args.sample_add_algorithm == 'cluster':
            cluster_idx = self.__get_worst_idx(error, pool_idx)
            K = self.kernel(self.dataset_pool.X[cluster_idx])
            add_idx = self.__find_distant_samples(K)
            return np.array(cluster_idx)[add_idx]
        elif self.args.sample_add_algorithm == 'nlargest':
            return self.__get_worst_idx(error, pool_idx)

    def __find_distant_samples(self, gram_matrix: List[List[float]]) -> List[int]:
        """Find distant samples from a pool using clustering method.

        Parameters
        ----------
        gram_matrix: gram matrix of the samples.

        Returns
        -------
        List of idx
        """
        embedding = SpectralEmbedding(
            n_components=self.args.add_size,
            affinity='precomputed'
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=self.args.add_size,
            # random_state=self.args.seed
        ).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(self.args.add_size)])
        total_distance = defaultdict(
            dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(
                embedding[i] - np.delete(center, cluster_class, axis=0))).sum(
                axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(
                       self.args.add_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def __get_worst_idx(self, error: List[float], pool_idx: List[int]) -> List[int]:
        if self.args.cluster_size == 0 or len(error) < self.args.cluster_size:
            return pool_idx
        else:
            return pool_idx[np.argsort(error)[-self.args.cluster_size:]]
