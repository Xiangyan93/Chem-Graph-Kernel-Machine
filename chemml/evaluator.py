#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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
from chemml import TrainArgs
from chemml.data import Dataset
from chemml.models.regression.GPRgraphdot import GPR, LRAGPR
from chemml.models.classification.gpc import GPC
from chemml.models.classification.svm import SVC
from chemml.models.regression.consensus import ConsensusRegressor


class Evaluator:
    def __init__(self, args: TrainArgs,
                 dataset: Dataset,
                 kernel):
        self.args = args
        self.dataset = dataset
        self.kernel = kernel
        self.set_model(args)

    def evaluate(self):
        if self.args.split_type == 'loocv':
            X, y, gid = self.dataset.X, self.dataset.y, self.dataset.X_gid
            y_pred, y_std = self.model.predict_loocv(X, y, return_std=True)
            print('LOOCV:')
            for metric in self.args.metrics:
                print('%s: %.5f' % (metric, self._evaluate(y, y_pred, metric)))
            self._df(target=y, predict=y_pred, uncertainty=y_std).to_csv(
                '%s/loocv.log' % self.args.save_dir, sep='\t', index=False,
                float_format='%15.10f')
            return self._evaluate(y, y_pred, self.args.metric)

        train_dict = dict()
        test_dict = dict()
        for metric in self.args.metrics:
            train_dict[metric] = []
            test_dict[metric] = []

        for i in range(self.args.num_folds):
            dataset_train, dataset_test = self.dataset.split(
                self.args.split_type,
                self.args.split_sizes,
                seed=self.args.seed + i)
            X_train, y_train = dataset_train.X, dataset_train.y.ravel()
            X_test, y_test, gid = dataset_test.X, dataset_test.y.ravel(), dataset_test.X_gid

            if self.args.dataset_type == 'regression':
                self.model.fit(X_train, y_train, loss=self.args.loss,
                               verbose=True)
                y_pred, y_std = self.model.predict(X_test, return_std=True)
                self._df(target=y_test, predict=y_pred, uncertainty=y_std).\
                    to_csv('%s/test_%d.log' % (self.args.save_dir, i), sep='\t',
                    index=False, float_format='%15.10f')
                for metric in self.args.metrics:
                    test_dict[metric].append(
                        self._evaluate(y_test, y_pred, metric))

                if self.args.evaluate_train:
                    y_pred, y_std = self.model.predict(X_train, return_std=True)
                    for metric in self.args.metrics:
                        train_dict[metric].append(
                            self._evaluate(y_train, y_pred, metric))
            else:
                self.model.fit(X_train, y_train)
                y_pred, y_std = self.model.predict(X_test)
                self._df(target=y_test, predict=y_pred, uncertainty=y_std).\
                    to_csv('%s/test_%d.log' % (self.args.save_dir, i), sep='\t',
                    index=False, float_format='%15.10f')
                for metric in self.args.metrics:
                    test_dict[metric].append(
                        self._evaluate(y_test, y_pred, metric))
        if self.args.evaluate_train:
            for metric, result in train_dict.items():
                print(metric,
                      ': %.5f +/- %.5f' % (np.mean(result), np.std(result)))
                # print(np.asarray(result).ravel())
        for metric, result in test_dict.items():
            print(metric, ': %.5f +/- %.5f' % (np.mean(result), np.std(result)))
        return np.mean(test_dict[self.args.metric])

    def set_model(self, args: TrainArgs):
        if args.model_type == 'gpr':
            self.model = GPR(
                kernel=self.kernel,
                optimizer=args.optimizer,
                alpha=args.alpha,
                normalize_y=True
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
                alpha=args.alpha,
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
                C=args.C
            )
        else:
            raise RuntimeError(f'Unsupport model:{args.model_type}')

    @staticmethod
    def _df(**kwargs):
        return pd.DataFrame(kwargs)

    @staticmethod
    def _evaluate(y, y_pred, metrics):
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
            return np.sqrt(Evaluator._evaluate(y, y_pred, 'mse'))
        else:
            raise RuntimeError(f'Unsupported metrics {metrics}')
