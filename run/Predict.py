#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
import numpy as np
import pandas as pd
from chemml.args import PredictArgs
from chemml.data.data import Dataset
from chemml.kernels.utils import get_kernel_config
from chemml.model import Evaluator


def main(args: PredictArgs) -> None:
    dataset = Dataset.load(args.save_dir, args=args)
    dataset_predict = Dataset.from_df(args, pd.read_csv(args.test_path))
    dataset_predict.graph_kernel_type = 'graph'
    X_graph = np.concatenate([dataset.X_graph, dataset_predict.X_graph])
    dataset.unify_datatype(X_graph)
    kernel_config = get_kernel_config(args, dataset)
    evaluator = Evaluator(args, dataset, kernel_config)
    evaluator.model.load(args.save_dir)
    _, results = evaluator.evaluate_train_test(dataset, dataset_predict, test_log=args.preds_path)
    if results is not None:
        for i, metric in enumerate(args.metrics):
            print(metric, ': %.5f' % results[i])


if __name__ == '__main__':
    main(args=PredictArgs().parse_args())
