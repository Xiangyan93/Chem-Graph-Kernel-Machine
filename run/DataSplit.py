#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from tap import Tap
from typing import Dict, Iterator, List, Optional, Union, Literal, Tuple
from random import Random
import pandas as pd


class DataSplitArgs(Tap):
    split_type: Literal['random', 'scaffold_balanced', 'loocv'] = 'random'
    """Method of splitting the data into train/val/test."""
    split_sizes: List[float] = [0.8, 0.2]
    """Split proportions for train/validation/test sets."""
    split_tags: List[str] = ['train', 'test']
    """"""
    num_folds: int = 1
    """Number of folds when performing cross validation."""
    save_dir: str
    """The output directory."""
    n_jobs: int = 1
    """The cpu numbers used for parallel computing."""
    data_path: str = None
    """The Path of input data CSV file."""

    @property
    def N_data(self) -> int:
        return len(self.df)

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.df = pd.read_csv(self.data_path)


def main(args: DataSplitArgs) -> None:
    df = pd.read_csv(args.data_path)
    assert sum(args.split_sizes) == 1
    for seed in range(args.num_folds):
        random = Random(seed)
        if args.split_type == 'random':
            indices = list(range(args.N_data))
            random.shuffle(indices)
            end = 0
            for i, size in enumerate(args.split_sizes):
                start = end
                end = start + int(size * args.N_data)
                df_split = df[df.index.isin(indices[start:end])]
                save_file = os.path.join(args.save_dir,
                                         'random_%s_size%d_seed%d.csv' % (args.split_tags[i], len(df_split), seed))
                df_split.to_csv(save_file, index=False)
        else:
            pass


if __name__ == '__main__':
    main(args=DataSplitArgs().parse_args())
