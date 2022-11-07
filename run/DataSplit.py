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
from mgktools.data.split import data_split_index


class DataSplitArgs(Tap):
    split_type: Literal['random', 'scaffold_order', 'scaffold_random', 'stratified', 'n_heavy'] = 'random',
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
    n_cv: int = None
    """n-fold cross-validation"""
    data_path: str = None
    """The Path of input data CSV file."""
    smiles_column: str = None
    """The smiles column of the CSV file, used for scaffold split"""
    target_column: str = None
    """The smiles column of the CSV file, used for scaffold split"""
    n_heavy_cutoff: int = None

    @property
    def N_data(self) -> int:
        return len(self.df)

    def process_args(self) -> None:
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.df = pd.read_csv(self.data_path)

        if self.n_heavy_cutoff is not None:
            assert self.split_type == 'n_heavy'
            assert self.num_folds == 1
        if self.smiles_column is not None:
            assert self.split_type in ['scaffold_order', 'scaffold_random', 'n_heavy']
        if self.target_column is not None:
            assert self.split_type == 'stratified'
        if self.split_type == 'stratified':
            assert self.target_column is not None
        if self.n_cv is not None:
            assert self.split_type in ['random', 'stratified']


def main(args: DataSplitArgs) -> None:
    assert sum(args.split_sizes) == 1
    for seed in range(args.num_folds):
        if args.n_cv is None:
            sizes = args.split_sizes
            assert len(args.split_tags) == len(sizes)
            split_index = data_split_index(n_samples=len(args.df),
                                           mols=None if args.smiles_column is None else args.df[args.smiles_column],
                                           targets=None if args.target_column is None else args.df[args.target_column],
                                           split_type=args.split_type,
                                           sizes=sizes,
                                           n_heavy_cutoff=args.n_heavy_cutoff,
                                           seed=seed)
            for i, tag in enumerate(args.split_tags):
                df_split = args.df[args.df.index.isin(split_index[i])]
                save_file = os.path.join(args.save_dir,
                                         '%d_%s_%s_size%d.csv' % (seed, args.split_type, tag, len(df_split)))
                df_split.to_csv(save_file, index=False)
        else:
            # cross-validation
            sizes = [1 / args.n_cv] * args.n_cv
            split_index = data_split_index(n_samples=len(args.df),
                                           mols=None if args.smiles_column is None else args.df[args.smiles_column],
                                           targets=None if args.target_column is None else args.df[args.target_column],
                                           split_type=args.split_type,
                                           sizes=sizes,
                                           seed=seed)
            for i in range(args.n_cv):
                df_train = args.df[~args.df.index.isin(split_index[i])]
                df_test = args.df[args.df.index.isin(split_index[i])]
                save_file = os.path.join(args.save_dir,
                                         '%d_%s_cv%d_train_size%d.csv' % (seed, args.split_type, i, len(df_train)))
                df_train.to_csv(save_file, index=False)
                save_file = os.path.join(args.save_dir,
                                         '%d_%s_cv%d_test_size%d.csv' % (seed, args.split_type, i, len(df_test)))
                df_test.to_csv(save_file, index=False)


if __name__ == '__main__':
    main(args=DataSplitArgs().parse_args())
