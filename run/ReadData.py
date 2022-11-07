#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from mgktools.data import Dataset
import pandas as pd
from chemml import CommonArgs


def main(args: CommonArgs) -> None:
    print('Preprocessing Dataset.')
    if args.data_path is not None:
        dataset = Dataset.from_df(df=pd.read_csv(args.data_path),
                                  pure_columns=args.pure_columns,
                                  mixture_columns=args.mixture_columns,
                                  reaction_columns=args.reaction_columns,
                                  feature_columns=args.feature_columns,
                                  target_columns=args.target_columns,
                                  features_generator=args.features_generator,
                                  features_combination=args.features_combination,
                                  mixture_type=args.mixture_type,
                                  reaction_type=args.reaction_type,
                                  group_reading=args.group_reading,
                                  n_jobs=args.n_jobs)
    else:
        assert args.data_public is not None
        # TODO
        raise ValueError('Public datasets are not available.')
    if args.features_mol_normalize:
        dataset.normalize_features_mol()
    if args.features_add_normalize:
        dataset.normalize_features_add()
    dataset.save(args.save_dir, overwrite=True)
    print('Preprocessing Dataset Finished.')


if __name__ == '__main__':
    main(args=CommonArgs().parse_args())
