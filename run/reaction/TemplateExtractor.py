#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '../..'))
from chemml.graph.molecule.RxnTemplate import *


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract chemical reactions templates.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--reaction', type=str, help='Columns in input data that contains'
                                     'reaction smarts.\n'
    )
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=1,
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\s+')
    print('\nExtracting reaction templates')
    df['template'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(ExtractReactionTemplate)(df.iloc[i][args.reaction])
        for i in df.index)
    df.to_csv('reaction.csv', sep=' ', index=False)


if __name__ == '__main__':
    tqdm.pandas()
    main()
