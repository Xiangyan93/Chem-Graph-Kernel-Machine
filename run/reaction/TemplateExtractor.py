#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '../..'))
from chemml.graph.molecule.RxnTemplate import *


def rxn2reactants(rxn):
    reactants = rxn.GetReactants()
    for mol in reactants:
        RemoveAtomMap(mol)
    return '.'.join(list(map(Chem.MolToSmiles, reactants)))


def rxn2products(rxn):
    products = rxn.GetProducts()
    for mol in products:
        RemoveAtomMap(mol)
    return '.'.join(list(map(Chem.MolToSmiles, products)))


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
    print('\nCounting the number of reactants')
    df['rxn'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(RxnFromSmarts)(df.iloc[i][args.reaction])
        for i in df.index)
    df['reactants'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(rxn2reactants)(df.iloc[i]['rxn'])
        for i in df.index)
    df['products'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(rxn2products)(df.iloc[i]['rxn'])
        for i in df.index)
    df['N_reactants'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(lambda x: len(x.GetReactants()))(df.iloc[i]['rxn'])
        for i in df.index)
    df['N_products'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(lambda x: len(x.GetProducts()))(df.iloc[i]['rxn'])
        for i in df.index)
    df.drop(columns=['rxn']).to_csv('reaction.csv', sep=' ', index=False)


if __name__ == '__main__':
    main()
