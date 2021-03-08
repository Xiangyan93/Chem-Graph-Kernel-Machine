#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
# from rxnmapper import RXNMapper
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '../..'))
from chemml.graph.molecule.reaction import *


def parse_reaction(reaction_smarts):
    try:
        RxnFromSmarts(reaction_smarts, Assign=False, canonical=False)
        return True
    except:
        return False

"""
def remap_reaction(reaction_smarts, confidence=0.0):
    unmapped_reaction_smarts = GetUnmappedReactionSmarts(reaction_smarts)
    try:
        result = rxn_mapper.get_attention_guided_atom_maps(
            [unmapped_reaction_smarts])
        if result[0]['confidence'] > confidence:
            return result[0]['mapped_rxn']
        else:
            return 'Unconfident Mapping'
    except:
        return 'Mapping Error'
"""

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Preprocessing chemical reactions.',
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
        '--mapping', action='store_true',
        help='Mapping the atomic labels during the reaction.',
    )
    parser.add_argument(
        '--mapping_config', type=str,
        help='format: algorithm:confidence_cutoff.\n'
             'examples:\n'
             'rxnmapper:0.95'
    )
    parser.add_argument(
        '-n', '--n_jobs', type=int, default=1,
        help='The cpu numbers for parallel computing.'
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\s+')
    # Delete repeated reactions.
    dfs = []
    for g in df.groupby(args.reaction):
        dfs.append(g[1].sample(1))
    df_ = pd.concat(dfs)
    print('%d reactions are repeated.' % (len(df) - len(df_)))
    df = df_.sort_index().reset_index().drop(columns=['index'])

    print('\nFinding parsing error reactions')
    # Delete parsing error reactions, using RDKit.
    # df['error'] = ~df[args.reaction].progress_apply(parse_reaction)
    df['error'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(parse_reaction)(df.iloc[i][args.reaction])
        for i in df.index)
    df['error'] = ~df['error']
    n_parse_error = df['error'].value_counts().get(True)
    if n_parse_error is not None:
        print('%d reactions cannot be correctly parsed by RDKit. Saved in '
              'parse_error.csv' % n_parse_error)
        df[df['error'] == True].copy().drop(columns=['error']).\
            to_csv('parse_error.csv', sep=' ', index=False)
        df = df[df['error'] == False]. \
            copy().sort_index().reset_index().drop(columns=['index'])

    # Delete trivial reactions.
    print('\nFinding trivial reactions.')
    df['error'] = Parallel(
        n_jobs=args.n_jobs, verbose=True,
        **_joblib_parallel_args(prefer='processes'))(
        delayed(IsTrivialReaction)(df.iloc[i][args.reaction])
        for i in df.index)
    n_trivial = df['error'].value_counts().get(True)
    if n_trivial is not None:
        print('%d reactions are trivial. Saved in trivial.csv' % n_trivial)
        df[df['error'] == True].copy().drop(columns=['error']).\
            to_csv('trivial.csv', sep=' ', index=False)
        df = df[df['error'] == False]. \
            copy().sort_index().reset_index().drop(columns=['index'])

    if args.mapping:
        mapping_rule, confidence = args.mapping_config.split(':')
        confidence = float(confidence)
        if mapping_rule == 'rxnmapper':
            df[args.reaction] = df[args.reaction].progress_apply(
                remap_reaction, confidence=confidence)
            print('rxnmapper mapping %d reactions with error.' %
                  (df[args.reaction] == 'Mapping error').sum())
            print('rxnmapper mapping %d reactions with confidence < %f' %
                  ((df[args.reaction] == 'Unconfident Mapping').sum(),
                   confidence))
            df = df[~df[args.reaction].isin(['Mapping error',
                                             'Unconfident Mapping'])]
        else:
            raise RuntimeError(f'Unknown mapping algorithm: {mapping_rule}')
    print('The sanitized reactions are saved in sanitize.csv')
    df.drop(columns=['error']).to_csv('sanitize.csv', sep=' ', index=False)


if __name__ == '__main__':
    # rxn_mapper = RXNMapper()
    main()
