#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdChemReactions
tqdm.pandas()
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from chemml.graph.hashgraph import HashGraph
from chemml.graph.from_rdkit import rdkit_config
from chemml.kernels.MultipleKernel import _get_uniX
from chemml.graph.substructure import AtomEnvironment


def unify_datatype(X):
    if X[0].__class__ == list:
        graphs = []
        for x in X:
            graphs += x[::2]
        HashGraph.unify_datatype(graphs, inplace=True)
    else:
        HashGraph.unify_datatype(X, inplace=True)


def get_df(csv, pkl, single_graph, multi_graph, reaction_graph):
    def single2graph(series):
        unique_series = _get_uniX(series)
        graphs = list(map(HashGraph.from_inchi_or_smiles, unique_series,
                          [rdkit_config()] * len(unique_series),
                          series['group_id']))
        unify_datatype(graphs)
        idx = np.searchsorted(unique_series, series)
        return np.asarray(graphs)[idx]

    def multi_graph_transform(line, hash):
        hashs = [str(hash) + '_%d' % i for i in range(int(len(line)/2))]
        line[::2] = list(map(HashGraph.from_inchi_or_smiles, line[::2],
                             [rdkit_config()] * int(len(line) / 2),
                             hashs))
        return line

    def reaction2agent(reaction_smarts, hash):
        agents = []
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        # print(line)
        for i, mol in enumerate(rxn.GetAgents()):
            Chem.SanitizeMol(mol)
            try:
                hash_ = hash + '_%d' % i
                config_ = rdkit_config()
                agents += [HashGraph.from_rdkit(mol, config_, hash_), 1.0]
            except:
                print(reaction_smarts)
                exit(0)
        return agents

    def reaction2rp(reaction_smarts, hash):
        reaction = []
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)

        # rxn.Initialize()
        def getAtomMapDict(mols):
            AtomMapDict = dict()
            for mol in mols:
                Chem.SanitizeMol(mol)
                for atom in mol.GetAtoms():
                    AMN = atom.GetPropsAsDict().get('molAtomMapNumber')
                    if AMN is not None:
                        AtomMapDict[AMN] = AtomEnvironment(
                            mol, atom, depth=1)
            return AtomMapDict

        def getReactingAtoms(rxn):
            ReactingAtoms = []
            reactantAtomMap = getAtomMapDict(rxn.GetReactants())
            productAtomMap = getAtomMapDict(rxn.GetProducts())
            for id, AE in reactantAtomMap.items():
                if AE != productAtomMap.get(id):
                    ReactingAtoms.append(id)
            return ReactingAtoms

        ReactingAtoms = getReactingAtoms(rxn)
        for i, reactant in enumerate(rxn.GetReactants()):
            Chem.SanitizeMol(reactant)
            hash_ = hash + '_r%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(reactant, config_, hash_), 1.0]
            if True not in reaction[-2].nodes.to_pandas()['group_reaction']:
                raise Exception('Reactants error:', reaction_smarts)
        for i, product in enumerate(rxn.GetProducts()):
            Chem.SanitizeMol(product)
            hash_ = hash + '_p%d' % i
            config_ = rdkit_config(reaction_center=ReactingAtoms)
            reaction += [HashGraph.from_rdkit(product, config_, hash_), -1.0]
            if True not in reaction[-2].nodes.to_pandas()['group_reaction']:
                raise Exception('Products error:', reaction_smarts)
        return reaction

    if pkl is not None and os.path.exists(pkl):
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(csv, sep='\s+', header=0)
        if 'id' not in df:
            df['id'] = df.index + 1
            df['group_id'] = df['id']
        else:
            groups = df.groupby(single_graph + multi_graph + reaction_graph)
            df['group_id'] = 0
            for g in groups:
                g[1]['group_id'] = int(g[1]['id'].min())
                df.update(g[1])
            df['id'] = df['id'].astype(int)
            df['group_id'] = df['group_id'].astype(int)
        for sg in single_graph:
            print('Processing single graph.')
            if len(np.unique(df[sg])) > 0.5 * len(df[sg]):
                df[sg] = df.progress_apply(
                    lambda x: HashGraph.from_inchi_or_smiles(
                        x[sg], rdkit_config(), str(x['group_id'])), axis=1)
                unify_datatype(df[sg])
            else:
                df[sg] = single2graph(df[sg])
        for mg in multi_graph:
            print('Processing multi graph.')
            df[mg] = df.progress_apply(
                lambda x: multi_graph_transform(
                    x[mg], str(x['group_id'])), axis=1)
            unify_datatype(df[mg])

        for rg in reaction_graph:
            print('Processing reagents graph.')
            print(df[rg])
            df[rg + '_agents'] = df.progress_apply(
                lambda x: reaction2agent(x[rg], str(x['group_id'])), axis=1)
            unify_datatype(df[rg + '_agents'])
            print('Processing reactions graph.')
            df[rg] = df.progress_apply(
                lambda x: reaction2rp(x[rg], str(x['group_id'])), axis=1)
            unify_datatype(df[rg])
        if pkl is not None:
            df.to_pickle(pkl)
    return df


def set_graph_property(input_config):
    single_graph, multi_graph, r_graph, properties = input_config.split(':')
    single_graph = single_graph.split(',') if single_graph else []
    multi_graph = multi_graph.split(',') if multi_graph else []
    reaction_graph = r_graph.split(',') if r_graph else []
    properties = properties.split(',')
    return single_graph, multi_graph, reaction_graph, properties


def main():
    parser = argparse.ArgumentParser(
        description='Transform input file into pickle file, in which the InChI '
                    'or SMILES string was transformed into graphs.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The output directory.',
    )
    parser.add_argument(
        '-i', '--input', type=str, help='Input data in csv format.'
    )
    parser.add_argument(
        '--input_config', type=str, help='Columns in input data.\n'
                                         'format: single_graph:multi_graph:reaction_graph:targets\n'
                                         'examples: inchi:::tt\n'
    )
    args = parser.parse_args()

    # set result directory
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    single_graph, multi_graph, reaction_graph, properties = \
        set_graph_property(args.input_config)

    # set kernel_config
    get_df(args.input,
           os.path.join(result_dir, '%s.pkl' % ','.join(properties)),
           single_graph, multi_graph, reaction_graph)


if __name__ == '__main__':
    main()
