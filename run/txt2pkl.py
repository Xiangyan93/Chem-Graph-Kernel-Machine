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


def get_df(csv, pkl, single_graph, multi_graph, reaction_graph):
    def single2graph(series):
        unique_series = _get_uniX(series)
        graphs = list(map(HashGraph.from_inchi_or_smiles, unique_series,
                          [rdkit_config()] * len(unique_series)))
        idx = np.searchsorted(unique_series, series)
        return np.asarray(graphs)[idx]

    def multi_graph_transform(line):
        line[::2] = list(map(HashGraph.from_inchi_or_smiles, line[::2],
                             [rdkit_config()] * int(len(line) / 2)))

    def reaction2agent(line):
        agents = []
        rxn = rdChemReactions.ReactionFromSmarts(line)
        # print(line)
        for mol in rxn.GetAgents():
            Chem.SanitizeMol(mol)
            try:
                agents += [HashGraph.from_rdkit(mol, rdkit_config()), 1.0]
            except:
                print(line)
                exit(0)
        return agents

    def reaction2rp(line):
        reaction = []
        rxn = rdChemReactions.ReactionFromSmarts(line)

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
            reaction += [HashGraph.from_rdkit(
                reactant, rdkit_config(reaction_center=ReactingAtoms)), 1.0]
            if reaction[-2].nodes.to_pandas()[
                'group_reaction'].unique().tolist() == [False]:
                raise Exception('Reactants error:', line)
        for i, product in enumerate(rxn.GetProducts()):
            Chem.SanitizeMol(product)
            reaction += [HashGraph.from_rdkit(
                product, rdkit_config(reaction_center=ReactingAtoms)), -1.0]
            if reaction[-2].nodes.to_pandas()[
                'group_reaction'].unique().tolist() == [False]:
                raise Exception('Products error:', line)
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
            if len(_get_uniX(df[sg])) > 0.5 * len(df[sg]):
                df[sg] = df[sg].progress_apply(HashGraph.from_inchi_or_smiles,
                                               args=[rdkit_config()])
            else:
                df[sg] = single2graph(df[sg])
        for mg in multi_graph:
            print('Processing multi graph.')
            df[mg] = df[mg].progress_apply(multi_graph_transform)
        for rg in reaction_graph:
            print('Processing reagents graph.')
            df[rg + '_agents'] = df[rg].progress_apply(reaction2agent)
            print('Processing reactions graph.')
            df[rg] = df[rg].progress_apply(reaction2rp)
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
